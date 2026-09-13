"""DWPose (YOLOX-L detector + RTMPose-L whole-body) in ONNX Runtime.

Why DWPose: it is what the community ControlNet preprocessors (`controlnet_aux`) use to
make the pose images that the video ControlNets — MiniMax H3 Fun ControlNet among them —
were trained on. Producing the same 18-joint OpenPose layout, hands included, from the
same estimator means our skeleton video looks like the model's training data, which
matters more than a point of keypoint accuracy. Both the code (IDEA-Research/DWPose) and
the weights (`yzd-v/DWPose` on Hugging Face) are Apache-2.0, so commercial use is fine.

What was ruled out and why: OpenPose itself (CMU licence, non-commercial); Ultralytics
YOLO-pose (AGPL, incompatible with an Apache package); Sapiens (non-commercial);
MediaPipe (single person — useless for a fight).

Pre/post-processing follows the reference implementation in DWPose's ControlNet folder
(`annotator/dwpose/onnxdet.py`, `onnxpose.py`), rewritten here in NumPy with two
corrections: the detector gets BGR (YOLOX was trained on raw BGR pixels) and the pose
model gets RGB normalised with ImageNet mean/std (its mmpose config sets
`bgr_to_rgb=True`); the reference feeds the same array to both.

Speed: on a CPU the detector is ~80 % of the time (368 ms vs 75 ms per person for the pose
model, M2 Max, 864×496). So the detector runs every `detect_every` frames and in between
the boxes come from the previous frame's skeleton (padded) — standard pose tracking. Two
guards keep that honest: a cut (large frame-to-frame difference on a thumbnail) or a
person whose skeleton has gone unreliable forces a detection on that frame. Measured on
the 289-frame corridor demo, every-3 vs every-1: see the module test / changelog.

Runtime: ONNX Runtime on CPU, or its CUDA provider when `onnxruntime-gpu` is installed.
CoreML is not requested: RTMPose's SimCC heads are known to fall back op by op under the
CoreML provider, so plain CPU is the predictable choice on a Mac (not benchmarked by us).
"""

from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from . import PoseClip

log = logging.getLogger(__name__)

HF_REPO = "yzd-v/DWPose"
DET_FILE = "yolox_l.onnx"
POSE_FILE = "dw-ll_ucoco_384.onnx"
DET_INPUT = (640, 640)  # (h, w) the YOLOX-L export was traced at
DET_SCORE = 0.3  # keep a person box above this (after class score × objectness)
DET_NMS = 0.45
DET_PRESCORE = 0.1
POSE_BBOX_PADDING = 1.25  # mmpose top-down convention: enlarge the box before cropping
DETECT_EVERY = 3  # detector cadence; between detections boxes follow the previous skeleton
TRACK_BOX_PAD = 1.2  # a box derived from a skeleton is grown by this before cropping
CUT_THRESHOLD = 28.0  # mean |Δ| on a 64-px grey thumbnail above this = a cut → detect now
TRACK_MIN_SCORE = 0.5  # mean body-joint score below this = the skeleton is not to be trusted
IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
IMAGENET_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def _weights(checkpoint_dir: str | None) -> tuple[str, str]:
    """Paths of the two ONNX files: from `checkpoint_dir` if given, else the HF download
    (which honours `HF_ENDPOINT`, so a mirror works without special-casing)."""
    if checkpoint_dir:
        from pathlib import Path

        d = Path(checkpoint_dir)
        return str(d / DET_FILE), str(d / POSE_FILE)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(HF_REPO, DET_FILE), hf_hub_download(HF_REPO, POSE_FILE)


def _session(path: str, device: str):
    import onnxruntime as ort

    available = ort.get_available_providers()
    providers = ["CPUExecutionProvider"]
    if device == "cuda" and "CUDAExecutionProvider" in available:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(path, sess_options=opts, providers=providers)


# ── detector ──────────────────────────────────────────────────────────────────────


def _det_preprocess(img_bgr: np.ndarray) -> tuple[np.ndarray, float]:
    """Letterbox onto a grey 640×640 canvas (top-left aligned, as YOLOX does). Returns the
    CHW float32 tensor and the scale that maps canvas pixels back to the source."""
    h, w = img_bgr.shape[:2]
    r = min(DET_INPUT[0] / h, DET_INPUT[1] / w)
    nh, nw = int(h * r), int(w * r)
    canvas = np.full((DET_INPUT[0], DET_INPUT[1], 3), 114, dtype=np.uint8)
    canvas[:nh, :nw] = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(canvas.transpose(2, 0, 1), dtype=np.float32), r


def _det_grid_decode(out: np.ndarray) -> np.ndarray:
    """YOLOX raw head output `[1, A, 85]` → boxes in canvas pixels (cx, cy, w, h) + scores."""
    grids, strides_all = [], []
    for stride in (8, 16, 32):
        hs, ws = DET_INPUT[0] // stride, DET_INPUT[1] // stride
        xv, yv = np.meshgrid(np.arange(ws), np.arange(hs))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        strides_all.append(np.full((1, grid.shape[1], 1), stride))
    grid = np.concatenate(grids, 1)
    stride = np.concatenate(strides_all, 1)
    out = out.copy()
    out[..., :2] = (out[..., :2] + grid) * stride
    out[..., 2:4] = np.exp(out[..., 2:4]) * stride
    return out[0]


def _nms(boxes: np.ndarray, scores: np.ndarray, thr: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= thr]
    return keep


def detect_people(session, img_bgr: np.ndarray) -> np.ndarray:
    """Person boxes `[N, 4]` (x1, y1, x2, y2) in source pixels, best first. COCO class 0 only."""
    tensor, r = _det_preprocess(img_bgr)
    raw = session.run(None, {session.get_inputs()[0].name: tensor[None]})[0]
    pred = _det_grid_decode(raw)
    boxes = pred[:, :4]
    scores = pred[:, 4:5] * pred[:, 5:]  # objectness × class prob
    person = scores[:, 0]
    xyxy = np.empty_like(boxes)
    xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    xyxy /= r
    mask = person > DET_PRESCORE
    if not mask.any():
        return np.zeros((0, 4), dtype=np.float32)
    cand, cand_s = xyxy[mask], person[mask]
    keep = _nms(cand, cand_s, DET_NMS)
    boxes_k, scores_k = cand[keep], cand_s[keep]
    good = scores_k > DET_SCORE
    return boxes_k[good].astype(np.float32)


# ── pose (top-down, SimCC) ─────────────────────────────────────────────────────


def _box_to_center_scale(box: np.ndarray, aspect: float) -> tuple[np.ndarray, np.ndarray]:
    """mmpose `bbox_xyxy2cs` + `_fix_aspect_ratio`: centre and padded, aspect-fixed size."""
    x1, y1, x2, y2 = box
    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)
    w, h = (x2 - x1) * POSE_BBOX_PADDING, (y2 - y1) * POSE_BBOX_PADDING
    if w > h * aspect:
        h = w / aspect
    else:
        w = h * aspect
    return center, np.array([w, h], dtype=np.float32)


def _warp_matrix(center: np.ndarray, scale: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Affine that maps the padded box to the model's input rectangle (no rotation)."""
    src = np.zeros((3, 2), dtype=np.float32)
    src[0] = center
    src[1] = center + np.array([0.0, -scale[0] * 0.5])
    d = src[0] - src[1]
    src[2] = src[1] + np.array([-d[1], d[0]])
    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0] = (out_w * 0.5, out_h * 0.5)
    dst[1] = dst[0] + np.array([0.0, -out_w * 0.5])
    d = dst[0] - dst[1]
    dst[2] = dst[1] + np.array([-d[1], d[0]])
    return cv2.getAffineTransform(src, dst)


def _simcc_decode(simcc_x: np.ndarray, simcc_y: np.ndarray, split_ratio: float = 2.0):
    """SimCC heads `[1, K, W·r]`, `[1, K, H·r]` → `(xy [K, 2] in input px, score [K])`."""
    sx, sy = simcc_x[0], simcc_y[0]
    x = sx.argmax(axis=1).astype(np.float32)
    y = sy.argmax(axis=1).astype(np.float32)
    score = np.minimum(sx.max(axis=1), sy.max(axis=1))
    xy = np.stack((x, y), axis=-1) / split_ratio
    xy[score <= 0] = -1
    return xy, score


def estimate_pose(session, img_rgb: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Whole-body keypoints for every box: `([N, 133, 2] pixels, [N, 133] scores)`."""
    _, _, in_h, in_w = session.get_inputs()[0].shape
    in_h, in_w = int(in_h), int(in_w)
    name = session.get_inputs()[0].name
    kps, scs = [], []
    for box in boxes:
        center, scale = _box_to_center_scale(box, in_w / in_h)
        m = _warp_matrix(center, scale, in_w, in_h)
        crop = cv2.warpAffine(img_rgb, m, (in_w, in_h), flags=cv2.INTER_LINEAR)
        tensor = ((crop.astype(np.float32) - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)
        simcc_x, simcc_y = session.run(None, {name: tensor[None]})
        xy, score = _simcc_decode(simcc_x, simcc_y)
        xy = xy / np.array([in_w, in_h], dtype=np.float32) * scale + center - scale / 2
        kps.append(xy)
        scs.append(score)
    if not kps:
        return np.zeros((0, 133, 2), dtype=np.float32), np.zeros((0, 133), dtype=np.float32)
    return np.stack(kps).astype(np.float32), np.stack(scs).astype(np.float32)


# ── backend ───────────────────────────────────────────────────────────────────────


class DWPoseBackend:
    """`infer(frames, fps) → PoseClip` with 134-point layout (see :mod:`.skeleton`)."""

    device: str
    fp32 = True

    def __init__(
        self, device: str = "auto", checkpoint_dir: str | None = None, detect_every: int = DETECT_EVERY
    ) -> None:
        from ..backends.base import BackendInfo
        from ..errors import BackendError

        try:
            import onnxruntime  # noqa: F401
        except ImportError as exc:
            raise BackendError(
                'pose needs ONNX Runtime: pip install "reshot[pose]"  (or onnxruntime-gpu for CUDA)'
            ) from exc
        try:
            det_path, pose_path = _weights(checkpoint_dir)
        except Exception as exc:
            raise BackendError(
                f"could not obtain DWPose weights: {exc}. "
                "Set HF_ENDPOINT to a mirror, or pass --checkpoint-dir with yolox_l.onnx + dw-ll_ucoco_384.onnx."
            ) from exc
        self.device = "cuda" if device in ("auto", "cuda") and _cuda_provider_available() else "cpu"
        self.detect_every = max(1, int(detect_every))
        t0 = time.time()
        self.det = _session(det_path, self.device)
        self.pose = _session(pose_path, self.device)
        log.info("DWPose loaded on %s in %.1fs", self.device, time.time() - t0)
        self.info = BackendInfo(
            name="dwpose", variant="yolox_l+dw-ll_ucoco_384", license="Apache-2.0", commercial_ok=True
        )

    def infer(self, frames: np.ndarray, fps: float, *, progress=None) -> PoseClip:
        """`frames`: uint8 RGB `[T, H, W, 3]`. `progress(i, t)` is called per frame if given."""
        from .skeleton import wholebody_to_openpose

        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected [T, H, W, 3] uint8 RGB, got {frames.shape}")
        t, h, w = frames.shape[:3]
        clip = PoseClip(width=w, height=h, fps=fps)
        prev_thumb = None
        prev_kps = prev_scs = None
        detections = 0
        for i in range(t):
            rgb = frames[i]
            thumb = cv2.resize(rgb, (64, 36), interpolation=cv2.INTER_AREA).mean(axis=2)
            cut = prev_thumb is None or float(np.abs(thumb - prev_thumb).mean()) > CUT_THRESHOLD
            prev_thumb = thumb
            boxes = None if cut or i % self.detect_every == 0 else boxes_from_skeletons(prev_kps, prev_scs, w, h)
            if boxes is None:
                boxes = detect_people(self.det, np.ascontiguousarray(rgb[..., ::-1]))
                detections += 1
            kps, scs = estimate_pose(self.pose, rgb, boxes)
            kps, scs = wholebody_to_openpose(kps, scs)
            clip.keypoints.append(kps)
            clip.scores.append(scs)
            prev_kps, prev_scs = kps, scs
            if progress:
                progress(i + 1, t)
        log.info("detector ran on %d of %d frames", detections, t)
        return clip

    def peak_memory_bytes(self) -> dict[str, int]:
        return {}


def boxes_from_skeletons(kps: np.ndarray | None, scs: np.ndarray | None, w: int, h: int) -> np.ndarray | None:
    """Person boxes for the next frame from this frame's OpenPose skeletons, or None when
    any skeleton is missing / unreliable (→ the caller runs the detector instead).

    Nobody in the frame is *not* a reason to detect every frame: an empty frame stays
    empty until the cadence or a cut brings the detector back."""
    if kps is None or scs is None:
        return None
    from .skeleton import VISIBLE_SCORE

    boxes = []
    for n in range(kps.shape[0]):
        body_s = scs[n, :18]
        vis = body_s > VISIBLE_SCORE
        if vis.sum() < 4 or float(body_s[vis].mean()) < TRACK_MIN_SCORE:
            return None
        pts = kps[n, :18][vis]
        x1, y1 = pts.min(axis=0)
        x2, y2 = pts.max(axis=0)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        bw, bh = max(x2 - x1, 32.0) * TRACK_BOX_PAD, max(y2 - y1, 32.0) * TRACK_BOX_PAD
        boxes.append(
            [max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(float(w), cx + bw / 2), min(float(h), cy + bh / 2)]
        )
    return np.array(boxes, dtype=np.float32).reshape(-1, 4)


def _cuda_provider_available() -> bool:
    try:
        import onnxruntime as ort

        return "CUDAExecutionProvider" in ort.get_available_providers()
    except Exception:
        return False


__all__ = ["DWPoseBackend", "boxes_from_skeletons", "detect_people", "estimate_pose"]
