"""Keypoint layout and the OpenPose-style renderer.

Layout (134 points per person, index → meaning), in the order DWPose's ControlNet
annotator produces it, so a skeleton video from ReShot matches what `controlnet_aux`
draws and what the video ControlNets were trained on:

    0–17    body, OpenPose-18 order: nose, neck, R-shoulder, R-elbow, R-wrist, L-shoulder,
            L-elbow, L-wrist, R-hip, R-knee, R-ankle, L-hip, L-knee, L-ankle, R-eye, L-eye,
            R-ear, L-ear
    18–23   feet (COCO-WholeBody, unused by the renderer)
    24–91   face, 68 points
    92–112  left hand, 21 points
    113–133 right hand, 21 points

The estimator emits 133 COCO-WholeBody points (17 COCO body + 6 feet + 68 face + 42 hand);
`wholebody_to_openpose` inserts the synthetic neck (mid-shoulders) and reorders the body.

Rendering follows OpenPose's `draw_bodypose` / `draw_handpose` (via DWPose's util.py):
18 joint colours, 17 limb ellipses at 60 % brightness with the joints on top, hands as
rainbow-coloured edges with blue joints. Stroke sizes were made for ~512-px images; they
are scaled with the frame so a 1280-px skeleton is not a hairline.
"""

from __future__ import annotations

import colorsys

import cv2
import numpy as np

LAYOUT = {
    "body": [0, 18],
    "feet": [18, 24],
    "face": [24, 92],
    "hand_left": [92, 113],
    "hand_right": [113, 134],
    "body_names": [
        "nose", "neck", "r_shoulder", "r_elbow", "r_wrist", "l_shoulder", "l_elbow", "l_wrist",
        "r_hip", "r_knee", "r_ankle", "l_hip", "l_knee", "l_ankle", "r_eye", "l_eye", "r_ear", "l_ear",
    ],
}  # fmt: skip

#: COCO-WholeBody (with the neck inserted at 17 → 18 body points) → OpenPose order.
_MMPOSE_IDX = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
_OPENPOSE_IDX = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]

#: Limbs as 1-based OpenPose pairs (the classic table); only the first 17 are drawn.
LIMBS = [
    [2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], [10, 11], [2, 12],
    [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], [1, 16], [16, 18], [3, 17], [6, 18],
]  # fmt: skip
BODY_COLORS = [
    (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
    (0, 255, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
    (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255), (255, 0, 170), (255, 0, 85),
]  # fmt: skip
HAND_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], [10, 11],
    [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20],
]  # fmt: skip
_HAND_COLORS = [
    tuple(int(c * 255) for c in colorsys.hsv_to_rgb(i / len(HAND_EDGES), 1.0, 1.0)) for i in range(len(HAND_EDGES))
]

#: Below this a joint is not drawn (OpenPose/DWPose convention).
VISIBLE_SCORE = 0.3
#: Reference frame size the stroke widths were designed for.
_REF_SIZE = 512.0


def wholebody_to_openpose(kps: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`[N, 133, 2]`, `[N, 133]` → `[N, 134, 2]`, `[N, 134]` in the layout above.

    The neck is the shoulder midpoint and is visible only when both shoulders are."""
    if kps.shape[0] == 0:
        return np.zeros((0, 134, 2), np.float32), np.zeros((0, 134), np.float32)
    neck = kps[:, [5, 6]].mean(axis=1, keepdims=True)  # [N, 1, 2]
    both = (scores[:, 5] > VISIBLE_SCORE) & (scores[:, 6] > VISIBLE_SCORE)
    neck_s = np.where(both, np.minimum(scores[:, 5], scores[:, 6]), 0.0)[:, None]  # [N, 1]
    kps2 = np.concatenate([kps[:, :17], neck, kps[:, 17:]], axis=1)
    sc2 = np.concatenate([scores[:, :17], neck_s, scores[:, 17:]], axis=1)
    out_k, out_s = kps2.copy(), sc2.copy()
    out_k[:, _OPENPOSE_IDX] = kps2[:, _MMPOSE_IDX]
    out_s[:, _OPENPOSE_IDX] = sc2[:, _MMPOSE_IDX]
    return out_k.astype(np.float32), out_s.astype(np.float32)


def stroke_sizes(h: int, w: int) -> tuple[int, int, int, int]:
    """(limb half-width, body joint radius, hand line width, hand joint radius) for a frame."""
    k = max(0.5, min(h, w) / _REF_SIZE)
    return max(1, round(4 * k)), max(1, round(4 * k)), max(1, round(2 * k)), max(1, round(4 * k))


def render_frame(
    kps: np.ndarray,
    scores: np.ndarray,
    h: int,
    w: int,
    *,
    src_w: int | None = None,
    src_h: int | None = None,
    hands: bool = True,
    face: bool = False,
) -> np.ndarray:
    """Draw every person of one frame onto black. `kps` are pixels in a `src_w × src_h`
    frame (defaults to `w × h`); the output is `[h, w, 3]` uint8 RGB."""
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    if kps.shape[0] == 0:
        return canvas
    sx = w / (src_w or w)
    sy = h / (src_h or h)
    pts = kps * np.array([sx, sy], dtype=np.float32)
    vis = scores > VISIBLE_SCORE
    limb_w, joint_r, hand_w, hand_r = stroke_sizes(h, w)

    # body limbs (ellipses), then dim, then joints — the OpenPose look
    for n in range(pts.shape[0]):
        for i in range(17):
            a, b = LIMBS[i][0] - 1, LIMBS[i][1] - 1
            if not (vis[n, a] and vis[n, b]):
                continue
            (xa, ya), (xb, yb) = pts[n, a], pts[n, b]
            mx, my = (xa + xb) / 2, (ya + yb) / 2
            length = float(np.hypot(xa - xb, ya - yb))
            angle = float(np.degrees(np.arctan2(ya - yb, xa - xb)))
            poly = cv2.ellipse2Poly((int(mx), int(my)), (max(1, int(length / 2)), limb_w), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(canvas, poly, BODY_COLORS[i])
    canvas = (canvas * 0.6).astype(np.uint8)
    for n in range(pts.shape[0]):
        for i in range(18):
            if vis[n, i]:
                cv2.circle(canvas, (int(pts[n, i, 0]), int(pts[n, i, 1])), joint_r, BODY_COLORS[i], thickness=-1)

    if hands:
        for n in range(pts.shape[0]):
            for lo, hi in (LAYOUT["hand_left"], LAYOUT["hand_right"]):
                hp, hv = pts[n, lo:hi], vis[n, lo:hi]
                for ie, (a, b) in enumerate(HAND_EDGES):
                    if hv[a] and hv[b]:
                        cv2.line(canvas, _pt(hp[a]), _pt(hp[b]), _HAND_COLORS[ie], thickness=hand_w)
                for j in range(hi - lo):
                    if hv[j]:
                        cv2.circle(canvas, _pt(hp[j]), hand_r, (0, 0, 255), thickness=-1)

    if face:
        lo, hi = LAYOUT["face"]
        for n in range(pts.shape[0]):
            for j in range(lo, hi):
                if vis[n, j]:
                    cv2.circle(canvas, _pt(pts[n, j]), max(1, hand_r // 2), (255, 255, 255), thickness=-1)
    return canvas


def _pt(p: np.ndarray) -> tuple[int, int]:
    return int(p[0]), int(p[1])


def body_bbox(kps: np.ndarray, scores: np.ndarray) -> np.ndarray | None:
    """Axis-aligned box of the visible body joints of one person, or None if fewer than two."""
    vis = scores[:18] > VISIBLE_SCORE
    if vis.sum() < 2:
        return None
    p = kps[:18][vis]
    return np.array([p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()], dtype=np.float32)


__all__ = [
    "BODY_COLORS",
    "HAND_EDGES",
    "LAYOUT",
    "LIMBS",
    "VISIBLE_SCORE",
    "body_bbox",
    "render_frame",
    "stroke_sizes",
    "wholebody_to_openpose",
]
