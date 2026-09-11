"""Video in / video out.

Reading uses OpenCV (no decord dependency — decord has no arm64 macOS wheels).
Writing pipes raw 8-bit grey frames into ffmpeg. We deliberately do not go through
imageio's writer: we need explicit control over pixel format, GOP length and CRF,
because the output is a *control signal*, not a video for humans.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def probe_video(path: str | Path) -> tuple[int, int, float, int]:
    """Cheap header read → `(width, height, fps, frame_count)`; nothing is decoded."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    try:
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
    finally:
        cap.release()


def read_video(
    path: str | Path,
    *,
    max_frames: int = -1,
    target_fps: float = -1,
    max_res: int = -1,
) -> tuple[np.ndarray, float]:
    """Decode to uint8 RGB `[T, H, W, 3]` and return `(frames, fps)`.

    `target_fps` picks frames by *timestamp*, not by integer stride, so 30 → 24 fps
    actually yields 24 fps instead of silently staying at 30 (which is what a
    round(30/24)=1 stride does).
    `max_res` downsizes the longer side; depth is estimated at that size and the
    output video is written at that size too.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = src_fps if target_fps <= 0 or target_fps >= src_fps else float(target_fps)

    frames: list[np.ndarray] = []
    src_index = 0
    next_pick_time = 0.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        t = src_index / src_fps
        src_index += 1
        if out_fps < src_fps and t + 1e-9 < next_pick_time:
            continue
        next_pick_time += 1.0 / out_fps
        if max_res > 0 and max(bgr.shape[:2]) > max_res:
            scale = max_res / max(bgr.shape[:2])
            # Even dims keep yuv420p encoders happy later; rounding here avoids a crop.
            new_w = int(round(bgr.shape[1] * scale / 2) * 2)
            new_h = int(round(bgr.shape[0] * scale / 2) * 2)
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if 0 < max_frames <= len(frames):
            break
    cap.release()
    if not frames:
        raise ValueError(f"no frames decoded from {path}")
    return np.stack(frames), out_fps


def ffmpeg_binary() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("ffmpeg not found; install ffmpeg or `pip install imageio-ffmpeg`") from exc


def write_gray_video(
    gray: "np.ndarray | Iterable[np.ndarray]",
    path: str | Path,
    fps: float,
    *,
    crf: int = 12,
    keyint_seconds: float = 2.0,
    size: tuple[int, int] | None = None,
) -> Path:
    """Encode uint8 `[T, H, W]` as H.264 (yuv420p, neutral chroma).

    CRF 12 is generous on purpose: depth maps are large smooth gradients, exactly the
    content where quantisation shows up as banding, and banding in a control video
    becomes jitter in the generated clip. Measured cost is small (~0.5–1 Mbps at 720p).
    A key frame every `keyint_seconds` lets downstream tools cut the clip anywhere.
    """
    # Accept either a whole [T, H, W] array or a lazy frame iterator (with `size`), so
    # the caller can upsample frame by frame instead of materialising the full-res clip.
    if isinstance(gray, np.ndarray):
        if gray.ndim != 3 or gray.dtype != np.uint8:
            raise ValueError(f"expected uint8 [T, H, W], got {gray.shape} {gray.dtype}")
        h, w = gray.shape[1], gray.shape[2]
        frames: Iterable[np.ndarray] = gray
    else:
        if size is None:
            raise ValueError("size=(h, w) is required when passing a frame iterator")
        h, w = size
        frames = gray
    if h % 2 or w % 2:
        raise ValueError("frame size must be even for yuv420p; use targets.fit_dimensions first")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_binary(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", f"{fps:.6f}", "-i", "-",
        "-an",
        "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
        "-pix_fmt", "yuv420p", "-profile:v", "high",
        "-g", str(max(1, int(round(fps * keyint_seconds)))),
        "-movflags", "+faststart",
        str(path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdin is not None and proc.stderr is not None
    try:
        # Stream frame by frame so a 300-frame 720p clip doesn't need a 300 MB
        # intermediate bytes object. BrokenPipe means ffmpeg died early; its stderr
        # (read below) carries the real reason, so swallow the pipe error here.
        for frame in frames:
            if frame.shape != (h, w) or frame.dtype != np.uint8:
                raise ValueError(f"frame {frame.shape} {frame.dtype} does not match {(h, w)} uint8")
            proc.stdin.write(np.ascontiguousarray(frame).tobytes())
    except BrokenPipeError:
        pass
    finally:
        proc.stdin.close()
    err = proc.stderr.read()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {err.decode(errors='replace')}")
    return path
