"""depthcat — turn any video into a depth "blockout" video for video-generation control.

    from depthcat import extract, to_gray, write_gray_video
    depths, fps = extract("in.mp4")            # float32 [T, H, W], larger = closer
    write_gray_video(to_gray(depths), "out.mp4", fps)
"""

from __future__ import annotations

import os

# Apple Silicon: cap the MPS allocator at 60 % of the recommended working set. Unified
# memory means GPU allocations count against the same RAM as everything else; without
# a cap a long clip grows until the OS starts killing processes (seen 2026-09-11).
# With the cap we get a clean MPS out-of-memory error instead. Must be set before
# torch is imported, which is why it lives here and not in the backend.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")

import numpy as np

from .backends import VDABackend, pick_device
from .io import probe_video, read_video, write_gray_video
from .planning import memory_verdict, processing_max_res
from .postprocess import to_gray, upsample_frames
from .targets import TARGETS, center_crop, fit_dimensions

__version__ = "0.1.0"


def extract(
    video: str,
    *,
    model: str = "small",
    device: str = "auto",
    max_res: int = 1280,
    target_fps: float = -1,
    max_frames: int = -1,
    input_size: int = 518,
    checkpoint: str | None = None,
) -> tuple[np.ndarray, float]:
    """One call: decode → depth. Returns `(depths[T,H,W] float32, fps)`."""
    # Infer at model resolution regardless of `max_res` — see planning.processing_max_res.
    w, h, _, _ = probe_video(video)
    proc = min(processing_max_res(w, h, input_size), max_res if max_res > 0 else 10**9)
    frames, fps = read_video(video, max_frames=max_frames, target_fps=target_fps, max_res=proc)
    backend = VDABackend(model, device=device, checkpoint=checkpoint)
    return backend.infer(frames, fps, input_size=input_size), fps


__all__ = [
    "__version__",
    "extract",
    "probe_video",
    "read_video",
    "write_gray_video",
    "to_gray",
    "upsample_frames",
    "processing_max_res",
    "memory_verdict",
    "TARGETS",
    "fit_dimensions",
    "center_crop",
    "VDABackend",
    "pick_device",
]
