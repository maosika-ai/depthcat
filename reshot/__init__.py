"""ReShot — turn any video into a depth-map video for video-generation control.

from pathlib import Path
from reshot import RunConfig, run
run(RunConfig(input=Path("in.mp4"), output=Path("out.mp4"), target="h3"))

from reshot import extract, to_gray, write_gray_video
depths, fps = extract("in.mp4")            # float32 [T, H, W], larger = closer
write_gray_video(to_gray(depths), "out.mp4", fps)
"""

from __future__ import annotations

import os

# Apple Silicon: cap the MPS allocator at 60 % of the recommended working set. Unified
# memory means GPU allocations count against the same RAM as everything else; without a
# cap a long clip grows until the OS kills processes (seen 2026-09-11). Must be set
# before torch is imported, which is why it lives here.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")

from ._version import __version__
from .config import RunConfig
from .errors import BackendError, InputError, RamBudgetError, ReshotError, ToolMissingError
from .io import probe_video, read_video, write_gray_video
from .pipeline import Plan, RunResult, extract, plan, run
from .planning import memory_verdict, processing_max_res
from .postprocess import to_gray, upsample_frames
from .targets import TARGETS, Target, center_crop, fit_dimensions

__all__ = [
    "TARGETS",
    "BackendError",
    "InputError",
    "Plan",
    "RamBudgetError",
    "ReshotError",
    "RunConfig",
    "RunResult",
    "Target",
    "ToolMissingError",
    "__version__",
    "center_crop",
    "extract",
    "fit_dimensions",
    "memory_verdict",
    "plan",
    "probe_video",
    "processing_max_res",
    "read_video",
    "run",
    "to_gray",
    "upsample_frames",
    "write_gray_video",
]
