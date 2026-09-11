"""Output presets for specific video-generation models.

A control video only helps if its frame rate and size match what the generator was
trained to consume; otherwise the generator resamples it and you get drift. Values
here are the published constraints of each model's control/ControlNet path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Target:
    key: str
    fps: float | None  # None = keep source fps
    multiple: int  # frame size rounded down to a multiple of this
    max_seconds: float | None  # warn beyond this
    note: str


TARGETS: dict[str, Target] = {
    "none": Target("none", None, 2, None, "keep source fps; only make dimensions even"),
    # MiniMax-H3-Fun-Controlnet-Union: 24 fps fixed, H/W multiples of 32, ≤15 s.
    "h3": Target("h3", 24.0, 32, 15.0, "MiniMax H3 Fun ControlNet (depth/pose/canny/hed/mlsd)"),
    # Wan 2.1 VACE control videos: 16 fps, H/W multiples of 16.
    "wan": Target("wan", 16.0, 16, None, "Wan 2.1 VACE control video"),
}


def fit_dimensions(h: int, w: int, multiple: int) -> tuple[int, int]:
    """Largest (h, w) ≤ input that are multiples of `multiple` (min one block)."""
    return max(multiple, h // multiple * multiple), max(multiple, w // multiple * multiple)


def center_crop(frames: np.ndarray, h: int, w: int) -> np.ndarray:
    """Crop `[T, H, W, ...]` to `(h, w)` around the centre.

    Crop, not pad: a padded black border reads to the generator as a far wall.
    """
    src_h, src_w = frames.shape[1], frames.shape[2]
    top, left = (src_h - h) // 2, (src_w - w) // 2
    return frames[:, top : top + h, left : left + w]
