"""Canny line video: the third control type, and the only one without a model.

Per frame: resize to the output size, grey, Gaussian blur, `cv2.Canny`. White lines on
black, which is what the canny ControlNets (MiniMax H3 Fun ControlNet's `canny` slot
included) expect. Blur first: video compression noise otherwise turns into a crawl of
short edges that the generator reads as texture.

Caveat for users, stated in the docs: unlike depth, lines carry the outline of clothes,
hair and a face. "Copy the shot, not the actors" is only half true for canny.
"""

from __future__ import annotations

from collections.abc import Iterator

import cv2
import numpy as np


def canny_frame(rgb: np.ndarray, h: int, w: int, *, low: int = 100, high: int = 200, blur: int = 3) -> np.ndarray:
    """One RGB frame → uint8 `[h, w]` edge map."""
    if rgb.shape[0] != h or rgb.shape[1] != w:
        rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if blur and blur > 1:
        k = blur if blur % 2 else blur + 1
        grey = cv2.GaussianBlur(grey, (k, k), 0)
    return cv2.Canny(grey, low, high)


def canny_frames(frames: np.ndarray, h: int, w: int, *, low: int = 100, high: int = 200) -> Iterator[np.ndarray]:
    """Lazy per-frame edges for the writer, so a whole edge clip is never held in RAM."""
    for f in frames:
        yield canny_frame(f, h, w, low=low, high=high)


__all__ = ["canny_frame", "canny_frames"]
