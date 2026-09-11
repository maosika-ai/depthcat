"""Backend protocol: a depth model that turns a frame stack into per-frame depth.

Every backend returns *relative inverse depth* (larger = closer to camera) as float32
`[T, H, W]` at the input frame resolution, already temporally aligned across the whole
clip. Normalisation to 8-bit grey happens later in :mod:`depthcat.postprocess`,
so backends never decide what "white" means.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class BackendInfo:
    name: str
    variant: str
    license: str
    commercial_ok: bool


class DepthBackend(Protocol):
    info: BackendInfo

    def infer(self, frames: np.ndarray, fps: float, *, input_size: int = 518) -> np.ndarray:
        """`frames`: uint8 RGB `[T, H, W, 3]` → float32 inverse depth `[T, H, W]`."""
        ...
