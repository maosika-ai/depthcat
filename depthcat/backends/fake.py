"""A backend that returns a plausible depth field without loading a model.

For exercising everything *around* inference — I/O, encoding, presets, metrics,
remote job runners — in CI or on a laptop, in milliseconds and without downloads.
Select with `--backend fake` (or the env var `DEPTHCAT_FAKE_BACKEND=1`)."""

from __future__ import annotations

import numpy as np

from .base import BackendInfo


class FakeBackend:
    info = BackendInfo(name="fake", variant="none", license="n/a", commercial_ok=True)
    fp32 = True
    device = "cpu"

    def infer(self, frames: np.ndarray, fps: float, *, input_size: int = 518) -> np.ndarray:
        t, h, w = frames.shape[:3]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        # a soft radial "subject" in the middle that brightens over time
        base = 1.0 / (1.0 + ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.15 * w * h))
        ramp = np.linspace(0.8, 1.0, num=max(t, 1), dtype=np.float32)
        return (base[None] * ramp[:, None, None]).astype(np.float32)
