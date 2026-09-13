"""A pose backend that draws two walking stick figures without loading a model.

Same purpose as :class:`reshot.backends.fake.FakeBackend`: exercise everything around
inference (tracking, rendering, encoding, the CLI, the web UI) in CI, in milliseconds.
Deterministic: frame `t` of a `w × h` clip always yields the same skeletons, and every
joint scores 0.9 except the right wrist, which dips to 0.1 every 7th frame so the
hysteresis path is exercised too."""

from __future__ import annotations

import numpy as np

from ..pose import PoseClip
from .base import BackendInfo


def _figure(cx: float, cy: float, size: float, phase: float) -> np.ndarray:
    """134 points of a front-facing figure; limbs swing with `phase` (radians)."""
    s = size
    sw = 0.3 * s * np.sin(phase)
    p = np.zeros((134, 2), dtype=np.float32)
    p[0] = (cx, cy - 0.9 * s)  # nose
    p[1] = (cx, cy - 0.7 * s)  # neck
    p[2] = (cx - 0.25 * s, cy - 0.7 * s)  # r shoulder
    p[3] = (cx - 0.3 * s, cy - 0.35 * s + sw)  # r elbow
    p[4] = (cx - 0.3 * s, cy + sw)  # r wrist
    p[5] = (cx + 0.25 * s, cy - 0.7 * s)
    p[6] = (cx + 0.3 * s, cy - 0.35 * s - sw)
    p[7] = (cx + 0.3 * s, cy - sw)
    p[8] = (cx - 0.15 * s, cy)  # r hip
    p[9] = (cx - 0.15 * s, cy + 0.45 * s - sw)
    p[10] = (cx - 0.15 * s, cy + 0.9 * s)
    p[11] = (cx + 0.15 * s, cy)
    p[12] = (cx + 0.15 * s, cy + 0.45 * s + sw)
    p[13] = (cx + 0.15 * s, cy + 0.9 * s)
    p[14] = (cx - 0.05 * s, cy - 0.95 * s)
    p[15] = (cx + 0.05 * s, cy - 0.95 * s)
    p[16] = (cx - 0.1 * s, cy - 0.9 * s)
    p[17] = (cx + 0.1 * s, cy - 0.9 * s)
    # hands: a small fan of 21 points around each wrist
    for base, wrist in ((92, p[4]), (113, p[7])):
        ang = np.linspace(-0.6, 0.6, 5)
        p[base] = wrist
        k = base + 1
        for a in ang:
            for r in (1, 2, 3, 4):
                p[k] = wrist + 0.03 * s * r * np.array([np.sin(a), np.cos(a)])
                k += 1
    # face: a ring of 68 points around the nose (not rendered by default)
    ang = np.linspace(0, 2 * np.pi, 68, endpoint=False)
    p[24:92] = p[0] + 0.08 * s * np.stack([np.cos(ang), np.sin(ang)], 1)
    # feet
    p[18:24] = p[[10, 10, 10, 13, 13, 13]] + np.array([[0, 0.02 * s]] * 6)
    return p


class FakePoseBackend:
    info = BackendInfo(name="fake-pose", variant="none", license="n/a", commercial_ok=True)
    fp32 = True
    device = "cpu"

    def infer(self, frames: np.ndarray, fps: float, *, progress=None) -> PoseClip:
        t, h, w = frames.shape[:3]
        clip = PoseClip(width=w, height=h, fps=fps)
        size = 0.4 * min(h, w)
        for i in range(t):
            ph = 2 * np.pi * i / max(fps, 1) * 1.5
            a = _figure(w * (0.3 + 0.1 * i / max(t, 1)), h * 0.5, size, ph)
            b = _figure(w * 0.7, h * 0.5, size * 0.8, ph + np.pi)
            kps = np.stack([a, b])
            scs = np.full((2, 134), 0.9, dtype=np.float32)
            if i % 7 == 3:
                scs[:, 4] = 0.1  # right wrist drops out
            clip.keypoints.append(kps)
            clip.scores.append(scs)
            if progress:
                progress(i + 1, t)
        return clip

    def peak_memory_bytes(self) -> dict[str, int]:
        return {}


__all__ = ["FakePoseBackend"]
