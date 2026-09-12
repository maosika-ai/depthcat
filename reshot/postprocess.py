"""Float depth → 8-bit grey the way video ControlNets expect it.

Two rules, both learned the hard way:

1. **Normalise over the whole clip, never per frame.** Per-frame min/max makes the
   brightness of a static wall change whenever someone walks closer to the camera,
   and the generator reads that as the scene "breathing".
2. **Near is white.** Video Depth Anything outputs inverse depth (larger = closer), and
   depth ControlNets (MiniMax H3 Fun, Wan VACE, SD ControlNet-depth) were trained on
   that convention. `invert=True` exists for tools that want the opposite.
"""

from __future__ import annotations

import numpy as np


def to_gray(
    depths: np.ndarray,
    *,
    invert: bool = False,
    clip_percent: float = 0.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """`[T, H, W]` float → `[T, H, W]` uint8.

    `clip_percent` (0–10) trims that many percent from each tail before scaling, so a
    single hot pixel can't crush the contrast of the whole clip. 0 reproduces the
    upstream reference behaviour exactly.
    `gamma` > 1 pushes mid-tones darker (more separation near the camera); 1 is linear.
    """
    d = np.asarray(depths, dtype=np.float32)
    if d.ndim != 3:
        raise ValueError(f"expected [T, H, W], got {d.shape}")
    if not 0 <= clip_percent < 50:
        raise ValueError("clip_percent must be in [0, 50)")
    if clip_percent > 0:
        lo, hi = np.percentile(d, (clip_percent, 100 - clip_percent))
    else:
        lo, hi = float(d.min()), float(d.max())
    if hi - lo < 1e-6:
        return np.zeros(d.shape, dtype=np.uint8)
    # One scale for the clip (lo/hi above), but convert frame by frame: doing the
    # arithmetic on the whole [T, H, W] array allocated four float32 temporaries of the
    # clip's size — 1.2 GB extra for a 12 s clip, 3 GB for 30 s — which is what decided
    # whether a 16 GB laptop could run a long clip at all. Same operations in the same
    # order, so the bytes are identical to the whole-array version.
    out = np.empty(d.shape, dtype=np.uint8)
    for i in range(d.shape[0]):
        n = np.clip((d[i] - lo) / (hi - lo), 0.0, 1.0)
        if gamma != 1.0:
            n = n ** float(gamma)
        if invert:
            n = 1.0 - n
        out[i] = (n * 255.0 + 0.5).astype(np.uint8)
    return out


def upsample_frames(gray: np.ndarray, height: int, width: int):
    """Yield each `[H, W]` uint8 frame resized to `(height, width)`.

    Depth is estimated at the model's working resolution (short side 518 px) and only
    the 8-bit result is scaled up — the upstream reference upsamples the *float* depth
    of every frame to full size before alignment, which is what made a 12 s 720p clip
    need 8–20 GB of RAM. Bilinear on 8-bit grey is visually identical for a control video.
    """
    import cv2

    for frame in gray:
        if frame.shape == (height, width):
            yield frame
        else:
            yield cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
