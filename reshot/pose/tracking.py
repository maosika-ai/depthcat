"""Turn per-frame detections into stable skeletons over time.

A per-frame estimator has three kinds of flicker, and a video model reads every one of
them as motion:

1. **Identity** — person A and person B swap slots between frames. Colours in the
   OpenPose render are per joint, not per person, so this is invisible in the video,
   but smoothing needs to know who is who. `track` assigns ids by greedy IoU of body
   boxes against the previous frame; a track that is unmatched for `max_gap` frames ends.
2. **Position jitter** — sub-pixel noise on every joint. `smooth` runs a One-Euro filter
   (Casiez et al., 2012) per joint per track: it removes jitter at rest and follows fast
   moves with little lag, which is exactly the trade-off a fight needs.
3. **Visibility** — a joint hovering around the 0.3 threshold appears and disappears.
   `apply_hysteresis` needs `on_score` to switch a joint on and only drops it below
   `off_score`, so a limb at 0.28–0.33 stays drawn.

The same idea as depth's whole-clip normalisation: decide once, over time, not per frame.
"""

from __future__ import annotations

import math

import numpy as np

from . import PoseClip
from .skeleton import VISIBLE_SCORE, body_bbox


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / area) if area > 0 else 0.0


def track(clip: PoseClip, *, iou_threshold: float = 0.3, max_gap: int = 8) -> PoseClip:
    """Fill `clip.track_ids`. Greedy: best IoU pair first, each side used once."""
    live: dict[int, tuple[np.ndarray, int]] = {}  # id → (last body box, last frame index)
    next_id = 0
    clip.track_ids = []
    for t in range(clip.frames):
        kps, scs = clip.keypoints[t], clip.scores[t]
        n = kps.shape[0]
        ids = np.full(n, -1, dtype=np.int64)
        boxes = [body_bbox(kps[i], scs[i]) for i in range(n)]
        pairs = []
        for i, box in enumerate(boxes):
            if box is None:
                continue
            for tid, (prev_box, _) in live.items():
                iou = _iou(box, prev_box)
                if iou >= iou_threshold:
                    pairs.append((iou, i, tid))
        pairs.sort(reverse=True)
        used_i, used_t = set(), set()
        for _, i, tid in pairs:
            if i in used_i or tid in used_t:
                continue
            ids[i] = tid
            used_i.add(i)
            used_t.add(tid)
        for i in range(n):
            if ids[i] < 0:
                ids[i] = next_id
                next_id += 1
            if boxes[i] is not None:
                live[int(ids[i])] = (boxes[i], t)
        # forget tracks that have not been seen for a while
        live = {tid: v for tid, v in live.items() if t - v[1] <= max_gap}
        clip.track_ids.append(ids)
    return clip


class OneEuro:
    """One-Euro filter for a vector signal. `min_cutoff` sets jitter removal at rest,
    `beta` how quickly the cutoff opens up when the signal moves fast."""

    def __init__(self, freq: float, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0) -> None:
        self.freq = freq
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev: np.ndarray | None = None
        self.dx_prev: np.ndarray | None = None

    @staticmethod
    def _alpha(cutoff: np.ndarray | float, freq: float) -> np.ndarray | float:
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            return x
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff, self.freq)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, self.freq)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev = x_hat, dx_hat
        return x_hat


def smooth(clip: PoseClip, *, min_cutoff: float = 1.0, beta: float = 0.5) -> PoseClip:
    """One-Euro per track per joint, in place. Joints below `VISIBLE_SCORE` are not fed to
    the filter (their coordinates are noise) and reset it when they come back."""
    if not clip.track_ids:
        track(clip)
    filters: dict[int, OneEuro] = {}
    seen_vis: dict[int, np.ndarray] = {}
    for t in range(clip.frames):
        kps, scs, ids = clip.keypoints[t], clip.scores[t], clip.track_ids[t]
        for n in range(kps.shape[0]):
            tid = int(ids[n])
            vis = scs[n] > VISIBLE_SCORE
            f = filters.get(tid)
            if f is None:
                f = filters[tid] = OneEuro(clip.fps, min_cutoff, beta)
                seen_vis[tid] = vis.copy()
            x = kps[n].copy()
            # a joint that was invisible last frame restarts from its raw position
            if f.x_prev is not None:
                fresh = vis & ~seen_vis[tid]
                f.x_prev[fresh] = x[fresh]
                f.dx_prev[fresh] = 0.0
                hidden = ~vis
                x[hidden] = f.x_prev[hidden]  # hold, so the filter state does not drift on noise
            kps[n] = f(x)
            seen_vis[tid] = vis
    return clip


def apply_hysteresis(clip: PoseClip, *, on_score: float = VISIBLE_SCORE, off_score: float = 0.2) -> PoseClip:
    """Rewrite scores so a joint, once on, only turns off below `off_score`. In place.

    Implemented by lifting the score of a held joint to `on_score + ε` so the renderer's
    single threshold still works; the raw score is not preserved (the JSON dump comes
    after this, deliberately: what you see is what you get)."""
    if not clip.track_ids:
        track(clip)
    state: dict[int, np.ndarray] = {}
    for t in range(clip.frames):
        scs, ids = clip.scores[t], clip.track_ids[t]
        for n in range(scs.shape[0]):
            tid = int(ids[n])
            prev = state.get(tid)
            on = scs[n] > on_score
            if prev is not None:
                hold = prev & (scs[n] >= off_score) & ~on
                scs[n][hold] = on_score + 1e-3
                on = on | hold
            state[tid] = on
    return clip


def stabilise(clip: PoseClip) -> PoseClip:
    """The default chain: track → hysteresis → smooth. In place; returns the clip."""
    track(clip)
    apply_hysteresis(clip)
    smooth(clip)
    return clip


__all__ = ["OneEuro", "apply_hysteresis", "smooth", "stabilise", "track"]
