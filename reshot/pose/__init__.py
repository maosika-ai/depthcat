"""Pose: video → per-frame 2-D skeletons → an OpenPose-style skeleton video.

The parts, in the order a run uses them:

* :mod:`.dwpose`   — DWPose in ONNX Runtime: a YOLOX-L person detector and an RTMPose
                     whole-body estimator (133 COCO-WholeBody keypoints per person).
* :mod:`.skeleton` — COCO-WholeBody → OpenPose-18 body layout, and the renderer that
                     draws the coloured skeleton the video ControlNets were trained on.
* :mod:`.tracking` — identity across frames, One-Euro smoothing, visibility hysteresis:
                     what keeps a per-frame estimator from flickering on video.

Keypoints travel between the parts as `PoseClip`: for every frame an `[N, 134, 2]` float32
array of pixel coordinates and an `[N, 134]` score array, N people, 134 = 18 OpenPose body
joints + 6 feet + 68 face + 21 + 21 hand joints (see :mod:`.skeleton`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseClip:
    """Skeletons for a whole clip. `keypoints[t]` is `[N_t, 134, 2]` pixels in a
    `width × height` frame, `scores[t]` is `[N_t, 134]` in [0, 1]. N_t may differ per
    frame. `track_ids[t]` (`[N_t]` ints) is filled by :func:`reshot.pose.tracking.track`."""

    width: int
    height: int
    fps: float
    keypoints: list[np.ndarray] = field(default_factory=list)
    scores: list[np.ndarray] = field(default_factory=list)
    track_ids: list[np.ndarray] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return len(self.keypoints)

    def to_json_dict(self) -> dict:
        """Compact, self-describing JSON: coordinates rounded to 0.1 px, scores to 0.01."""
        from .skeleton import LAYOUT

        return {
            "format": "reshot-pose/1",
            "layout": LAYOUT,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "frames": [
                {
                    "people": [
                        {
                            "id": int(self.track_ids[t][n]) if self.track_ids else n,
                            "keypoints": np.round(self.keypoints[t][n], 1).tolist(),
                            "scores": np.round(self.scores[t][n], 2).tolist(),
                        }
                        for n in range(len(self.keypoints[t]))
                    ]
                }
                for t in range(self.frames)
            ],
        }


__all__ = ["PoseClip"]
