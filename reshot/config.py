"""Everything a run needs, in one immutable object. Built by the CLI from argv, or by
library users directly. Validation that does not need I/O happens in `__post_init__`
so mistakes surface before any work starts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .errors import InputError
from .targets import TARGETS

MODEL_VARIANTS = ("small", "base", "large")
QUALITIES = ("fast", "full")  # model working size: fast = 644×364 for 16:9 (default), full = 924×518
BACKENDS = ("vda", "fake")  # "fake" also selects the fake pose backend when control == "pose"
CONTROLS = ("depth", "pose", "canny")  # what the output video shows; see pipeline.py


@dataclass(frozen=True)
class RunConfig:
    input: Path
    output: Path
    model: str = "small"
    backend: str = "vda"
    device: str = "auto"
    target: str = "none"
    control: str = "depth"  # depth | pose | canny — see CONTROLS
    fps: float | None = None  # None = preset or source
    max_res: int = 1280  # cap on the OUTPUT's longer side
    max_frames: int | None = None
    quality: str = "fast"  # fast | full — see QUALITIES; `input_size` overrides it
    input_size: int | None = None  # model short side (multiple of 14); expert override of `quality`
    invert: bool = False
    clip_percent: float = 0.0
    gamma: float = 1.0
    crf: int = 12
    npz: Path | None = None
    metrics: Path | None = None
    checkpoint: Path | None = None
    force: bool = False  # ignore the RAM budget
    # pose only
    pose_hands: bool = True  # draw the 21-point hands
    pose_face: bool = False  # draw the 68 face points (off: faces are what ReShot throws away)
    pose_smooth: bool = True  # track + hysteresis + One-Euro; off = raw per-frame output
    keypoints: Path | None = None  # also write the skeletons as JSON (reshot-pose/1)
    checkpoint_dir: Path | None = None  # local folder with the two DWPose .onnx files
    pose_detect_every: int = 3  # detector cadence in frames; 1 = every frame (slowest, no tracking assumptions)
    # canny only
    canny_low: int = 100
    canny_high: int = 200
    cuda_memory_fraction: float | None = None  # cap VRAM to this share of the card (verification aid)
    extra: dict = field(default_factory=dict)  # free-form, echoed into metrics

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise InputError(f"unknown --target {self.target!r}; choose from {', '.join(sorted(TARGETS))}")
        if self.model not in MODEL_VARIANTS:
            raise InputError(f"unknown --model {self.model!r}; choose from {', '.join(MODEL_VARIANTS)}")
        if self.backend not in BACKENDS:
            raise InputError(f"unknown backend {self.backend!r}; choose from {', '.join(BACKENDS)}")
        if self.control not in CONTROLS:
            raise InputError(f"unknown --control {self.control!r}; choose from {', '.join(CONTROLS)}")
        if self.pose_detect_every < 1:
            raise InputError("--pose-detect-every must be ≥ 1")
        if not 0 <= self.canny_low <= self.canny_high <= 1000:
            raise InputError("--canny must be LOW,HIGH with 0 <= LOW <= HIGH")
        if self.quality not in QUALITIES:
            raise InputError(f"unknown --quality {self.quality!r}; choose from {', '.join(QUALITIES)}")
        if self.input_size is not None and (self.input_size < 14 or self.input_size % 14):
            raise InputError("--input-size must be a positive multiple of 14 (ViT patch size)")
        if not 0 <= self.clip_percent < 50:
            raise InputError("--clip must be in [0, 50)")
        if self.gamma <= 0:
            raise InputError("--gamma must be > 0")
        if not 0 <= self.crf <= 51:
            raise InputError("--crf must be in [0, 51]")
        if self.max_frames is not None and self.max_frames < 1:
            raise InputError("--max-frames must be ≥ 1")
        if self.cuda_memory_fraction is not None and not 0 < self.cuda_memory_fraction <= 1:
            raise InputError("--cuda-memory-fraction must be in (0, 1]")
