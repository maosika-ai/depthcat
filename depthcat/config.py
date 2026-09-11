"""Everything a run needs, in one immutable object. Built by the CLI from argv, or by
library users directly. Validation that does not need I/O happens in `__post_init__`
so mistakes surface before any work starts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .errors import InputError
from .targets import TARGETS

MODEL_VARIANTS = ("small", "base", "large")
BACKENDS = ("vda", "fake")


@dataclass(frozen=True)
class RunConfig:
    input: Path
    output: Path
    model: str = "small"
    backend: str = "vda"
    device: str = "auto"
    target: str = "none"
    fps: float | None = None  # None = preset or source
    max_res: int = 1280  # cap on the OUTPUT's longer side
    max_frames: int | None = None
    input_size: int = 518  # model short side (multiple of 14)
    invert: bool = False
    clip_percent: float = 0.0
    gamma: float = 1.0
    crf: int = 12
    npz: Path | None = None
    metrics: Path | None = None
    checkpoint: Path | None = None
    force: bool = False  # ignore the RAM budget
    extra: dict = field(default_factory=dict)  # free-form, echoed into metrics

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise InputError(f"unknown --target {self.target!r}; choose from {', '.join(sorted(TARGETS))}")
        if self.model not in MODEL_VARIANTS:
            raise InputError(f"unknown --model {self.model!r}; choose from {', '.join(MODEL_VARIANTS)}")
        if self.backend not in BACKENDS:
            raise InputError(f"unknown backend {self.backend!r}; choose from {', '.join(BACKENDS)}")
        if self.input_size % 14:
            raise InputError("--input-size must be a multiple of 14 (ViT patch size)")
        if not 0 <= self.clip_percent < 50:
            raise InputError("--clip must be in [0, 50)")
        if self.gamma <= 0:
            raise InputError("--gamma must be > 0")
        if not 0 <= self.crf <= 51:
            raise InputError("--crf must be in [0, 51]")
        if self.max_frames is not None and self.max_frames < 1:
            raise InputError("--max-frames must be ≥ 1")
