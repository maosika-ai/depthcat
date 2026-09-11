"""User-facing errors. The CLI prints `str(exc)` and maps each to an exit code; library
users can catch `DepthcatError` for anything the tool considers a *user* problem
(bad input, over budget, missing tool) as opposed to a bug."""

from __future__ import annotations


class DepthcatError(Exception):
    """Base class; `exit_code` is what the CLI returns."""

    exit_code = 1


class InputError(DepthcatError):
    """The input video could not be opened or decoded."""

    exit_code = 2


class RamBudgetError(DepthcatError):
    """The planned run would exceed the host-RAM budget (see planning.py)."""

    exit_code = 3

    def __init__(self, estimate: int, budget: int, max_frames_ok: int) -> None:
        self.estimate, self.budget, self.max_frames_ok = estimate, budget, max_frames_ok
        from .planning import gib

        super().__init__(
            f"estimated host RAM {gib(estimate)} exceeds the budget {gib(budget)}. "
            f"Try --max-frames {max_frames_ok} (or split the clip), a smaller --input-size, "
            "or --force if you know the machine can take it."
        )


class ToolMissingError(DepthcatError):
    """A required external tool (ffmpeg) is missing."""

    exit_code = 4


class BackendError(DepthcatError):
    """Unknown backend / model variant, or weights could not be loaded."""

    exit_code = 5
