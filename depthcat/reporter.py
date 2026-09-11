"""Progress reporting. The pipeline talks to a `Reporter`; the CLI supplies one that
writes timestamped lines to stderr, library users get silence by default."""

from __future__ import annotations

import sys
import time
from typing import Protocol


class Reporter(Protocol):
    def step(self, tag: str, message: str) -> None: ...


class NullReporter:
    def step(self, tag: str, message: str) -> None:
        return None


class StderrReporter:
    """`[+12.3s] read   294 frames …` — tag padded so columns line up."""

    def __init__(self) -> None:
        self._t0 = time.time()

    def step(self, tag: str, message: str) -> None:
        print(f"[+{time.time() - self._t0:6.1f}s] {tag:<7s}{message}", file=sys.stderr, flush=True)
