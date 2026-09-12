"""Decide *how big* to run before touching the GPU: processing resolution and a host-RAM
estimate. Pure functions, unit-tested without a model.

Why this module exists: on 2026-09-11 a 12 s 736×1280 clip took the reference pipeline to
20.8 GB of RSS on a 32 GB Mac and the OS killed the desktop. The model itself only ever
sees a 518 px short side; everything above that was full-resolution float32 depth kept
for every frame. So: infer at model resolution, estimate memory up front, refuse loudly
instead of swapping the machine to death. The estimate is a measured line, not a guess.
"""

from __future__ import annotations

import os

#: Peak host RSS = HOST_BASE_BYTES + HOST_BYTES_PER_PIXEL_FRAME × frames × processing pixels.
#: Fitted on 2026-09-12 (reshot 0.3.2, RTX 4090, CUDA 12.8, torch 2.8, Small model) over six
#: runs spanning 22M–138M pixel·frames: base 1.73 GiB, slope 16.9 B, max residual 0.05 GiB
#: (docs/ops/depthcat-autodl/runs/calib_20260912_145501 in the Maosika repo). The base is
#: torch + CUDA context + the model; the slope is RGB frames (3 B) + the model's float depth
#: list and its aligned copy (2 × 4 B) + grey output (1 B) + transient window tensors.
#: Both are padded ~15 % so a laptop with a browser open is still refused before it swaps.
HOST_BASE_BYTES = int(2.0 * 2**30)
HOST_BYTES_PER_PIXEL_FRAME = 20.0

#: Refuse when the estimate exceeds this share of physical RAM. Half, not 90 %: the
#: OS, the browser and the GPU driver (unified memory on Apple Silicon) all need room.
HOST_RAM_BUDGET_RATIO = 0.5


def processing_max_res(width: int, height: int, input_size: int = 518) -> int:
    """Longest side (even) at which the *short* side equals `input_size`.

    Feeding frames larger than this is pure waste: the model resizes the short side to
    `input_size` anyway, and the extra pixels only inflate the float depth we keep.
    """
    short, long = min(width, height), max(width, height)
    if short <= input_size:
        return long
    scaled = long * input_size / short
    return int(round(scaled / 2) * 2)


def estimate_host_bytes(frames: int, proc_w: int, proc_h: int) -> int:
    """Peak host RSS for a run: fixed base plus a per-pixel-frame slope (see constants)."""
    return HOST_BASE_BYTES + int(frames * proc_w * proc_h * HOST_BYTES_PER_PIXEL_FRAME)


def physical_ram_bytes() -> int:
    """Total RAM; falls back to 16 GB when the platform won't tell us."""
    try:
        import psutil  # optional

        return int(psutil.virtual_memory().total)
    except Exception:
        pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page > 0:
            return int(pages * page)
    except (ValueError, OSError, AttributeError):
        pass
    return 16 * 2**30


def memory_verdict(frames: int, proc_w: int, proc_h: int, ram_bytes: int | None = None) -> dict:
    """`{"estimate", "budget", "ok", "max_frames_ok"}` — the CLI turns this into text."""
    ram = ram_bytes if ram_bytes is not None else physical_ram_bytes()
    budget = int(ram * HOST_RAM_BUDGET_RATIO)
    est = estimate_host_bytes(frames, proc_w, proc_h)
    per_frame = max(1, estimate_host_bytes(1, proc_w, proc_h) - HOST_BASE_BYTES)
    return {
        "estimate": est,
        "budget": budget,
        "ok": est <= budget,
        "max_frames_ok": max(1, (budget - HOST_BASE_BYTES) // per_frame),
    }


def gib(n: int | float) -> str:
    return f"{n / 2**30:.1f} GiB"
