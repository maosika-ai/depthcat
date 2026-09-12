"""The pipeline: plan → read → depth → grey → encode. This is the only module that wires
the steps together; the CLI and the Python API both call `run()`.

    from reshot import RunConfig, run
    result = run(RunConfig(input=Path("in.mp4"), output=Path("out.mp4"), target="h3"))

Design notes
------------
* Inference always runs at the model's working resolution (short side `input_size`).
  The 8-bit result is scaled to the output size frame by frame during encoding, so a
  12 s 720p clip needs ~4 GB of host RAM instead of 8–20 GB (see planning.py).
* Normalisation is computed once over the whole clip; per-frame normalisation makes
  static geometry flicker, which the downstream generator reads as motion.
* Failure modes that are the *user's* to fix raise `ReshotError` subclasses with a
  concrete suggestion; everything else propagates as a bug.
"""

from __future__ import annotations

import json
import os
import platform
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ._version import __version__
from .backends import get_backend, pick_device
from .config import RunConfig
from .errors import InputError, RamBudgetError
from .io import probe_video, read_video, write_gray_video
from .planning import gib, memory_verdict, processing_max_res
from .postprocess import to_gray, upsample_frames
from .reporter import NullReporter, Reporter
from .targets import TARGETS, fit_dimensions


@dataclass(frozen=True)
class Plan:
    """What the run will do, decided before any heavy work."""

    src_w: int
    src_h: int
    src_fps: float
    src_frames: int
    out_fps: float
    frames: int  # frames that will be processed (after fps / max-frames)
    proc_w: int
    proc_h: int
    proc_max_res: int  # value handed to read_video
    out_w: int
    out_h: int
    ram_estimate: int
    ram_budget: int


@dataclass(frozen=True)
class RunResult:
    output: Path
    frames: int
    fps: float
    width: int
    height: int
    device: str
    precision: str
    license: str
    depth_seconds: float
    total_seconds: float
    peak_rss_bytes: int
    metrics: dict

    @property
    def ms_per_frame(self) -> float:
        return self.depth_seconds / max(self.frames, 1) * 1000.0


def plan(cfg: RunConfig) -> Plan:
    """Probe the source and decide sizes and frame counts. Cheap; no decoding."""
    try:
        src_w, src_h, src_fps, src_n = probe_video(cfg.input)
    except FileNotFoundError as exc:
        raise InputError(f"cannot open {cfg.input}") from exc
    if src_w <= 0 or src_h <= 0:
        raise InputError(f"{cfg.input} has no video stream")

    target = TARGETS[cfg.target]
    fps_req = cfg.fps if cfg.fps else (target.fps or -1)
    out_fps = src_fps if fps_req <= 0 or fps_req >= src_fps else fps_req
    frames = src_n if cfg.max_frames is None else min(src_n, cfg.max_frames)
    if out_fps < src_fps:
        frames = round(frames * out_fps / src_fps)
    frames = max(frames, 1)

    cap = cfg.max_res if cfg.max_res > 0 else 10**9
    proc_max_res = min(processing_max_res(src_w, src_h, cfg.input_size), cap)
    scale = min(1.0, proc_max_res / max(src_w, src_h))
    proc_w, proc_h = (int(round(src_w * scale / 2) * 2), int(round(src_h * scale / 2) * 2))

    out_scale = min(1.0, cap / max(src_w, src_h))
    out_h, out_w = fit_dimensions(int(src_h * out_scale), int(src_w * out_scale), target.multiple)

    verdict = memory_verdict(frames, proc_w, proc_h)
    return Plan(
        src_w,
        src_h,
        src_fps,
        src_n,
        out_fps,
        frames,
        proc_w,
        proc_h,
        proc_max_res,
        out_w,
        out_h,
        verdict["estimate"],
        verdict["budget"],
    )


def run(cfg: RunConfig, reporter: Reporter | None = None) -> RunResult:
    """Execute a full run. See module docstring."""
    rep = reporter or NullReporter()
    t0 = time.time()
    target = TARGETS[cfg.target]

    # ── plan ──────────────────────────────────────────────────────────────────
    p = plan(cfg)
    rep.step(
        "plan",
        f"{p.src_w}x{p.src_h} @ {p.src_fps:.3f} fps × {p.src_frames} → "
        f"process {p.proc_w}x{p.proc_h} × ~{p.frames}, host RAM ≈ {gib(p.ram_estimate)} "
        f"(budget {gib(p.ram_budget)})",
    )
    if p.ram_estimate > p.ram_budget and not cfg.force:
        raise RamBudgetError(
            p.ram_estimate, p.ram_budget, memory_verdict(p.frames, p.proc_w, p.proc_h)["max_frames_ok"]
        )

    # ── read ──────────────────────────────────────────────────────────────────
    fps_req = cfg.fps if cfg.fps else (target.fps or -1)
    frames, fps = read_video(cfg.input, max_frames=cfg.max_frames or -1, target_fps=fps_req, max_res=p.proc_max_res)
    t, h, w = frames.shape[:3]
    rep.step("read", f"{t} frames @ {fps:.3f} fps, {w}x{h}")
    if target.max_seconds and t / fps > target.max_seconds + 1e-6:
        rep.step("warn", f"clip is {t / fps:.1f}s; {target.key} accepts ≤ {target.max_seconds:.0f}s — trim it")

    # ── depth ─────────────────────────────────────────────────────────────────
    backend_name = "fake" if os.environ.get("RESHOT_FAKE_BACKEND") else cfg.backend
    device = pick_device(cfg.device)
    backend = (
        get_backend("fake")
        if backend_name == "fake"
        else get_backend(
            "vda", model=cfg.model, device=device, checkpoint=str(cfg.checkpoint) if cfg.checkpoint else None
        )
    )
    precision = "fp32" if backend.fp32 else "fp16"
    rep.step(
        "model",
        f"{backend.info.name}/{backend.info.variant} on {backend.device} ({precision}), licence {backend.info.license}",
    )
    t1 = time.time()
    depths = backend.infer(frames, fps, input_size=cfg.input_size)
    depth_seconds = time.time() - t1
    del frames
    _release_accelerator(backend.device)
    del backend
    rep.step("depth", f"{depth_seconds:.1f}s = {depth_seconds / t * 1000:.0f} ms/frame")

    if cfg.npz:
        np.savez_compressed(cfg.npz, depths=depths, fps=fps)
        rep.step("npz", str(cfg.npz))

    # ── grey → encode ─────────────────────────────────────────────────────────
    gray = to_gray(depths, invert=cfg.invert, clip_percent=cfg.clip_percent, gamma=cfg.gamma)
    del depths
    if (p.out_h, p.out_w) != (h, w):
        rep.step("size", f"{w}x{h} → {p.out_w}x{p.out_h} (multiple of {target.multiple})")
    if target.min_pixels and p.out_w * p.out_h < target.min_pixels:
        rep.step(
            "warn",
            f"{p.out_w}x{p.out_h} is below {target.key}'s minimum of {target.min_pixels:,} pixels "
            "— the API will reject it; use a larger source or raise --max-res",
        )
    out = write_gray_video(
        upsample_frames(gray, p.out_h, p.out_w), cfg.output, fps, crf=cfg.crf, size=(p.out_h, p.out_w)
    )
    total = time.time() - t0
    peak = _peak_rss_bytes()
    rep.step("wrote", f"{out}  ({total:.1f}s total, peak RSS {gib(peak)})")

    metrics = {
        "version": __version__,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in asdict(cfg).items()},
        "plan": asdict(p),
        "device": device,
        "precision": precision,
        "frames": t,
        "fps": fps,
        "depth_seconds": round(depth_seconds, 3),
        "ms_per_frame": round(depth_seconds / t * 1000, 1),
        "total_seconds": round(total, 2),
        "peak_rss_bytes": peak,
        "output": str(out),
        "output_size": [p.out_w, p.out_h],
        "output_bytes": os.path.getsize(out),
    }
    if cfg.metrics:
        with open(cfg.metrics, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        rep.step("metrics", str(cfg.metrics))
    if not _license_ok(metrics):
        rep.step("note", "weights are non-commercial; use --model small for commercial work")

    return RunResult(
        out, t, fps, p.out_w, p.out_h, device, precision, backend_name, depth_seconds, total, peak, metrics
    )


def extract(
    video: str | Path,
    *,
    model: str = "small",
    device: str = "auto",
    max_res: int = 1280,
    target_fps: float = -1,
    max_frames: int = -1,
    input_size: int = 518,
    checkpoint: str | None = None,
) -> tuple[np.ndarray, float]:
    """Convenience: decode → depth, no encoding. Returns `(depths[T,H,W] float32, fps)`.

    Same resolution policy as `run()`: frames are read at model resolution."""
    w, h, _, _ = probe_video(video)
    proc = min(processing_max_res(w, h, input_size), max_res if max_res > 0 else 10**9)
    frames, fps = read_video(video, max_frames=max_frames, target_fps=target_fps, max_res=proc)
    backend = get_backend("vda", model=model, device=device, checkpoint=checkpoint)
    return backend.infer(frames, fps, input_size=input_size), fps


def _license_ok(metrics: dict) -> bool:
    return metrics["config"]["model"] == "small" or metrics["config"]["backend"] == "fake"


def _peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(ru if platform.system() == "Darwin" else ru * 1024)  # macOS bytes, Linux KiB


def _release_accelerator(device: str) -> None:
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:
        pass
