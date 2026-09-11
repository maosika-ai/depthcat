"""`depthcat` command line.

    depthcat in.mp4 -o out.mp4                # Apache-2.0 Small model, auto device
    depthcat in.mp4 -o out.mp4 --target h3    # 24 fps, ×32 dims, ≤15 s for MiniMax H3
    depthcat in.mp4 -o out.mp4 --npz depth.npz --metrics run.json

Memory model: depth is estimated at the model's working resolution (short side 518 px)
and only the 8-bit result is upscaled to the output size, frame by frame while
encoding. A host-RAM estimate is printed and, if it exceeds half of physical RAM, the
run is refused with a concrete `--max-frames` suggestion (override with `--force`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import resource
import sys
import time

import numpy as np

from . import __version__
from .backends import VDABackend, pick_device
from .io import probe_video, read_video, write_gray_video
from .planning import gib, memory_verdict, processing_max_res
from .postprocess import to_gray, upsample_frames
from .targets import TARGETS, fit_dimensions


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="depthcat",
        description="Video → depth blockout video for video-generation ControlNets.",
    )
    p.add_argument("input", help="input video (anything ffmpeg/OpenCV can read)")
    p.add_argument("-o", "--output", required=True, help="output .mp4")
    p.add_argument("--model", default="small", choices=["small", "base", "large"],
                   help="VDA variant; only 'small' is Apache-2.0 (default)")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--target", default="none", choices=sorted(TARGETS),
                   help="fps / size preset for a specific generator")
    p.add_argument("--fps", type=float, default=-1, help="override output fps (-1 = preset or source)")
    p.add_argument("--max-res", type=int, default=1280,
                   help="cap the longer side of the OUTPUT (inference always runs at model resolution)")
    p.add_argument("--max-frames", type=int, default=-1, help="stop after N frames (-1 = all)")
    p.add_argument("--input-size", type=int, default=518, help="model short side (multiple of 14)")
    p.add_argument("--invert", action="store_true", help="far = white instead of near = white")
    p.add_argument("--clip", type=float, default=0.0, metavar="PCT",
                   help="percentile clip on both tails before scaling (0 = exact min/max)")
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--crf", type=int, default=12, help="x264 CRF; lower = larger, cleaner")
    p.add_argument("--npz", help="also save raw float depth (at processing resolution) to this .npz")
    p.add_argument("--metrics", help="write timing / memory / size metrics as JSON to this path")
    p.add_argument("--checkpoint", help="local .pth instead of the Hugging Face download")
    p.add_argument("--force", action="store_true", help="run even if the RAM estimate exceeds the budget")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=__version__)
    return p


def _peak_rss_bytes() -> int:
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    return int(ru if platform.system() == "Darwin" else ru * 1024)


def _say(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    target = TARGETS[args.target]
    fps_req = args.fps if args.fps > 0 else (target.fps or -1)
    metrics: dict = {"version": __version__, "input": args.input, "model": args.model, "target": args.target}
    t0 = time.time()

    # ── plan: how big will this run be? ────────────────────────────────────────────
    src_w, src_h, src_fps, src_n = probe_video(args.input)
    out_fps = src_fps if fps_req <= 0 or fps_req >= src_fps else fps_req
    n_frames = src_n if args.max_frames <= 0 else min(src_n, args.max_frames)
    n_frames = int(round(n_frames * out_fps / src_fps)) if out_fps < src_fps else n_frames
    proc_res = min(processing_max_res(src_w, src_h, args.input_size), args.max_res if args.max_res > 0 else 10**9)
    scale = min(1.0, proc_res / max(src_w, src_h))
    proc_w, proc_h = int(round(src_w * scale / 2) * 2), int(round(src_h * scale / 2) * 2)
    verdict = memory_verdict(n_frames, proc_w, proc_h)
    _say(f"plan   {src_w}x{src_h} @ {src_fps:.3f} fps × {src_n} frames → process {proc_w}x{proc_h} × ~{n_frames} frames, "
         f"host RAM ≈ {gib(verdict['estimate'])} (budget {gib(verdict['budget'])})")
    metrics.update(source=[src_w, src_h, src_fps, src_n], process=[proc_w, proc_h, n_frames],
                   ram_estimate=verdict["estimate"], ram_budget=verdict["budget"])
    if not verdict["ok"] and not args.force:
        _say(f"abort  estimated RAM exceeds budget. Try --max-frames {verdict['max_frames_ok']} "
             f"(or split the clip), a smaller --input-size, or --force if you know the machine can take it.")
        return 3

    # ── read ────────────────────────────────────────────────────────────────────────
    frames, fps = read_video(args.input, max_frames=args.max_frames, target_fps=fps_req, max_res=proc_res)
    t, h, w = frames.shape[:3]
    _say(f"read   {t} frames @ {fps:.3f} fps, {w}x{h}  ({time.time() - t0:.1f}s)")
    if target.max_seconds and t / fps > target.max_seconds + 1e-6:
        _say(f"warn   clip is {t / fps:.1f}s; {target.key} accepts ≤ {target.max_seconds:.0f}s — trim it")

    # ── depth ───────────────────────────────────────────────────────────────────────
    device = pick_device(args.device)
    if os.environ.get("DEPTHCAT_FAKE_BACKEND"):
        # Dry-run mode for exercising the whole pipeline (I/O, encoding, metrics, remote
        # runbook) without loading a model. Produces a plausible radial gradient.
        _say("model  FAKE backend (DEPTHCAT_FAKE_BACKEND set) — no inference")
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        base = 1.0 / (1.0 + ((xx - w / 2) ** 2 + (yy - h / 2) ** 2) / (0.15 * w * h))
        depths = np.stack([base * (0.8 + 0.2 * i / max(1, t - 1)) for i in range(t)]).astype(np.float32)
        precision, license_ = "none", "n/a"
        dt = 0.0
    else:
        backend = VDABackend(args.model, device=device, checkpoint=args.checkpoint)
        precision, license_ = ("fp32" if backend.fp32 else "fp16"), backend.info.license
        _say(f"model  {backend.info.name}/{backend.info.variant} on {device} ({precision}), licence {license_}")
        t1 = time.time()
        depths = backend.infer(frames, fps, input_size=args.input_size)
        dt = time.time() - t1
        del backend
        _release_accelerator(device)
        _say(f"depth  {dt:.1f}s = {dt / t * 1000:.0f} ms/frame")
    del frames
    metrics.update(device=device, precision=precision, license=license_, frames=t, fps=fps,
                   depth_seconds=round(dt, 2), ms_per_frame=round(dt / t * 1000, 1))

    if args.npz:
        np.savez_compressed(args.npz, depths=depths, fps=fps)
        _say(f"npz    {args.npz}")

    # ── grey → output size → encode (frame by frame) ────────────────────────────────
    gray = to_gray(depths, invert=args.invert, clip_percent=args.clip, gamma=args.gamma)
    del depths
    out_scale = min(1.0, (args.max_res if args.max_res > 0 else 10**9) / max(src_w, src_h))
    out_h, out_w = fit_dimensions(int(src_h * out_scale), int(src_w * out_scale), target.multiple)
    if (out_h, out_w) != (h, w):
        _say(f"size   {w}x{h} → {out_w}x{out_h} (multiple of {target.multiple})")
    out = write_gray_video(upsample_frames(gray, out_h, out_w), args.output, fps, crf=args.crf, size=(out_h, out_w))
    total = time.time() - t0
    peak = _peak_rss_bytes()
    _say(f"wrote  {out}  ({total:.1f}s total, peak RSS {gib(peak)})")
    metrics.update(output=str(out), output_size=[out_w, out_h], total_seconds=round(total, 1),
                   peak_rss_bytes=peak, output_bytes=os.path.getsize(out))
    if args.metrics:
        with open(args.metrics, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        _say(f"metrics {args.metrics}")
    if license_ not in ("Apache-2.0", "n/a"):
        _say("note   weights are non-commercial; use --model small for commercial work")
    return 0


def _release_accelerator(device: str) -> None:
    """Give cached GPU/MPS memory back before encoding; harmless on CPU."""
    try:
        import torch

        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            torch.mps.empty_cache()
    except Exception:  # pragma: no cover
        pass


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
