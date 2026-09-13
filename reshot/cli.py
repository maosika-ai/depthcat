"""`reshot` command line: argv → RunConfig(s) → pipeline.run() / run_many().

    reshot in.mp4 -o out.mp4                 # Apache-2.0 Small model, auto device
    reshot in.mp4 -o out.mp4 --target h3     # 24 fps, ×32 dims, ≤15 s for MiniMax H3
    reshot in.mp4 -o pose.mp4 --control pose # OpenPose-style skeleton video instead of depth
    reshot in.mp4 -o out/ --control depth,pose,canny   # all three, out/<name>_<control>.mp4
    reshot in.mp4 -o out.mp4 --npz d.npz --metrics run.json
    reshot clips/*.mp4 -o depth/ --target seedance   # batch: one model load, out/<name>_depth.mp4
    reshot                                    # no arguments: local web UI, opens in the browser
    reshot web --port 8765 --no-browser --out ~/ReShot

Batch mode (several inputs, or `-o` naming a directory): `--npz` / `--metrics`, if given,
are directories too and get `<name>.npz` / `<name>.json` per clip. One bad clip is
reported and skipped; the exit code is that of the first failure.

Exit codes: 0 ok · 1 unexpected · 2 bad input/arguments · 3 over RAM budget ·
4 ffmpeg missing · 5 backend/weights problem.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ._version import __version__
from .config import BACKENDS, CONTROLS, MODEL_VARIANTS, QUALITIES, RunConfig
from .errors import InputError, ReshotError
from .pipeline import run, run_many
from .reporter import StderrReporter
from .targets import TARGETS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reshot",
        description="Video → depth-map / skeleton / line video for video-generation ControlNets.",
        epilog="Docs: https://github.com/maosika-ai/reshot",
    )
    p.add_argument("input", type=Path, nargs="+", help="input video(s) (anything ffmpeg/OpenCV can read)")
    p.add_argument(
        "-o", "--output", type=Path, required=True, help="output .mp4, or a directory when there are several inputs"
    )
    g = p.add_argument_group("control")
    g.add_argument(
        "--control",
        default="depth",
        metavar="KIND[,KIND]",
        help="what to output: depth (default), pose (OpenPose-style skeletons via DWPose), canny (lines); "
        "several at once with commas — then -o must be a directory",
    )
    g.add_argument("--no-hands", action="store_true", help="pose: body only, no 21-point hands")
    g.add_argument("--face", action="store_true", help="pose: also draw the 68 face points (off by default)")
    g.add_argument("--no-smooth", action="store_true", help="pose: raw per-frame output, no tracking / smoothing")
    g.add_argument("--keypoints", type=Path, help="pose: also write the skeletons as JSON (reshot-pose/1)")
    g.add_argument(
        "--checkpoint-dir",
        type=Path,
        help="pose: folder with yolox_l.onnx + dw-ll_ucoco_384.onnx instead of the download",
    )
    g.add_argument(
        "--pose-detect-every",
        type=int,
        default=3,
        metavar="N",
        help="pose: run the person detector every N frames and follow the skeletons in between "
        "(cuts always re-detect); 1 = every frame, ~2.5x slower on CPU (default 3)",
    )
    g.add_argument(
        "--canny", default="100,200", metavar="LOW,HIGH", help="canny: hysteresis thresholds (default 100,200)"
    )
    g = p.add_argument_group("depth model")
    g.add_argument(
        "--model",
        default="small",
        choices=MODEL_VARIANTS,
        help="Video Depth Anything variant; only 'small' is Apache-2.0 (default)",
    )
    g.add_argument("--backend", default="vda", choices=BACKENDS, help=argparse.SUPPRESS)
    g.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    g.add_argument(
        "--quality",
        default="fast",
        choices=QUALITIES,
        help="what the model sees: fast = 644x364 for 16:9 / 364x644 for 9:16, ~3 GB VRAM (default); "
        "full = 924x518 / 518x924, ~11 GB VRAM, 2.5x slower, sharper fine detail",
    )
    g.add_argument(
        "--input-size",
        type=int,
        default=None,
        help="expert: model short side in px, multiple of 14; overrides --quality",
    )
    g.add_argument("--checkpoint", type=Path, help="local .pth instead of the Hugging Face download")
    g = p.add_argument_group("output")
    g.add_argument(
        "--target",
        default="none",
        choices=sorted(TARGETS),
        help="fps / frame-size preset for a generator: " + ", ".join(f"{k}={v.note}" for k, v in TARGETS.items()),
    )
    g.add_argument("--fps", type=float, help="override output fps (default: preset or source)")
    g.add_argument(
        "--max-res", type=int, default=1280, help="cap the OUTPUT's longer side (inference runs at model resolution)"
    )
    g.add_argument("--max-frames", type=int, help="stop after N frames")
    g.add_argument("--invert", action="store_true", help="far = white instead of near = white")
    g.add_argument(
        "--clip", type=float, default=0.0, metavar="PCT", help="percentile clip on both tails before scaling"
    )
    g.add_argument("--gamma", type=float, default=1.0, help=">1 darkens mid-tones")
    g.add_argument("--crf", type=int, default=12, help="x264 CRF; lower = larger, cleaner (default 12)")
    g.add_argument("--npz", type=Path, help="also save raw float depth (processing resolution)")
    g.add_argument("--metrics", type=Path, help="write timing / memory / size metrics as JSON")
    g = p.add_argument_group("safety")
    g.add_argument("--force", action="store_true", help="run even if the RAM estimate exceeds the budget")
    # Verification aid, not a user feature: makes a big card behave like a small one so a
    # "runs on N GB" claim can be tested (torch.cuda.set_per_process_memory_fraction).
    g.add_argument("--cuda-memory-fraction", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"reshot {__version__}")
    return p


def _controls(args: argparse.Namespace) -> list[str]:
    kinds = [k.strip() for k in str(args.control).split(",") if k.strip()]
    for k in kinds:
        if k not in CONTROLS:
            raise InputError(f"unknown --control {k!r}; choose from {', '.join(CONTROLS)}")
    return list(dict.fromkeys(kinds)) or ["depth"]


def _is_batch(args: argparse.Namespace) -> bool:
    """Several inputs or controls, or an output that is (or is spelled like) a directory."""
    o = args.output
    return len(args.input) > 1 or len(_controls(args)) > 1 or o.is_dir() or str(o).endswith(("/", "\\"))


def configs_from_args(args: argparse.Namespace) -> list[RunConfig]:
    """One RunConfig per input. In batch mode the outputs are derived from the input names."""
    batch = _is_batch(args)

    def per_clip(option: Path | None, stem: str, ext: str) -> Path | None:
        if option is None:
            return None
        return option / f"{stem}{ext}" if batch else option

    try:
        low, high = (int(x) for x in str(args.canny).split(","))
    except ValueError as exc:
        raise InputError("--canny must be LOW,HIGH, e.g. 100,200") from exc

    cfgs = []
    for src in args.input:
        stem = src.stem
        for control in _controls(args):
            # several controls → several output files; the per-clip npz/metrics/keypoints
            # names carry the control too so they do not overwrite each other
            tag = f"{stem}_{control}" if len(_controls(args)) > 1 else stem
            cfgs.append(
                RunConfig(
                    input=src,
                    output=args.output / f"{stem}_{control}.mp4" if batch else args.output,
                    model=args.model,
                    backend=args.backend,
                    device=args.device,
                    target=args.target,
                    control=control,
                    fps=args.fps,
                    max_res=args.max_res,
                    max_frames=args.max_frames,
                    quality=args.quality,
                    input_size=args.input_size,
                    invert=args.invert,
                    clip_percent=args.clip,
                    gamma=args.gamma,
                    crf=args.crf,
                    npz=per_clip(args.npz, tag, ".npz"),
                    metrics=per_clip(args.metrics, tag, ".json"),
                    checkpoint=args.checkpoint,
                    force=args.force,
                    cuda_memory_fraction=args.cuda_memory_fraction,
                    pose_hands=not args.no_hands,
                    pose_face=args.face,
                    pose_smooth=not args.no_smooth,
                    keypoints=per_clip(args.keypoints, tag, ".keypoints.json"),
                    checkpoint_dir=args.checkpoint_dir,
                    pose_detect_every=args.pose_detect_every,
                    canny_low=low,
                    canny_high=high,
                )
            )
    return cfgs


def config_from_args(args: argparse.Namespace) -> RunConfig:
    """Single-clip form, kept for callers that build a Namespace themselves."""
    return configs_from_args(args)[0]


def _web_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reshot web", description="Local web UI for ReShot (everything stays on this machine)."
    )
    p.add_argument(
        "--port", type=int, default=8765, help="port to listen on (default 8765; the next free one is used if taken)"
    )
    p.add_argument("--out", type=Path, default=None, help="where depth videos are saved (default ~/ReShot)")
    p.add_argument("--no-browser", action="store_true", help="don't open the browser automatically")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # `reshot` alone, or `reshot web …`: the web UI. Anything else is the classic CLI.
    if not argv or argv[0] == "web":
        wargs = _web_parser().parse_args(argv[1:] if argv else [])
        logging.basicConfig(
            level=logging.INFO if wargs.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
        )
        from .web import serve

        return serve(out_dir=wargs.out, port=wargs.port, open_browser=not wargs.no_browser)
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        cfgs = configs_from_args(args)
        if len(cfgs) == 1 and not _is_batch(args):
            run(cfgs[0], StderrReporter())
            return 0
        results = run_many(cfgs, StderrReporter())
        failures = [r for r in results if isinstance(r, ReshotError)]
        if failures:
            print(f"reshot: {len(failures)} of {len(results)} clips failed", file=sys.stderr)
            return failures[0].exit_code
        return 0
    except ReshotError as exc:
        print(f"reshot: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("reshot: interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
