"""`reshot` command line: argv → RunConfig → pipeline.run().

    reshot in.mp4 -o out.mp4                 # Apache-2.0 Small model, auto device
    reshot in.mp4 -o out.mp4 --target h3     # 24 fps, ×32 dims, ≤15 s for MiniMax H3
    reshot in.mp4 -o out.mp4 --npz d.npz --metrics run.json

Exit codes: 0 ok · 1 unexpected · 2 bad input/arguments · 3 over RAM budget ·
4 ffmpeg missing · 5 backend/weights problem.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ._version import __version__
from .config import BACKENDS, MODEL_VARIANTS, RunConfig
from .errors import ReshotError
from .pipeline import run
from .reporter import StderrReporter
from .targets import TARGETS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reshot",
        description="Video → depth-map video for video-generation ControlNets.",
        epilog="Docs: https://github.com/maosika-ai/reshot",
    )
    p.add_argument("input", type=Path, help="input video (anything ffmpeg/OpenCV can read)")
    p.add_argument("-o", "--output", type=Path, required=True, help="output .mp4")
    g = p.add_argument_group("model")
    g.add_argument(
        "--model",
        default="small",
        choices=MODEL_VARIANTS,
        help="Video Depth Anything variant; only 'small' is Apache-2.0 (default)",
    )
    g.add_argument("--backend", default="vda", choices=BACKENDS, help=argparse.SUPPRESS)
    g.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    g.add_argument("--input-size", type=int, default=518, help="model short side, multiple of 14 (default 518)")
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
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"reshot {__version__}")
    return p


def config_from_args(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        input=args.input,
        output=args.output,
        model=args.model,
        backend=args.backend,
        device=args.device,
        target=args.target,
        fps=args.fps,
        max_res=args.max_res,
        max_frames=args.max_frames,
        input_size=args.input_size,
        invert=args.invert,
        clip_percent=args.clip,
        gamma=args.gamma,
        crf=args.crf,
        npz=args.npz,
        metrics=args.metrics,
        checkpoint=args.checkpoint,
        force=args.force,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    try:
        run(config_from_args(args), StderrReporter())
    except ReshotError as exc:
        print(f"reshot: {exc}", file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("reshot: interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
