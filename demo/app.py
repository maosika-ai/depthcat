"""reshot — Hugging Face Space demo.

Upload a short clip, get a depth-map video back. Runs on the free CPU tier, so the
demo caps the clip (see LIMITS) and defaults to the faster 364-px model input. The same
`reshot.run()` the CLI uses does all the work; this file is only the UI.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import gradio as gr

from reshot import RunConfig, __version__, run
from reshot.errors import ReshotError

LIMITS = {"max_seconds": 4, "max_res": 640}  # free CPU: ~2–4 s per frame at model resolution
FAKE = bool(os.environ.get("RESHOT_FAKE_BACKEND"))  # used by the local smoke test only


def make_depth_map(video: str | None, target: str, input_size: int, invert: bool, gamma: float):
    if not video:
        raise gr.Error("Upload a video first.")
    workdir = Path(tempfile.mkdtemp(prefix="reshot-"))
    out = workdir / "depth.mp4"
    fps_cap = {"none": 24.0, "seedance": None, "h3": None, "wan": None}[target]  # keep the free CPU honest
    max_frames = int(LIMITS["max_seconds"] * 24)
    cfg = RunConfig(
        input=Path(video),
        output=out,
        backend="fake" if FAKE else "vda",
        device="cpu",
        target=target,
        fps=fps_cap,
        max_res=LIMITS["max_res"],
        max_frames=max_frames,
        input_size=input_size,
        invert=invert,
        gamma=gamma,
        metrics=workdir / "metrics.json",
    )
    try:
        res = run(cfg)
    except ReshotError as exc:
        raise gr.Error(str(exc)) from exc
    note = (
        f"{res.frames} frames @ {res.fps:.0f} fps · {res.width}×{res.height} · "
        f"{res.ms_per_frame:.0f} ms/frame on {res.device} · reshot {__version__}"
    )
    return str(out), note


with gr.Blocks(title="ReShot") as demo:
    gr.Markdown(
        f"""
# ReShot
**Video → depth map** for video generation: Seedance reference video, MiniMax H3 Fun ControlNet, Wan VACE, ….
Keeps a shot's staging and camera, drops faces, wardrobe and style.
Code and CLI: [github.com/maosika-ai/reshot](https://github.com/maosika-ai/reshot) · Apache-2.0

*This free-CPU demo processes the first **{LIMITS["max_seconds"]} s** at ≤ {LIMITS["max_res"]} px.
Expect 1–3 minutes. For full clips and GPU speed, run it locally: `pip install git+https://github.com/maosika-ai/reshot`.*
"""
    )
    with gr.Row():
        with gr.Column():
            src = gr.Video(label="Reference video", sources=["upload"])
            target = gr.Radio(
                ["seedance", "h3", "wan", "none"],
                value="seedance",
                label="Target preset",
                info="seedance = 24 fps, ×16 (Seedance 2.0/2.5 reference video); h3 = 24 fps, ×32 (MiniMax H3 Fun ControlNet)",
            )
            with gr.Accordion("Advanced", open=False):
                input_size = gr.Radio(
                    [364, 518], value=364, label="Model input size", info="518 = sharper edges, ~2× slower"
                )
                invert = gr.Checkbox(False, label="Invert (far = white)")
                gamma = gr.Slider(0.6, 1.8, value=1.0, step=0.05, label="Gamma")
            btn = gr.Button("Make depth map", variant="primary")
        with gr.Column():
            out = gr.Video(label="Depth map (download → feed to your ControlNet)")
            note = gr.Markdown()
    btn.click(make_depth_map, [src, target, input_size, invert, gamma], [out, note])
    ex = Path(__file__).with_name("example.mp4")
    if ex.exists():
        gr.Examples([[str(ex), "h3", 364, False, 1.0]], [src, target, input_size, invert, gamma])

if __name__ == "__main__":
    demo.launch()
