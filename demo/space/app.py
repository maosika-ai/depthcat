"""ReShot — Hugging Face Space (ZeroGPU).

Drop a clip, get its depth-map video back, plus the prompt line that points a video model at
it. The same `reshot.run()` the CLI and the local web page use does the work; this file is only
the Gradio UI and the ZeroGPU plumbing.

ZeroGPU rules that shape this file (docs/hub/spaces-zerogpu):
- the model must be placed on `cuda` at import time (CUDA is emulated outside `@spaces.GPU`),
  so one backend is built here and handed to `run(..., backend=BACKEND)` for every clip;
- only the decorated function gets a real GPU, and its `duration` is the visitor's quota cost,
  so it is computed per clip from the frame count rather than a flat maximum.

Runs unchanged on CPU / a normal GPU too: `@spaces.GPU` is a no-op outside ZeroGPU.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import spaces

from reshot import RunConfig, __version__, run
from reshot.backends import get_backend
from reshot.errors import ReshotError

MAX_SECONDS = 15  # the longest clip Seedance / MiniMax H3 accept as a reference; longer input is cut
FPS = 24  # every preset here is 24 fps; `none` keeps the source, capped by MAX_FRAMES anyway
MAX_FRAMES = MAX_SECONDS * FPS
FAKE = bool(os.environ.get("RESHOT_FAKE_BACKEND"))  # local smoke test without weights / GPU

# Built once at import, on cuda (emulated until a @spaces.GPU call), reused for every clip.
BACKEND = get_backend("fake") if FAKE else get_backend("vda", model="small", device="cuda")

# What to do with the file, per preset — the same lines the local web page shows.
NEXT_STEP = {
    "seedance": (
        "Upload `depth.mp4` to **Seedance 2.0 / 2.5** as a reference video and start the prompt with:",
        "参考@视频1的动作与运镜，顺序与视频保持一致。\n(then describe your people, wardrobe, set and look)",
    ),
    "h3": (
        "In **MiniMax H3** attach `depth.mp4` as `<Video 1>` (a character sheet as `<Picture 1>`), "
        "define it in the prompt and say the grey look is not to be copied:",
        "<Subject 3> is the choreography and camera movement shown in <Video 1>, a grey depth map "
        "in which near objects are white and far objects are black: (one sentence on what happens in the clip)\n"
        "<Subject 3>: attribute_transfer - every action, position, timing and camera move of <Video 1> "
        "is transferred onto <Subject 1>; its grey depth look is not transferred.",
    ),
    "wan": ("Connect `depth.mp4` to **WanVaceToVideo → control_video** in ComfyUI.", ""),
    "none": ("Hand `depth.mp4` to any model that reads a depth video.", ""),
}


def _frame_count(video: str) -> int:
    cap = cv2.VideoCapture(video)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or FPS
    cap.release()
    # frames that will actually be processed: resampled to 24 fps, then cut at MAX_FRAMES
    return int(min(max(n, 1) * FPS / fps, MAX_FRAMES))


def gpu_seconds(video: str | None, target: str, quality: str, invert: bool, gamma: float) -> int:
    """Quota to reserve for one clip: read + model + encode. Measured on a 4090: ~35 ms/frame at
    `fast`, ~85 ms at `full` (docs/USAGE.md); the rest is decode/encode on the CPU side. Padded
    so a clip never dies at the deadline, kept small so visitors get a good queue position."""
    if not video:
        return 10
    per_frame = 0.25 if quality == "full" else 0.12
    return int(min(120, 15 + _frame_count(video) * per_frame))


@spaces.GPU(duration=gpu_seconds)
def make_depth_map(video: str | None, target: str, quality: str, invert: bool, gamma: float):
    if not video:
        raise gr.Error("Drop a clip in first.")
    workdir = Path(tempfile.mkdtemp(prefix="reshot-"))
    cfg = RunConfig(
        input=Path(video),
        output=workdir / "depth.mp4",
        backend="fake" if FAKE else "vda",
        device="cuda",
        target=target,
        max_frames=MAX_FRAMES,
        quality=quality,
        invert=invert,
        gamma=gamma,
        metrics=workdir / "metrics.json",
    )
    try:
        res = run(cfg, backend=BACKEND)
    except ReshotError as exc:
        raise gr.Error(str(exc)) from exc
    cut = " · first 15 s" if _frame_count(video) >= MAX_FRAMES else ""
    stats = (
        f"{res.frames} frames · {res.fps:.0f} fps · {res.width}×{res.height} · "
        f"{res.ms_per_frame:.0f} ms/frame on {res.device}{cut} · reshot {__version__}"
    )
    how, prompt = NEXT_STEP[target]
    return str(res.output), stats, how, gr.update(value=prompt, visible=bool(prompt))


CSS = """
.gradio-container { max-width: 1100px !important; }
#prompt textarea { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 13px; }
"""

with gr.Blocks(title="ReShot — copy the shot, not the actors", css=CSS) as demo:
    gr.Markdown(
        """
# ReShot
**Copy the shot, not the actors.** Drop a reference clip; you get its **depth map** back — a grey
video that keeps where everyone stands, how they move and how the camera moves, and drops faces,
clothes and style. Give it to Seedance or MiniMax H3 as the reference and describe your own
characters in the prompt.

Free GPU (ZeroGPU): a 12-second clip takes about 30 seconds. Clips longer than 15 s are cut to 15 s —
that is what the video models accept. Everything is deleted when you leave the page.
[GitHub](https://github.com/maosika-ai/reshot) · `pip install reshot` runs it locally on any clip length ·
[ComfyUI nodes](https://github.com/maosika-ai/ComfyUI-ReShot) · [Show what you made](https://github.com/maosika-ai/reshot/discussions/2)
"""
    )
    with gr.Row():
        with gr.Column():
            src = gr.Video(label="Reference clip (mp4 / mov / webm)", sources=["upload"])
            target = gr.Radio(
                choices=[
                    ("Seedance 2.0 / 2.5", "seedance"),
                    ("MiniMax H3", "h3"),
                    ("Wan 2.1 VACE", "wan"),
                    ("Just the depth map", "none"),
                ],
                value="seedance",
                label="Which model is it for?",
                info="Sets fps and frame size to that model's reference-video rules.",
            )
            quality = gr.Radio(
                choices=[("fast", "fast"), ("full", "full")],
                value="fast",
                label="Quality",
                info="fast: large shapes identical, fine detail softer · full: sharper edges, ~2× the GPU time",
            )
            with gr.Accordion("Advanced", open=False):
                invert = gr.Checkbox(False, label="Invert (far = white)")
                gamma = gr.Slider(0.6, 1.8, value=1.0, step=0.05, label="Gamma")
            btn = gr.Button("Make depth map", variant="primary")
        with gr.Column():
            out = gr.Video(label="Depth map — download this", autoplay=True)
            stats = gr.Markdown()
            how = gr.Markdown()
            prompt = gr.Textbox(
                label="Prompt line to paste", lines=4, show_copy_button=True, visible=False, elem_id="prompt"
            )
    btn.click(make_depth_map, [src, target, quality, invert, gamma], [out, stats, how, prompt])
    ex = Path(__file__).with_name("example.mp4")
    if ex.exists():
        gr.Examples(
            [[str(ex), "seedance", "fast", False, 1.0]],
            [src, target, quality, invert, gamma],
            label="Try it with the demo clip (12 s, a corridor fight rendered by Seedance)",
        )
    gr.Markdown(
        """
<sub>ReShot is Apache-2.0, open-sourced by <a href="https://www.maosika.com">Maosika 猫斯卡</a>, the AI short-drama
production system. Model: Video Depth Anything Small (Apache-2.0, ByteDance, CVPR 2025).</sub>
"""
    )

if __name__ == "__main__":
    demo.queue().launch()
