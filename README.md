<h1 align="center">ReShot</h1>
<p align="center"><b>Copy the shot, not the actors.</b></p>
<p align="center">ReShot turns a reference video into a depth map, so Seedance or MiniMax H3 can repeat its choreography and camera moves — with your own characters in it.</p>

<p align="center"><img src="docs/demo-fight.gif" width="720" alt="a fight scene, its depth map, and three new takes generated from it"></p>
<p align="center"><sub>Top: the reference and its depth map. Bottom: three takes generated from that depth map — two women, one rabbit. Same moves, same camera. <a href="docs/demo-fight.mp4">Full-resolution clip</a>.</sub></p>

<p align="center"><a href="README.zh-CN.md">中文</a> · <a href="docs/USAGE.md">User guide</a> · <a href="https://huggingface.co/spaces/maosika/reshot">Hugging Face</a> · <a href="CHANGELOG.md">Changelog</a></p>
<p align="center">
<a href="https://github.com/maosika-ai/reshot/actions/workflows/ci.yml"><img src="https://github.com/maosika-ai/reshot/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
<img src="https://img.shields.io/badge/runs%20on-CUDA%20%C2%B7%20Apple%20Silicon%20%C2%B7%20CPU-555" alt="platforms">
</p>

---

## The problem

You have a clip whose fight, dance or camera move is exactly what you want in your own AI video. There are two ways to get it, and both fail:

- **Feed the clip to the video model as a reference.** It copies the faces, the clothes and the look along with the moves. If the clip has real people in it, the platform's content check may refuse it outright.
- **Describe the moves in words.** "She kicks off the wall, grabs a pipe, throws the big guy over her shoulder" — the model gives you a different fight every time, and the camera never does what you said.

## What ReShot does

ReShot takes an `.mp4` in and writes an `.mp4` out. The output is a **depth map video**: every frame is grey, near things are white, far things are black. It keeps where everyone stands, how big they are, how they move and how the camera moves. It throws away faces, clothes, lighting and style.

<p align="center"><img src="docs/img/step1_reference.jpg" width="360" alt="reference frame"> <img src="docs/img/step2_depth.jpg" width="360" alt="the same frame as a depth map"></p>

You give that grey video to your video model as the reference and describe the people and the look in the prompt. The model takes the moves from the video and everything else from your words.

Technically: monocular video depth estimation. The model predicts relative inverse depth for every frame; ReShot normalises it once over the whole clip to 8-bit grey (near = white) and encodes it as a standard depth-map video.

## How to use it

The three takes in the demo were made exactly this way. Every file involved is in this repo, so you can repeat it.

### 1. Install

```bash
pip install git+https://github.com/maosika-ai/reshot        # needs ffmpeg on PATH
```

No ffmpeg? `pip install "reshot[ffmpeg] @ git+https://github.com/maosika-ai/reshot"` bundles one. The model weights (111 MB) download on first run.

### 2. Make the depth map

```bash
reshot reference.mp4 -o depth.mp4 --target seedance
```

`--target seedance` sets 24 fps, H.264, a frame size that is a multiple of 16 and at least 407,696 pixels, up to 15 seconds — the reference-video rules of the Seedance API. For MiniMax H3 use `--target h3` (multiples of 32). On an RTX 4090 a 12-second clip takes about 20 seconds; on a MacBook a few minutes. A whole folder at once, one model load: `reshot clips/*.mp4 -o depth/ --target seedance`.

### 3. Give it to the video model

**Seedance 2.0 / 2.5.** Upload `depth.mp4` as a reference video. In the prompt, point at it and describe the people and the look:

```
参考@视频1的动作与运镜，顺序与视频保持一致。
一名穿深绿色丝绒旗袍的女子在狭窄的金属走廊里与三名黑衣守卫搏斗，冷蓝走廊光，红色警示灯，电影感。
```

**MiniMax H3.** Attach `depth.mp4` as `<Video 1>`. If you want a specific face, attach a character sheet as `<Picture 1>`. These are the three sheets used for the demo:

<p align="center"><img src="docs/img/step3_sheets.jpg" width="720" alt="the three character sheets used as Picture 1"></p>

MiniMax H3 wants its prompt in a fixed six-section format. The full prompts for all three takes are in [`docs/prompts/`](docs/prompts/). The part that does the work is how `<Video 1>` is defined and what it is allowed to transfer:

```
<Subject 3> is the fight choreography and camera movement shown in <Video 1>, a grey depth map
in which near objects are white and far objects are black: one fighter leans on a corridor wall
in close-up, kicks off it to tear down a pipe, fights several opponents, is grabbed from behind
by the largest and throws him, slams the last one into a wall panel, wipes the mouth in close-up,
then walks away through a door past the fallen opponents.

<Subject 3>: attribute_transfer - every action, position, timing and camera move of <Video 1>
is transferred onto <Subject 1> and <Subject 2>; its grey depth look is not transferred.
```

Two things in there matter. **Say in words what happens in the grey clip** — the model reads the depth map far better when the prompt tells it what the blobs are doing. **Say that the grey look is not to be copied**, or you may get a grey film back.

### 4. What comes out

<p align="center"><img src="docs/img/step4_takes.jpg" width="720" alt="the three takes"></p>

Left to right: [Jiang Xue](docs/prompts/take1_jiangxue_armor.txt) in bronze armour, [Su Wan](docs/prompts/take2_suwan_qipao.txt) in a green velvet qipao, and a [rabbit boxer](docs/prompts/take3_rabbit_boxer.txt) against a wolf, a tiger and a bear as a 3D animated feature. Same six shots, same close-up at the start, same walk out through the door at the end. The reference clip itself was a Seedance 2.0 text-to-video render (864×496, 12 s) and the takes were made on MiniMax H3, so nobody's likeness was involved at any step.

## What transfers, and what doesn't

**Transfers:** who stands where, how big they are relative to each other, every move and its timing, the cuts, and the camera — push-ins, tracking, handheld shake.

**Doesn't:** faces (use a character sheet), clothes, lighting, colour, props in detail, and anything smaller than a hand. Those come from your prompt and your reference images.

Things we learned making the demo:

- **Keep the reference under 15 seconds** (Seedance's limit) and cut it to the shots you want before running ReShot. Everything in the clip gets copied, including the boring part at the end.
- **For MiniMax H3, make the depth map small: `--target h3 --max-res 320`** (that is 320×176 for a 16:9 clip). A full-size grey silhouette starts to pull the character's face shape towards the person in the reference; a small one carries the moves without the shape.
- **Change the species, keep the size ratio.** The bear take works because the prompt says the bear is "about 1.3× the rabbit, never more than 1.5×". The depth map already says who is bigger; the prompt must not contradict it.
- **Plain clothes on extras, no logos.** Whatever the prompt leaves open, the model fills with text and badges.

## What you need

| | Runs? | Measured |
|---|---|---|
| **NVIDIA, 12 GB or more** (RTX 3080 Ti, 4070, 4090 …) | Yes, `--quality full` | 4090: 62 ms/frame, 11 GB VRAM peak · 3080 Ti: 83 ms/frame |
| **NVIDIA, 8 GB** (RTX 3070, 4060 …) | Yes, `--quality fast` (picked automatically) | 3 GB VRAM, 34 ms/frame on a 3080 Ti |
| **Apple Silicon** | Yes, slower | M2 Max: ~500 ms/frame, a 12 s clip in about 2.5 min |
| **CPU only** | Yes, slow | ~1.8 s/frame |
| **Host RAM** | 16 GB covers clips up to ~27 s at 720p | peak = 2 GB + 224 MB per second of 720p; the tool refuses before starting if a clip won't fit |

**`--quality`** is yours to choose: `auto` (default) takes `full` on cards with 11.5 GB or more and `fast` below that; pass `full` or `fast` to decide yourself. The difference, measured on the same 294-frame clip: the model sees the frame at 518 px vs 364 px on the short side; large shapes — people, props, who is nearer — come out identical, fine detail is softer at `fast` (a loose strand of hair merges into the cheek). Mean difference 5.3 of 255 grey levels, 95 % of pixels within 15, edge energy −4.5 %. For copying blocking and camera, `fast` is enough; for close-ups where fine silhouettes matter, use `full`. Experts can set the exact size with `--input-size`.

Verified on rented cards on 2026-09-12; the numbers are in the `--metrics` output of those runs.

## Presets

| `--target` | fps | frame size | length | for |
|---|---|---|---|---|
| `seedance` | 24 | ×16, ≥ 407,696 px | ≤ 15 s | Seedance 2.0 / 2.5 reference video |
| `h3` | 24 | ×32 | ≤ 15 s | MiniMax H3 (reference video or Fun ControlNet depth) |
| `wan` | 16 | ×16 | – | Wan 2.1 VACE |
| `none` | source | even | – | anything that reads a depth video |

`reshot --help` lists every option; the [user guide](docs/USAGE.md) explains them.

## For developers

```python
from pathlib import Path
from reshot import RunConfig, run

run(RunConfig(input=Path("reference.mp4"), output=Path("depth.mp4"), target="seedance"))
```

Three details make the output something a video model will actually follow:

- **One scale for the whole clip.** Depth is normalised once over all frames, never per frame, so a wall keeps the same grey when someone walks past it. Per-frame normalisation makes the scene "breathe".
- **Frames picked by timestamp.** 30 fps → 24 fps really is 24; nothing is duplicated or dropped in a pattern the model could learn.
- **Cropped, never padded.** Frame size is trimmed to the model's grid. A black border would read as a far wall.

The model is Video Depth Anything Small (ByteDance, CVPR 2025). It works on overlapping 32-frame windows and aligns them, so depth doesn't jitter between frames. A 12-second 720p clip peaks at 3.9 GB of host RAM and 11 GB of VRAM (3 GB with `--input-size 364`) (measured; the estimate the tool shows before starting is a fitted line, within 0.05 GB of measurements) and it refuses up front if a clip won't fit. A `fake` backend runs the whole pipeline without a model for your own tests. Errors that are yours to fix are `ReshotError` subclasses with an exit code and a concrete fix in the message.

## License

Apache-2.0, and so is the default model. The vendored model code is under `reshot/third_party/`. Use it in a product, a pipeline, a service. The larger research-only weights (Base, Large) are CC-BY-NC and never load unless you ask for them.

Issues and pull requests are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md).

## About Maosika 猫斯卡

ReShot is open-sourced by **[Maosika 猫斯卡](https://www.maosika.com)** (www.maosika.com), a
professional **AI video production system for short drama and short video**. Maosika takes a
story from a one-line idea to a finished vertical AI short drama: it writes the episodic
script, designs the characters and scenes, generates consistent character sheets and scene
images, and renders every shot with video models such as **Seedance 2.0 / 2.5** and
**MiniMax H3** — a crew of digital specialists handling each step, so one person can
produce a series that used to take a studio. Individual screenwriters, MCNs and short-drama
companies use Maosika to produce AI short drama every day.

ReShot is the depth-map step of that pipeline, released under Apache-2.0 so anyone
can copy a reference shot's staging and camera into their own AI-generated video. To make
AI short drama, AI short video or AI manhua drama end to end, visit
**[https://www.maosika.com](https://www.maosika.com)**.

<p align="center"><sub>Made by <a href="https://www.maosika.com">Maosika 猫斯卡</a> · AI short drama, produced daily · <a href="https://www.maosika.com">www.maosika.com</a></sub></p>
