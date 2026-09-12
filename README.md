<h1 align="center">ReShot</h1>
<p align="center"><b>Copy the shot, not the actors.</b></p>
<p align="center"><sub>Open-source depth-map tool from <a href="https://www.maosika.com">Maosika 猫斯卡</a>, the AI short-drama production system.</sub></p>

<p align="center"><img src="docs/demo.gif" width="720" alt="a fight scene, its depth map, and a new fight generated from it"></p>
<p align="center"><sub>A fight scene. Its depth map. A new take — different fighters, same choreography, same camera.</sub></p>

<p align="center"><a href="README.zh-CN.md">中文</a> · <a href="docs/USAGE.md">User guide</a> · <a href="https://huggingface.co/spaces/maosika/reshot">Hugging Face</a> · <a href="CHANGELOG.md">Changelog</a></p>
<p align="center">
<a href="https://github.com/maosika-ai/reshot/actions/workflows/ci.yml"><img src="https://github.com/maosika-ai/reshot/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
<img src="https://img.shields.io/badge/runs%20on-CUDA%20%C2%B7%20Apple%20Silicon%20%C2%B7%20CPU-555" alt="platforms">
</p>

---

## Every shot is two things.

Where everyone stands. How big they are. The way they move. The way the camera moves.
That's the **staging** — and it's what makes a great shot great.

Then there's who they are, what they wear, how the light falls. That's the **look**.

Until now you couldn't take one without the other. Describe the staging in words and the
model gives you something else every time. Hand it the footage and it takes the faces,
the wardrobe, the style — everything you didn't ask for.

## Keep the choreography. Change everything else.

ReShot turns a reference clip into a depth map: a clean, silent record of the
staging and nothing else. Near is white, far is black, every frame in step with the last.
No faces. No costumes. No style. Just the shot.

Technically: monocular video depth estimation. The model predicts relative inverse depth
for every frame; ReShot normalises it once over the whole clip to 8-bit grey (near = white)
and encodes it as a standard depth-map video.

Give the depth map to your video model along with a prompt for the look, and you get the
reference's blocking and camera with your cast in it.

```bash
pip install git+https://github.com/maosika-ai/reshot
reshot reference.mp4 -o depth.mp4 --target seedance
```

That's the whole thing.

## Works with the models you already use.

**Seedance 2.0 and 2.5.** Add the depth map as a reference video and ask for its motion
and camera. The depth map meets the API's reference-video requirements out of the box —
24 fps, H.264, sized for the model — and because it carries no likeness, it passes the
content checks that stop real footage.

```
参考@视频1的动作与运镜。两名武者在雨夜屋顶对决，黑色劲装，冷蓝月光，电影感。
```

**MiniMax H3 Fun ControlNet.** Use the depth map as the depth condition. 24 fps, frame size
a multiple of 32, up to 15 seconds — the preset handles it.

**Wan VACE, and any depth ControlNet.** It's a standard near-white depth video. If your
model reads depth, it reads this.

| `--target` | fps | frame size | length | |
|---|---|---|---|---|
| `seedance` | 24 | ×16, ≥ 407,696 px | ≤ 15 s | Seedance 2.0 / 2.5 reference video |
| `h3` | 24 | ×32 | ≤ 15 s | MiniMax H3 Fun ControlNet |
| `wan` | 16 | ×16 | – | Wan 2.1 VACE |
| `none` | source | even | – | anything |

## Steady by design.

A depth map is only useful if the model trusts it. So ReShot is built around the details
that make a depth video hold still.

- **It sees the whole clip.** The model works on overlapping windows of 32 frames and
  aligns them, so depth doesn't jitter from frame to frame.
- **One scale for the entire shot.** Brightness means distance, and it means the same
  distance in frame 1 and frame 300. A wall doesn't pulse because someone walked past it.
- **Nothing the model didn't ask for.** Frames are picked by timestamp, so 24 fps is
  really 24. Frame size is trimmed to the model's grid, never padded with black. The
  encode is clean enough that the depth gradient has no banding.

## Made for the pace of a studio.

A 12-second shot takes about 20 seconds on an RTX 4090 — 70 ms a frame, 31 with the
fast setting. It runs on a MacBook. It needs about 4 GB of memory for a 720p clip, and it
tells you before it starts if a clip won't fit.

Batch a folder of references in a shell loop, or call it from Python:

```python
from pathlib import Path
from reshot import RunConfig, run

run(RunConfig(input=Path("reference.mp4"), output=Path("depth.mp4"), target="seedance"))
```

## Yours to ship.

ReShot is Apache-2.0, and so is the model it runs by default — Video Depth Anything
Small, from ByteDance (CVPR 2025), vendored under `reshot/third_party/`. Use it in a
product, a pipeline, a service. The larger research-only variants exist and are clearly
labelled; they never load unless you ask.

## Get started

```bash
pip install git+https://github.com/maosika-ai/reshot          # ffmpeg on PATH
pip install "reshot[ffmpeg] @ git+https://github.com/maosika-ai/reshot"   # or bundle one
reshot --help
```

Weights (111 MB) download on first run. Behind a firewall: `export HF_ENDPOINT=https://hf-mirror.com`.

Everything else — every option, presets, recipes for each model, troubleshooting — is in
the **[user guide](docs/USAGE.md)**. Issues and pull requests are welcome; see
[CONTRIBUTING.md](CONTRIBUTING.md).

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
