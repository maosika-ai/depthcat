<h1 align="center">depthcat</h1>
<p align="center"><b>Copy the shot, not the actors.</b></p>
<p align="center">Turn a reference video into a depth blockout your video model can follow — same staging, same camera, your cast, your style.</p>

<p align="center"><img src="docs/demo.gif" width="560" alt="left: reference video, right: the depth blockout depthcat makes from it"></p>

<p align="center">
<a href="https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml"><img src="https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
<img src="https://img.shields.io/badge/python-3.10%20%7C%203.12-blue" alt="python">
<img src="https://img.shields.io/badge/runs%20on-CUDA%20%C2%B7%20Apple%20Silicon%20%C2%B7%20CPU-555" alt="platforms">
</p>
<p align="center"><a href="README.zh-CN.md">中文</a> · <a href="docs/USAGE.md">User guide</a> · <a href="https://huggingface.co/spaces/maosika/depthcat">Hugging Face</a> · <a href="CHANGELOG.md">Changelog</a></p>

---

## The problem

You found a clip with exactly the blocking you want — she walks in from the door, he
struggles in the background, the camera pushes in slowly. Now try to get that out of a
video model:

- **Prompts don't do staging.** "Woman in foreground, man behind her, slow push-in" comes
  out different every run.
- **Feeding the clip itself copies too much.** Faces, wardrobe, lighting and style come
  along — and real-people footage gets rejected by most models' content checks.

## What depthcat does

It strips the reference down to the one thing you actually wanted from it: **where things
are, how big they are, how they and the camera move** — as a near-white / far-black depth
video, the format depth ControlNets are trained on.

```bash
pip install git+https://github.com/maosika-ai/depthcat
depthcat reference.mp4 -o blockout.mp4 --target h3
```

```
reference.mp4 ──▶ depthcat ──▶ blockout.mp4 ──┐
                                               ├──▶ video model (depth ControlNet) ──▶ your clip
your prompt: cast, wardrobe, lighting, style ──┘
```

Feed `blockout.mp4` to MiniMax H3 Fun ControlNet, Wan VACE or any depth-conditioned
generator, describe the *look* in the prompt, and the result keeps the reference's
staging and camera with none of its identity.

## Made for

- **Short-drama and film teams** — reuse a shot's blocking with your own characters.
- **Ads and product video** — put the camera move of a reference spot on your product.
- **Anyone iterating on a scene** — lock staging once, change cast and style freely.

## Why not …

| alternative | what you give up |
|---|---|
| Depth capture inside a hosted video app | Works only in that app; nothing to script, batch, or plug into your own pipeline. |
| A single-image depth node per frame in ComfyUI | Flickers — each frame is normalised on its own. You also build fps, size and encoding yourself. |
| A hosted depth-video API | Pay per second and upload your footage. Fine for a test, not for a pipeline. |
| **depthcat** | A CLI and a Python function. Video-native model, whole-clip normalisation, generator presets, Apache-2.0 end to end. Runs on an 8 GB GPU or a Mac. |

## Details that matter

- **Temporal consistency from the model, not a blur.** Video Depth Anything runs 32-frame
  windows with overlap and aligns them; single-image models cannot do this.
- **One normalisation for the whole clip.** Per-frame min/max makes a static wall change
  brightness when someone walks toward the camera; the generator reads that as motion.
- **Inference at model resolution, output at yours.** A 12 s 720p clip needs ~4 GB of RAM,
  not 8–20 GB. The RAM need is estimated before running and refused if it won't fit.
- **Frames by timestamp, size by crop, quality by CRF.** 30 → 24 fps really gives 24; frame
  size is cropped to the generator's multiple, never padded; x264 CRF 12 so the control
  signal has no banding.
- **Licence you can ship with.** Apache-2.0 model by default; the non-commercial variants
  are opt-in and say so out loud.

## Presets

| `--target` | fps | frame size | max | for |
|---|---|---|---|---|
| `h3` | 24 | multiple of 32 | 15 s | [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) |
| `wan` | 16 | multiple of 16 | – | Wan 2.1 VACE (values from the docs, not yet verified end-to-end) |
| `none` | source | even | – | anything |

## Measured

| device | precision | ms / frame | host RAM |
|---|---|---|---|
| RTX 4090, torch 2.8 cu128 | fp16 | **70** (31 with `--input-size 364`) | 4.5 GB |
| Apple M2 Max, torch 2.9.1 | fp32 | ~500 | ~4 GB |

294 frames at 736×1280, Small model. Details and more platforms in the [user guide](docs/USAGE.md).

## Python

```python
from pathlib import Path
from depthcat import RunConfig, run

result = run(RunConfig(input=Path("reference.mp4"), output=Path("blockout.mp4"), target="h3"))
print(result.frames, result.ms_per_frame)
```

## Licence

Code: Apache-2.0. Default weights: Video-Depth-Anything-**Small**, Apache-2.0
([DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything),
CVPR 2025; model code vendored under `depthcat/third_party/`, see [NOTICE](NOTICE)).
Base/Large weights are CC-BY-NC-4.0 and only load when asked for explicitly.

## Roadmap

`--people-only` (SAM 2 silhouettes) · pose / canny / normal outputs from the same pass ·
PyPI · Docker · ComfyUI example workflow · a runnable Hugging Face Space

## Contributing

Issues and PRs welcome — [CONTRIBUTING.md](CONTRIBUTING.md). `ruff check` and `pytest`
run in CI on Ubuntu and Windows, plus the real model on CPU.

<p align="center"><sub>Built at <a href="https://www.maosika.com">Maoska 猫斯卡</a>, a studio that ships AI short drama every day.</sub></p>
