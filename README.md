<p align="center"><img src="docs/demo.gif" width="560" alt="left: source video, right: depth blockout produced by depthcat"></p>

<h1 align="center">depthcat</h1>

<p align="center"><b>Turn any video into a depth blockout — the control signal that lets a video model copy a shot's staging and camera, without copying its faces, clothes or style.</b></p>

<p align="center">
<a href="https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml"><img src="https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml/badge.svg" alt="ci"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-blue.svg" alt="Apache-2.0"></a>
<img src="https://img.shields.io/badge/python-3.10%20%7C%203.12-blue" alt="python">
<img src="https://img.shields.io/badge/runs%20on-CUDA%20%C2%B7%20Apple%20Silicon%20%C2%B7%20CPU-555" alt="platforms">
</p>

<p align="center"><a href="README.zh-CN.md">中文</a> · <a href="docs/USAGE.md">User guide</a> · <a href="CHANGELOG.md">Changelog</a></p>

---

```bash
pip install git+https://github.com/maosika-ai/depthcat
depthcat reference.mp4 -o blockout.mp4 --target h3
```

That is the whole workflow. Feed `blockout.mp4` to a depth-conditioned video model
(MiniMax H3 Fun ControlNet, Wan VACE, …) together with your own prompt, and the
generated clip follows the reference's **blocking and camera** while its **look** comes
entirely from your prompt.

## Why a depth blockout

A reference video carries two kinds of information: *how the shot is staged* (who is
where, how big, how they move, how the camera moves) and *what it looks like* (faces,
wardrobe, lighting, style). Depth keeps the first and discards the second. That makes it

- **the cleanest control signal** for "shoot my script like this reference",
- **a compliance layer** — no likeness, no wardrobe, no brand leaves the reference,
- **model-agnostic** — every depth ControlNet was trained on exactly this kind of image.

depthcat produces that signal correctly and reproducibly, with the details that ad-hoc
pipelines get wrong already handled.

## What it gets right

| | depthcat | typical "run a depth model per frame" script |
|---|---|---|
| Temporal consistency | native video model (32-frame windows, overlap, cross-window alignment) | flickers |
| Normalisation | once over the whole clip | per frame → static walls "breathe" |
| Resolution | inference at model resolution, output upscaled frame by frame | full-res float depth for every frame → 8–20 GB RAM |
| Frame rate | picks frames by timestamp (30 → 24 really is 24) | integer stride → still 30 |
| Output size | crops to the generator's multiple (32 for H3) | pads → black border reads as a wall |
| Encoding | x264 CRF 12, key frame every 2 s | default CRF → banding → jitter in the generated clip |
| Licence | Apache-2.0 model by default, non-commercial variants opt-in with a warning | whatever was on disk |
| Safety | RAM estimate before running; refuses with a concrete `--max-frames` | swaps the machine to death |

## Install

```bash
pip install git+https://github.com/maosika-ai/depthcat      # needs ffmpeg on PATH
pip install "depthcat[ffmpeg] @ git+https://github.com/maosika-ai/depthcat"   # bundles an ffmpeg binary
```

Weights (111 MB, Apache-2.0) download from Hugging Face on first run.
Behind the Great Firewall: `export HF_ENDPOINT=https://hf-mirror.com`.

## Use

```bash
depthcat in.mp4 -o out.mp4                    # keep fps and size, near = white
depthcat in.mp4 -o out.mp4 --target h3        # MiniMax H3 ControlNet: 24 fps, ×32, ≤ 15 s
depthcat in.mp4 -o out.mp4 --npz d.npz --metrics run.json
```

```python
from pathlib import Path
from depthcat import RunConfig, run

result = run(RunConfig(input=Path("in.mp4"), output=Path("out.mp4"), target="h3"))
print(result.ms_per_frame, result.peak_rss_bytes)
```

Every option, preset, exit code and recipe: **[docs/USAGE.md](docs/USAGE.md)**.

## Targets

| `--target` | fps | frame size | max | for |
|---|---|---|---|---|
| `none` | source | even | – | anything |
| `h3` | 24 | multiple of 32 | 15 s | [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) |
| `wan` | 16 | multiple of 16 | – | Wan 2.1 VACE (values from the docs; not yet verified end-to-end) |

## Performance (measured, Small model, 294 frames @ 736×1280)

| device | precision | ms / frame | host RAM peak |
|---|---|---|---|
| RTX 4090, torch 2.8 cu128 | fp16 | **70** (31 with `--input-size 364`) | 4.5 GB |
| Apple M2 Max, torch 2.9.1 | fp32 | ~500 | ~4 GB |
| Apple M2 Max, torch 2.6.0 | fp32 | ~1700 | — |
| CPU (M2 Max) | fp32 | ~1850 | — |

fp16 is enabled on CUDA only: on MPS it is pathologically slow. Upgrade torch to ≥ 2.9 on Apple Silicon.

## Models and licences

| weights | licence | commercial |
|---|---|---|
| Video-Depth-Anything-**Small** (default) | Apache-2.0 | ✅ |
| Video-Depth-Anything-Base / -Large | CC-BY-NC-4.0 | ❌ opt-in with `--model`, prints a warning |

depthcat's own code is Apache-2.0. The model code from
[DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
(CVPR 2025) is vendored under `depthcat/third_party/` — see [NOTICE](NOTICE).

## Roadmap

- `--people-only`: keep person silhouettes, flatten the background (SAM 2)
- `--also pose,canny,normal`: the other MiniMax H3 Fun ControlNet inputs from one pass
- Docker image, PyPI package, ComfyUI example workflow
- Hugging Face Space demo

## Contributing

Issues and PRs welcome — see [CONTRIBUTING.md](CONTRIBUTING.md). Run `ruff check` and
`pytest` before pushing; the CI also runs the real model on CPU.

Built at [Maoska (猫斯卡)](https://www.maosika.com), an AI short-drama studio.
