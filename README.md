# depthcat

[![ci](https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml/badge.svg)](https://github.com/maosika-ai/depthcat/actions/workflows/ci.yml) [![license](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Turn any video into a **depth "blockout" video** — the grey near-white / far-black
clip that video-generation ControlNets (MiniMax H3 Fun ControlNet, Wan VACE, …) use to
copy the *staging and camera* of a reference shot without copying its faces, clothes
or style.

<p align="center"><img src="docs/demo.gif" width="560" alt="left: source, right: depth blockout"></p>

One command, one model download (Apache-2.0), runs on an 8 GB GPU or an Apple M-series.

[中文说明](README.zh-CN.md)

## Install

```bash
pip install git+https://github.com/maosika-ai/depthcat
# needs ffmpeg on PATH (or: pip install "depthcat[ffmpeg]")
```

Weights (111 MB) download from Hugging Face on first run. In China set
`HF_ENDPOINT=https://hf-mirror.com`.

## Use

```bash
depthcat in.mp4 -o blockout.mp4                 # keep fps/size, near = white
depthcat in.mp4 -o blockout.mp4 --target h3     # MiniMax H3: 24 fps, ×32 dims, ≤15 s
depthcat in.mp4 -o blockout.mp4 --npz depth.npz # also keep raw float depth
```

```python
from depthcat import extract, to_gray, write_gray_video
depths, fps = extract("in.mp4")                 # float32 [T, H, W], larger = closer
write_gray_video(to_gray(depths), "out.mp4", fps)
```

| Option | Default | What it does |
|---|---|---|
| `--model small\|base\|large` | `small` | Only `small` is Apache-2.0. `base`/`large` are CC-BY-NC and print a warning. |
| `--target none\|h3\|wan` | `none` | fps + frame-size preset for a generator (see below). |
| `--fps N` | source | Override output fps. Frames are picked by timestamp, so 30 → 24 really gives 24. |
| `--max-res N` | 1280 | Cap the longer side before inference; output is written at that size. |
| `--invert` | off | far = white instead of near = white. |
| `--clip PCT` | 0 | Percentile clip on both tails so one hot pixel can't crush contrast. |
| `--gamma G` | 1.0 | > 1 darkens mid-tones (more separation near camera). |
| `--crf N` | 12 | x264 quality. Deliberately generous — banding in a control video becomes jitter in the output. |
| `--npz PATH` | – | Save raw float depth for your own post-processing. |

### Targets

| `--target` | fps | frame size | max length | for |
|---|---|---|---|---|
| `none` | source | even | – | anything |
| `h3` | 24 | multiple of 32 (centre-crop) | 15 s | [MiniMax-H3-Fun-Controlnet-Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union) depth input |
| `wan` | 16 | multiple of 16 | – | Wan 2.1 VACE control video |

We crop rather than pad: a padded black border reads to the generator as a far wall.

## What it gets right (that ad-hoc pipelines get wrong)

1. **One normalisation for the whole clip.** Per-frame min/max makes a static wall
   change brightness whenever someone walks toward the camera; the generator reads that
   as the scene breathing. Depth is scaled once over all frames.
2. **Temporal consistency comes from the model, not a blur.** Video Depth Anything runs
   32-frame windows with 10-frame overlap and aligns scale/shift across windows. We call
   that path as-is instead of re-chunking outside it.
3. **Near is white**, the convention depth ControlNets were trained on. `--invert` if
   your tool wants the opposite.
4. **fps by timestamp, size by crop, quality by CRF.** See table above.

## Performance (measured end-to-end, Small model, 294 frames @ 736×1280)

| device | precision | speed | note |
|---|---|---|---|
| Apple M2 Max, torch 2.9.1 | fp32 | ~500 ms/frame (12 s clip in 2.7 min) | fp16 on MPS is pathologically slow — we force fp32 there |
| Apple M2 Max, torch 2.6.0 | fp32 | ~1700 ms/frame | 3.5× slower — **upgrade torch ≥ 2.9 on Apple Silicon** |
| CPU (M2 Max) | fp32 | ~1850 ms/frame | works, slow |
| NVIDIA RTX 4090 (AutoDL), torch 2.8 cu128 | fp16 | **70 ms/frame**, 31 ms with `--input-size 364` | host RAM peak 4.5 GB (was 8.2 before the model-resolution change) |
| NVIDIA A100 (upstream figure) | fp16 | ~8 ms/frame | 6.8 GB VRAM at 32-frame batch |

## Models and licences

| weights | licence | commercial |
|---|---|---|
| `Video-Depth-Anything-Small` (default) | Apache-2.0 | ✅ |
| `Video-Depth-Anything-Base` / `-Large` | CC-BY-NC-4.0 | ❌ opt-in only, with warning |

Code is Apache-2.0. The model code from
[DepthAnything/Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything)
is vendored under `depthcat/third_party/` (see `NOTICE`).

## ComfyUI

There is already a good node — [ComfyUI-Video-Depth-Anything](https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything).
This project is for people who want a CLI / Python API, a licence-safe default, and the
target presets.

## Roadmap

- `--people-only`: keep person silhouettes, flatten the background (SAM 2)
- `--also pose,canny,normal`: the other MiniMax H3 Fun ControlNet inputs from one pass
- Depth Anything 3 (per-frame + smoothing) and ViGeo backends
- Gradio demo / Hugging Face Space

## Credits

Video Depth Anything — Chen et al., CVPR 2025. Built at [Maoska](https://www.maosika.com).
