# ReShot — user guide

[中文版](USAGE.zh-CN.md)

1. [Install](#install)
2. [Command line](#command-line)
3. [Python API](#python-api)
4. [Recipes](#recipes)
5. [How it works](#how-it-works)
6. [Troubleshooting](#troubleshooting)
7. [FAQ](#faq)

## Install

Requirements: Python ≥ 3.10, PyTorch ≥ 2.1 (CPU, CUDA or Apple MPS), ffmpeg.

```bash
# with your own ffmpeg on PATH
pip install git+https://github.com/maosika-ai/reshot

# no ffmpeg? this extra bundles a static binary
pip install "reshot[ffmpeg] @ git+https://github.com/maosika-ai/reshot"
```

PyTorch is not pinned to a CUDA version; install the build for your machine first if
`pip` picks the wrong one (see https://pytorch.org/get-started/locally/).

Model weights (111 MB) download from Hugging Face on first run into `~/.cache/huggingface`.
Offline machines: copy that cache, or pass `--checkpoint /path/video_depth_anything_vits.pth`.
In China: `export HF_ENDPOINT=https://hf-mirror.com`.

## Command line

```
reshot INPUT -o OUTPUT [options]
```

### Model

| option | default | notes |
|---|---|---|
| `--model small\|base\|large` | `small` | Only `small` is Apache-2.0. `base`/`large` are CC-BY-NC-4.0: allowed for research, a warning is printed. |
| `--device auto\|cuda\|mps\|cpu` | `auto` | fp16 on CUDA, fp32 elsewhere (fp16 on MPS is unusably slow). |
| `--input-size N` | `518` | Model short side, multiple of 14. `364` is ~2× faster with visibly softer edges. |
| `--checkpoint PATH` | – | Local `.pth`, skips the download. |

### Output

| option | default | notes |
|---|---|---|
| `--target none\|seedance\|h3\|wan` | `none` | fps + frame-size preset, see [Targets](#targets). |
| `--fps N` | preset or source | Frames are chosen by timestamp, so 30 → 24 yields 24, not 30. Never upsamples. |
| `--max-res N` | `1280` | Cap on the output's longer side. Inference always runs at model resolution regardless. |
| `--max-frames N` | all | Stop after N source frames. |
| `--invert` | off | far = white. Default is near = white (what depth ControlNets expect). |
| `--clip PCT` | `0` | Trim PCT % from each tail before scaling to 0–255, so one hot pixel can't crush contrast. `0` = exact min/max. |
| `--gamma G` | `1.0` | > 1 darkens mid-tones: more separation close to the camera. |
| `--crf N` | `12` | x264 quality. Generous on purpose — banding in a control video becomes jitter in the generated clip. |
| `--npz PATH` | – | Raw float32 depth at processing resolution (`depths[T,H,W]`, `fps`). |
| `--metrics PATH` | – | JSON with config, plan, timings, peak RSS, output size. |

### Safety

| option | notes |
|---|---|
| `--force` | Run even when the host-RAM estimate exceeds half of physical RAM. |

### Exit codes

| code | meaning | typical fix |
|---|---|---|
| 0 | done | |
| 1 | unexpected error (a bug — please report with `-v` output) | |
| 2 | bad input or arguments | check the path / the option value in the message |
| 3 | RAM estimate over budget | use the suggested `--max-frames`, split the clip, or `--force` |
| 4 | ffmpeg missing | `pip install imageio-ffmpeg` or install ffmpeg |
| 5 | weights could not be obtained | set `HF_ENDPOINT` to a mirror, or `--checkpoint` |

### Targets

| `--target` | fps | frame size | max length | notes |
|---|---|---|---|---|
| `none` | source | even | – | |
| `seedance` | 24 | multiple of 16 (centre crop), ≥ 407,696 px (warns) | 15 s (warns) | Seedance 2.0 / 2.5 reference video (`@视频N` motion + camera) |
| `h3` | 24 | multiple of 32 (centre crop) | 15 s (warns) | MiniMax-H3-Fun-Controlnet-Union depth input |
| `wan` | 16 | multiple of 16 | – | Wan 2.1 VACE control video; values from the docs, not yet verified end-to-end |

Frame size is **cropped**, never padded: a padded black border reads to the generator as a far wall.

## Python API

```python
from pathlib import Path
from reshot import RunConfig, run, plan

cfg = RunConfig(input=Path("in.mp4"), output=Path("out.mp4"), target="h3", metrics=Path("m.json"))
print(plan(cfg))          # sizes, frame counts, RAM estimate — no decoding yet
result = run(cfg)         # same code path as the CLI
print(result.frames, result.fps, result.ms_per_frame, result.peak_rss_bytes)
```

Lower-level pieces, for your own post-processing:

```python
from reshot import extract, to_gray, upsample_frames, write_gray_video

depths, fps = extract("in.mp4")                     # float32 [T, H, W], larger = closer
gray = to_gray(depths, clip_percent=0.5, gamma=1.2)  # uint8 [T, H, W], whole-clip normalised
write_gray_video(upsample_frames(gray, 1280, 736), "out.mp4", fps, size=(1280, 736))
```

Errors that are the *user's* to fix are subclasses of `reshot.ReshotError` and carry an
`exit_code`; anything else is a bug.

`RunConfig(..., backend="fake")` runs the entire pipeline with a synthetic depth field and no
model — handy for testing your own integration in milliseconds.

## Recipes

### Seedance 2.0 / 2.5 (reference video)

1. `reshot ref.mp4 -o ref_depth.mp4 --target seedance` (24 fps, ×16, ≤ 15 s).
2. Submit it as a **reference video** (API: `role: "reference_video"`; in the console, attach it as
   a video reference) and address it in the prompt as `@视频1`, asking for its motion and camera:

   ```
   参考@视频1的动作与运镜。两名武者在雨夜屋顶对决，黑色劲装，冷蓝月光，电影感。
   ```

   Describe only the *look* — cast, wardrobe, lighting, style. Staging comes from the video.
3. What the official reference-video spec asks for, and what the preset does about it:

   | official requirement | preset |
   |---|---|
   | mp4 / mov, H.264 or H.265 | H.264 mp4, yuv420p |
   | 24 – 60 fps | 24 fps |
   | 2.0: 2 – 15 s per clip, ≤ 15 s total · 2.5: 2 – 30 s per clip, ≤ 30 s total | warns above 15 s |
   | width × height ≥ 407,696 px, each side 300 – 6,000 px, aspect 0.4 – 2.5 | warns below the pixel floor — raise `--max-res` or use a larger source |
   | ≤ 200 MB | CRF 12 keeps a 15 s 1080p clip well under |

   Seedance 2.5 allows up to 30 s; pass `--fps 24` without `--target` and check the size yourself
   if you need the longer window.

### MiniMax H3 Fun ControlNet (ComfyUI)

1. `reshot ref.mp4 -o ref_depth.mp4 --target h3` (24 fps, ×32, ≤ 15 s).
2. In ComfyUI with the [Fun ControlNet Union](https://huggingface.co/alibaba-pai/MiniMax-H3-Fun-Controlnet-Union)
   weights, load `ref_depth.mp4` as the control video, choose the **depth** condition, write your
   prompt for the *look* (cast, wardrobe, lighting, style). Staging and camera come from the control video.
3. Keep the generator's resolution equal to the control video's, or let the node resize — but do not
   pad.

### Wan 2.1 VACE

`reshot ref.mp4 -o ref_depth.mp4 --target wan`, then use it as the depth control video in your
VACE workflow. Report back if the fps/size values need adjusting — this preset has not been
verified end-to-end yet.

### Batch

```bash
for f in clips/*.mp4; do reshot "$f" -o "out/$(basename "$f" .mp4)_depth.mp4" --target h3 --metrics "out/$(basename "$f" .mp4).json"; done
```

The model loads once per process; for hundreds of clips write a short Python loop with
`extract()` and reuse the backend object:

```python
from reshot.backends import get_backend
from reshot import read_video, to_gray, write_gray_video
b = get_backend("vda", model="small")
for src in sources:
    frames, fps = read_video(src, max_res=900)
    write_gray_video(to_gray(b.infer(frames, fps)), src.with_suffix(".depth.mp4"), fps)
```

### Long videos

Everything is held in RAM for the whole clip (frames + depth). reshot estimates the need and
refuses beyond half of physical RAM with a `--max-frames` suggestion. For a 3-minute clip, split
it into ≤ 15 s pieces first (`ffmpeg -ss … -t 15 …`); the generators want short control videos anyway.

### Apple Silicon

Works out of the box (`--device auto` picks MPS). Two things measured on an M2 Max:
torch 2.9.1 is ~3.5× faster than 2.6.0, and fp16 is disabled on MPS because it does not finish.
Expect ~0.5 s/frame at 736×1280; a 12 s clip takes ~2.5 minutes.

### Reproducible depth for research

`--npz` writes the raw float depth at processing resolution. Combined with `--metrics` (which
records the exact config and plan) a run is fully reproducible.

## How it works

```
input video ─▶ probe ─▶ plan (sizes, frames, RAM) ─▶ decode at model resolution
            ─▶ Video Depth Anything (32-frame windows, 10-frame overlap, cross-window alignment)
            ─▶ whole-clip normalisation → uint8, near = white
            ─▶ per-frame upscale to output size ─▶ centre-crop to target multiple ─▶ x264
```

- **Model**: [Video Depth Anything](https://github.com/DepthAnything/Video-Depth-Anything)
  (CVPR 2025). Its temporal module is what keeps the depth stable across frames; single-image
  models flicker. The model code is vendored unmodified (two import lines) under `third_party/`.
- **Resolution**: the model works on a 518-px short side. Feeding larger frames only inflates the
  float depth kept per frame; reshot decodes at that size and upscales the 8-bit result.
- **Normalisation**: `(d − min) / (max − min)` over *all* frames; optional percentile clip.
- **Encoding**: `libx264 -crf 12 -pix_fmt yuv420p -g 2·fps -movflags +faststart`.

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `ModuleNotFoundError: torchvision` | torch installed without torchvision | `pip install torchvision` (matching your torch build) |
| exit 4 / "ffmpeg not found" | no ffmpeg | `pip install imageio-ffmpeg` or install ffmpeg |
| exit 5 / weights download fails | no route to huggingface.co | `export HF_ENDPOINT=https://hf-mirror.com`, or `--checkpoint` |
| exit 3 | clip too long for RAM | use the suggested `--max-frames`, split the clip |
| very slow on Mac | torch < 2.9, or fp16 forced | `pip install -U torch`; never force fp16 on MPS |
| output is 30 fps although `--fps 24` | source fps < 24 | reshot never upsamples |
| output smaller than source | `--target` crop or `--max-res` | expected; see Targets |
| banding in the depth map | `--crf` raised | keep ≤ 14 |
| people look flat / merged with background | scene has little depth range | try `--clip 0.5 --gamma 1.3` |

## FAQ

**Is Small good enough?** For a control video, yes: the generator reads silhouette, size and
motion, not fine geometry, and Small's silhouettes are clean (see the demo GIF). We have not
benchmarked Base/Large against Small ourselves; they are CC-BY-NC, so Small is also the only
choice for commercial work.

**Why near = white?** Depth ControlNets (H3 Fun, Wan VACE, SD depth) were trained on inverse-depth
maps where closer is brighter. `--invert` if a tool wants the opposite.

**Can it output pose / canny / normals?** Not yet — on the roadmap.

**Does it remove the background / keep only people?** Not yet (`--people-only` is on the roadmap).
The full-frame depth already carries no identity or style.

**GPU memory?** Small in fp16 needs ~2 GB at the default input size; a 4090 is idle most of the time.
