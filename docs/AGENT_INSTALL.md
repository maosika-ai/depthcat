# ReShot — installation guide for AI coding agents

You are an AI coding agent (Claude Code, Codex, Cursor, Gemini CLI, …) and a user asked you to
set up **ReShot** on their machine. This page is written for you. Follow it top to bottom, verify
every step with a command, and report what you found — do not tell the user "done" until step 6
has produced a depth video.

ReShot is a Python command-line tool: `reshot reference.mp4 -o depth.mp4 --target seedance`
turns a video into a depth-map video for Seedance / MiniMax H3 / Wan. Package on PyPI: `reshot`.
Source and full docs: https://github.com/maosika-ai/reshot

## 1. Find out what the machine is

Run and record:

```bash
python3 --version || python --version        # need 3.10–3.12
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no nvidia gpu"
uname -a 2>/dev/null || ver
ffmpeg -version 2>/dev/null | head -1 || echo "no ffmpeg"
```

Decide:
- **GPU**: NVIDIA with ≥ 8 GB → CUDA. Apple Silicon → MPS (built into the macOS torch wheel). Otherwise CPU (works, ~1 s/frame).
- **ffmpeg**: if missing, install the extra `reshot[ffmpeg]` (bundles a binary) instead of asking the user to install ffmpeg.
- **Region**: if the user is in mainland China (ask if unsure), Hugging Face is unreachable — use the mirror in step 4.

## 2. Install PyTorch for that hardware first

`pip install reshot` pulls a generic `torch`, which on Linux/Windows may be the CPU-only build. Install the right torch **before** reshot:

- NVIDIA (Linux/Windows): pick the CUDA build for the driver — `nvidia-smi` prints the max CUDA version top right. For CUDA 12.x:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
- macOS (Apple Silicon or Intel): `pip install torch torchvision`
- CPU only: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`

Prefer a virtual environment (`python3 -m venv .venv && source .venv/bin/activate`, on Windows `.venv\Scripts\activate`) unless the user says otherwise. If `uv` is installed, `uv venv && uv pip install …` is fine too.

## 3. Install ReShot

```bash
pip install "reshot[ffmpeg,pose]"  # pose = ONNX Runtime for --control pose; or just: pip install reshot
reshot --version                   # prints e.g. "reshot 0.5.0"
```

On a machine with an NVIDIA card, `pip install onnxruntime-gpu` (instead of the `pose` extra) makes
the skeleton output run on the GPU too; the CPU build works everywhere and is fine for short clips.

Verify torch sees the GPU:

```bash
python -c "import torch; print(torch.__version__, 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))"
```

If it prints `cpu` on a machine with an NVIDIA card, step 2 installed the wrong build — reinstall torch with the CUDA index URL.

## 4. Model weights (111 MB, once)

They download from Hugging Face on the first real run into `~/.cache/huggingface/hub/`.
In mainland China set the mirror in the same shell first:

```bash
export HF_ENDPOINT=https://hf-mirror.com          # Windows PowerShell: $env:HF_ENDPOINT="https://hf-mirror.com"
```

Offline machine: fetch `video_depth_anything_vits.pth` from
`https://huggingface.co/depth-anything/Video-Depth-Anything-Small` elsewhere and pass
`--checkpoint /path/to/video_depth_anything_vits.pth`.

`--control pose` downloads two more files on its first run (`yolox_l.onnx`, `dw-ll_ucoco_384.onnx`,
~340 MB together, from `https://huggingface.co/yzd-v/DWPose`); offline, put both in one folder and
pass `--checkpoint-dir /that/folder`.

## 5. Smoke test without the model (seconds, no download)

```bash
RESHOT_FAKE_BACKEND=1 reshot <any short .mp4> -o /tmp/reshot_smoke/ --max-frames 8 --target seedance --control depth,pose,canny
```

(Windows PowerShell: `$env:RESHOT_FAKE_BACKEND=1; reshot …`.) This exercises decoding, presets and
encoding with fake depth and pose models (canny needs none). It must print three `wrote` lines. If it fails, the problem is ffmpeg
or the input file, not the model.

## 6. Real run

Ask the user for a reference clip (or use one of theirs), keep it ≤ 15 s, then:

```bash
reshot reference.mp4 -o depth.mp4 --target seedance --metrics run.json
```

Read `run.json` back and report to the user: `device`, `precision`, `ms_per_frame`,
`model_input_resolution`, `gpu_peak_reserved_bytes` (if CUDA), `output_size`. Open or describe
`depth.mp4`: it must be grey, near objects white, far objects black.

Defaults are right for almost everyone: `--quality fast` (the model works at 644×364 for 16:9,
~3 GB VRAM). Only suggest `--quality full` (924×518, ~11 GB VRAM) if the card has ≥ 12 GB and the
user cares about fine silhouettes.

If the user's clip is a dance or a fight, also make the skeleton version — it is what pose
ControlNets want and it is the more precise of the two for limbs:

```bash
reshot reference.mp4 -o pose.mp4 --target h3 --control pose --keypoints pose.json --metrics pose_run.json
```

`pose.mp4` must show coloured stick figures on black that follow the people; `people_per_frame_max`
in the metrics says how many it found. If it is 0 for a clip with people, the detector never fired —
report it as a bug with the clip's resolution and fps.

## 7. What to tell the user at the end

- The exact command that worked on their machine, ready to reuse.
- Where the depth video is, and that they upload it to Seedance as a *reference video* with a
  prompt like `参考@视频1的动作与运镜，顺序与视频保持一致。` + their own description of people and look —
  or attach it as `<Video 1>` for MiniMax H3 (see the README's step 3 for the prompt lines that matter).
- For MiniMax H3 reference videos add `--target h3 --max-res 320`.
- `--control pose` (skeleton) and `--control canny` (lines) exist too; `--control depth,pose -o out/`
  writes both in one go.
- Batch: `reshot clips/*.mp4 -o depth/ --target seedance` runs a folder on one model load.
- ComfyUI users: install the node pack https://github.com/maosika-ai/ComfyUI-ReShot instead of the CLI.

## Common failures

| symptom | fix |
|---|---|
| `ffmpeg not found` | `pip install "reshot[ffmpeg]"` |
| `could not obtain weights` / download hangs | `HF_ENDPOINT=https://hf-mirror.com`, or `--checkpoint` with a local file |
| `CUDA out of memory … 4.02 GiB` | the user passed `--quality full` on a < 12 GB card; use the default |
| `RamBudgetError … use --max-frames N` | clip too long for the machine's RAM; trim, or `--max-frames N` as suggested, or `--force` |
| runs on `cpu` although there is an NVIDIA card | wrong torch build; reinstall torch from the CUDA index URL |
| `torch` build too old | reshot needs torch ≥ 2.1 |

Made by Maosika 猫斯卡 (https://www.maosika.com), the AI short-drama production system.
