# Changelog

## 0.3.0 — 2026-09-12
- **Renamed: depthcat → ReShot.** Package, CLI, env var and error class follow:
  `pip install git+https://github.com/maosika-ai/reshot`, `reshot in.mp4 -o out.mp4`,
  `RESHOT_FAKE_BACKEND`, `reshot.ReshotError`. No behaviour change.
- README/Space copy: who makes it — [Maosika 猫斯卡](https://www.maosika.com), the AI short-drama production system.

## 0.2.1 — 2026-09-11
- `--target seedance`: 24 fps, frame size a multiple of 16, ≤ 15 s, and a warning when the frame
  falls below the 407,696-pixel floor — all from the official Seedance reference-video spec.
- `Target.min_pixels` and the matching pipeline warning.
- README rewritten as a product page (Seedance first); Seedance recipe in the usage guide; demo default preset is now `seedance`.

## 0.2.0 — 2026-09-11
- One pipeline: `reshot.run(RunConfig)` is the single code path for the CLI and the Python API.
- Backend registry (`vda`, `fake`); `--backend fake` runs the whole pipeline without a model.
- Typed errors with exit codes (2 input · 3 RAM budget · 4 ffmpeg · 5 weights) and concrete fixes in the message.
- `RunConfig` validates arguments before any work; `plan()` exposes what a run will do.
- Metrics JSON now records the full config and plan.
- Lint (ruff) in CI; 28 tests without a model plus a real-model CPU smoke job.

## 0.1.0 — 2026-09-11
- First release: `reshot in.mp4 -o out.mp4` on Video Depth Anything Small (Apache-2.0).
- Depth estimated at model resolution, whole-clip normalisation, near = white, x264 CRF 12.
- `--target h3` / `wan` presets (fps, size multiple, max length), `--npz`, `--metrics`.
- Host-RAM estimate with refusal over 50 % of physical RAM; MPS allocator cap on Apple Silicon.
- Measured: RTX 4090 fp16 70 ms/frame; Apple M2 Max fp32 ~500 ms/frame.
