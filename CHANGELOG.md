# Changelog

## 0.3.4 — 2026-09-12
- **Small GPUs work out of the box.** Measured on an RTX 3080 Ti: the default `--input-size 518`
  peaks at 7.4 GiB allocated / 10.9 GiB reserved and OOMs on an 8 GB card; 364 peaks at
  2.25 / 3.0 GiB. Cards under 11.5 GB now get 364 automatically (reported as a `vram` step,
  recorded as `input_size_used` in metrics); an explicit `--input-size` is always respected.
- Metrics gain `gpu_peak_allocated_bytes` / `gpu_peak_reserved_bytes`; hidden
  `--cuda-memory-fraction` emulates a smaller card for verification.
- README: a "What you need" table with the measured numbers.

## 0.3.3 — 2026-09-12
- Memory estimate is now a measured line, not a guess: peak RSS = 2.0 GiB + 20 B × frames ×
  processing pixels (fitted on six RTX 4090 runs, base 1.73 GiB / slope 16.9 B, residual
  ≤ 0.05 GiB, padded ~15 %). The old flat 32 B/pixel-frame over-refused long clips and
  under-estimated short ones; `--max-frames` suggestions now account for the base.
- Measured on the same runs: 61–62 ms/frame on a 4090 for clips ≥ 190 frames (77–92 ms/frame
  for very short clips, where model load and the first window dominate).

## 0.3.2 — 2026-09-12
- **Fix: `--max-res` no longer lowers the inference resolution.** It capped the processing
  size too, so `--max-res 320` fed the model a thumbnail; inference now always runs at model
  resolution and `--max-res` only sizes the output (regression test added).
- **Batch mode**: `reshot a.mp4 b.mp4 … -o dir/` (or any `-o` that is a directory) runs
  every clip on one model load; `--metrics`/`--npz` become per-clip directories; a failed clip
  is skipped and reported. Python: `run_many()`.
- Memory: `to_gray` converts frame by frame (−1.2 GB peak for a 12 s clip, byte-identical
  output); decoding writes into one preallocated buffer instead of list + `np.stack` (−400 MB).
- `--npz` / `--metrics` create their parent directories. `torch.load(weights_only=True)`;
  inference under `torch.inference_mode()`; `extract()` honours `RESHOT_FAKE_BACKEND`.

## 0.3.1 — 2026-09-12
- Fix: the package failed to import on Windows since 0.2.0 (`import resource` is Unix-only); peak-memory metric now uses the Win32 counter there.
- README rewritten as a manual: the problem → what it does → four steps with the real prompts and character sheets from the demo (`docs/prompts/`, `docs/refs/`) → what transfers and what doesn't → presets → developers.
- New demo: a corridor fight → depth map → three generated takes (two humans, one bear), `docs/demo-fight.gif` + `docs/demo-fight.mp4`; social preview card updated.
- Wording: the output is called what it is — a **depth map** (relative inverse depth, monocular
  video depth estimation), no longer a "blockout" (that term means untextured 3D geometry).
  Docs, CLI help, Space page and demo labels updated; `demo.make_blockout` → `make_depth_map`.

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
