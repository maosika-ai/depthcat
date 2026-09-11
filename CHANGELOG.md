# Changelog

## 0.1.0 — 2026-09-11
- First release: `depthcat in.mp4 -o out.mp4` on Video Depth Anything Small (Apache-2.0).
- Depth estimated at model resolution, whole-clip normalisation, near = white, x264 CRF 12.
- `--target h3` / `wan` presets (fps, size multiple, max length), `--npz`, `--metrics`.
- Host-RAM estimate with refusal over 50 % of physical RAM; MPS allocator cap on Apple Silicon.
- Measured: RTX 4090 fp16 70 ms/frame; Apple M2 Max fp32 ~500 ms/frame.
