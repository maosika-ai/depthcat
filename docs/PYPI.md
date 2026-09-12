# ReShot

**Copy the shot, not the actors.**

ReShot turns a reference video into a depth-map video (near = white, far = black), so Seedance, MiniMax H3 or Wan can repeat its choreography and camera moves with your own characters.

```bash
pip install reshot
reshot reference.mp4 -o depth.mp4 --target seedance
```

- Presets for **Seedance 2.0 / 2.5** reference video (`--target seedance`), **MiniMax H3** (`h3`), **Wan 2.1 VACE** (`wan`).
- `--quality fast` (default; the model sees 644×364 for 16:9, ~3 GB VRAM) or `full` (924×518, ~11 GB).
- Depth normalised once over the whole clip — no flicker; frames picked by timestamp; sizes cropped to the model's grid, never padded.
- Batch a folder in one model load: `reshot clips/*.mp4 -o depth/`.
- Python API: `from reshot import RunConfig, run`.
- Apache-2.0, including the default model (Video Depth Anything Small).
- ComfyUI nodes: https://github.com/maosika-ai/ComfyUI-ReShot

Full guide, demo, measured performance and the prompts that made the demo takes: **https://github.com/maosika-ai/reshot**

Open-sourced by [Maosika 猫斯卡](https://www.maosika.com), the AI short-drama production system — from a one-line idea to episodic script, character sheets, scene images and every shot rendered with Seedance and MiniMax H3. https://www.maosika.com
