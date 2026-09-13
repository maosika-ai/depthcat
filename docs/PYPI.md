# ReShot

**Copy the shot, not the actors.**

ReShot turns a reference video into a depth-map video (near = white, far = black), an OpenPose-style skeleton video or a canny line video, so Seedance, MiniMax H3 or Wan can repeat its choreography and camera moves with your own characters.

```bash
pip install "reshot[pose]"
reshot                                               # web page: opens in your browser, drop the clip in
reshot reference.mp4 -o depth.mp4 --target seedance  # or the command line
reshot reference.mp4 -o pose.mp4  --target h3 --control pose      # skeletons (DWPose), hands included
reshot reference.mp4 -o out/ --control depth,pose,canny           # all three at once
```

- **Web page** (`reshot` with nothing after it): everything runs on your computer; pick depth map / skeleton / lines, reference and result side by side, download, and the prompt line for Seedance / MiniMax H3 to copy.
- **Skeletons** tracked across frames (identity, visibility hysteresis, One-Euro smoothing) so they don't flicker; `--keypoints` dumps them as JSON.
- Presets for **Seedance 2.0 / 2.5** reference video (`--target seedance`), **MiniMax H3** (`h3`), **Wan 2.1 VACE** (`wan`).
- `--quality fast` (default; the model sees 644×364 for 16:9, ~3 GB VRAM) or `full` (924×518, ~11 GB).
- Depth normalised once over the whole clip — no flicker; frames picked by timestamp; sizes cropped to the model's grid, never padded.
- Batch a folder in one model load: `reshot clips/*.mp4 -o depth/`.
- Python API: `from reshot import RunConfig, run`.
- Apache-2.0, including the default model (Video Depth Anything Small).
- ComfyUI nodes: https://github.com/maosika-ai/ComfyUI-ReShot

Full guide, demo, measured performance and the prompts that made the demo takes: **https://github.com/maosika-ai/reshot**

Open-sourced by [Maosika 猫斯卡](https://www.maosika.com), the AI short-drama production system — from a one-line idea to episodic script, character sheets, scene images and every shot rendered with Seedance and MiniMax H3. The full story with demo takes: [www.maosika.com/en/reshot](https://www.maosika.com/blog/reshot-open-source-depth-motion-capture-seedance-minimax-h3) · [中文](https://www.maosika.com/blog/reshot-open-source-depth-motion-capture)
