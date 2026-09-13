---
title: ReShot
emoji: 🎬
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: "5.49.1"
python_version: "3.12"
app_file: app.py
pinned: false
license: apache-2.0
short_description: Copy the shot, not the actors. Video → depth map. By Maosika.
---

# ReShot

Copy the shot, not the actors. ReShot turns a reference video into a depth map — the control
signal that lets a video model copy a shot's **staging and camera** without copying its faces,
wardrobe or style.

- Presets for Seedance 2.0 / 2.5 reference video, MiniMax H3 Fun ControlNet, Wan VACE.
- Code, CLI, local web page, docs: https://github.com/maosika-ai/reshot (Apache-2.0)
- Model: Video Depth Anything Small (Apache-2.0, ByteDance, CVPR 2025)

Open-sourced by [Maosika 猫斯卡](https://www.maosika.com) — the AI short-drama production system:
from a one-line idea to episodic script, character sheets, scene images and every shot rendered
with Seedance 2.0 / 2.5 and MiniMax H3.
