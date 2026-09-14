"""Render docs/social-preview.png (1280×640, the card GitHub / X / Slack show for a link).

Left: name, tagline, what it does, install line. Right: one frame of docs/demo-pose.mp4 —
reference | skeleton on top, the three MiniMax H3 takes below — with small labels.

    python scripts/make_social_preview.py            # writes docs/social-preview.png
    python scripts/make_social_preview.py --t 4.6    # pick another moment of the demo
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
W, H = 1280, 640
BG, INK, DIM, LIME = (13, 13, 15), (234, 234, 239), (173, 173, 184), (216, 255, 66)
FONTS = ["/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"]


def font(size: int, cjk: bool = False) -> ImageFont.FreeTypeFont:
    """Helvetica for Latin; Hiragino Sans GB (has arrows and CJK) for lines that need them."""
    for f in (FONTS[1:] if cjk else FONTS):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def demo_frame(t: float) -> Image.Image:
    cap = cv2.VideoCapture(str(ROOT / "docs/demo-pose.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * 24))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("cannot read docs/demo-pose.mp4")
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str) -> None:
    f = font(13)
    tw = draw.textlength(text, font=f)
    x, y = xy
    draw.rectangle([x, y, x + tw + 10, y + 20], fill=(0, 0, 0))
    draw.text((x + 5, y + 3), text, font=f, fill=INK)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t", type=float, default=2.6, help="second of the demo to show")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/social-preview.png")
    a = ap.parse_args()

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    x = 64
    d.text((x, 160), "Re", font=font(84), fill=INK)
    d.text((x + d.textlength("Re", font=font(84)), 160), "Shot", font=font(84), fill=LIME)
    d.text((x, 272), "Copy the shot,\nnot the actors.", font=font(36), fill=INK, spacing=10)
    d.text((x, 400), "video → depth map · skeleton · lines for", font=font(21, cjk=True), fill=DIM)
    d.text((x, 432), "Seedance 2.0/2.5 · MiniMax H3 · Wan VACE", font=font(21), fill=DIM)
    d.text((x, 540), 'pip install "reshot[pose]"', font=font(19), fill=LIME)
    d.text((x, 578), "Apache-2.0  ·  by Maosika 猫斯卡  ·  www.maosika.com", font=font(17, cjk=True), fill=DIM)

    frame = demo_frame(a.t)  # 1440×688 → fit into 640×306 on the right
    pw = 640
    ph = round(frame.height * pw / frame.width)
    frame = frame.resize((pw, ph), Image.LANCZOS)
    px, py = W - pw - 40, (H - ph) // 2
    img.paste(frame, (px, py))
    d = ImageDraw.Draw(img)
    top_h = round(412 * pw / 1440)
    label(d, (px + 4, py + top_h - 26), "Reference")
    label(d, (px + pw // 2 + 4, py + top_h - 26), "Skeleton · ReShot")
    for i in range(3):
        label(d, (px + i * pw // 3 + 4, py + ph - 26), f"Take {i + 1} · MiniMax H3")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(a.out, optimize=True)
    print("wrote", a.out, img.size)


if __name__ == "__main__":
    main()
