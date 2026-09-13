"""End-to-end with the real Small model on CPU. Opt-in (downloads 111 MB):

RUN_MODEL_TESTS=1 pytest tests/test_model_smoke.py
"""

import os

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(not os.environ.get("RUN_MODEL_TESTS"), reason="set RUN_MODEL_TESTS=1")


def _synthetic_clip(path, frames=8, w=224, h=160):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h))
    for i in range(frames):
        img = np.full((h, w, 3), 40, np.uint8)
        # a bright square that drifts right, on a gradient background
        img[:, :, 1] = np.linspace(20, 200, w, dtype=np.uint8)
        x = 20 + i * 8
        cv2.rectangle(img, (x, 50), (x + 60, 120), (230, 230, 230), -1)
        vw.write(img)
    vw.release()


def test_extract_and_write(tmp_path):
    from reshot import extract, to_gray, write_gray_video

    src = tmp_path / "src.mp4"
    _synthetic_clip(src)
    depths, fps = extract(str(src), device="cpu")
    assert depths.shape[0] == 8 and depths.ndim == 3
    assert np.isfinite(depths).all() and depths.max() > depths.min()
    gray = to_gray(depths)
    assert gray.dtype == np.uint8 and gray.max() == 255 and gray.min() == 0
    out = write_gray_video(gray, tmp_path / "out.mp4", fps)
    assert out.stat().st_size > 1000


def test_cli_end_to_end(tmp_path):
    from reshot.cli import main

    src = tmp_path / "src.mp4"
    _synthetic_clip(src)
    out = tmp_path / "out.mp4"
    metrics = tmp_path / "m.json"
    assert main([str(src), "-o", str(out), "--device", "cpu", "--target", "h3", "--metrics", str(metrics)]) == 0
    assert out.exists() and metrics.exists()


def test_pose_cli_end_to_end_finds_a_person(tmp_path):
    """Real DWPose on the CPU over the first 8 frames of the repo's demo clip (its top-left
    quadrant is the reference footage, a person in a corridor). Downloads ~340 MB once."""
    import json
    from pathlib import Path

    from reshot.cli import main

    src = Path(__file__).resolve().parents[1] / "docs" / "demo-fight.mp4"
    out, metrics, kp = tmp_path / "pose.mp4", tmp_path / "m.json", tmp_path / "kp.json"
    rc = main(
        [str(src), "-o", str(out), "--control", "pose", "--device", "cpu", "--target", "h3",
         "--max-frames", "8", "--max-res", "720", "--metrics", str(metrics), "--keypoints", str(kp)]
    )  # fmt: skip
    assert rc == 0 and out.exists()
    m = json.loads(metrics.read_text())
    assert m["control"] == "pose" and m["people_per_frame_max"] >= 1, m
    frames = json.loads(kp.read_text())["frames"]
    assert len(frames) == 8 and any(p["scores"][1] > 0.3 for p in frames[0]["people"])  # a neck was seen
    cap = cv2.VideoCapture(str(out))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.max() > 100  # coloured skeleton, not a black video
