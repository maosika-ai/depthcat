"""ffmpeg round-trip on synthetic frames — no model."""

import cv2
import numpy as np
import pytest

from video2blockout.io import read_video, write_gray_video

def _has_ffmpeg():
    try:
        from video2blockout.io import ffmpeg_binary

        return bool(ffmpeg_binary())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_ffmpeg(), reason="no ffmpeg (PATH or imageio-ffmpeg)")


def test_gray_video_roundtrip(tmp_path):
    t, h, w = 12, 64, 96
    ramp = np.linspace(0, 255, w, dtype=np.float32)
    gray = np.broadcast_to(ramp, (t, h, w)).astype(np.uint8).copy()
    out = write_gray_video(gray, tmp_path / "g.mp4", fps=24)

    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == t
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (h, w)
    # yuv420p + x264 at CRF 12 should keep a smooth ramp within a couple of levels
    err = np.abs(frame[:, :, 0].astype(int) - gray[0].astype(int))
    assert err.mean() < 2.0


def test_write_rejects_odd_dimensions(tmp_path):
    with pytest.raises(ValueError):
        write_gray_video(np.zeros((1, 63, 64), np.uint8), tmp_path / "x.mp4", 24)


def test_read_video_resamples_by_timestamp(tmp_path):
    # 30 fps, 30 frames → 1 s; ask for 24 fps and expect 24 frames, not 30 (stride=1 trap)
    p = tmp_path / "src.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 48))
    for i in range(30):
        vw.write(np.full((48, 64, 3), i * 8, np.uint8))
    vw.release()
    frames, fps = read_video(p, target_fps=24)
    assert fps == 24 and len(frames) == 24


def test_read_video_caps_longer_side(tmp_path):
    p = tmp_path / "big.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 24, (400, 200))
    for _ in range(3):
        vw.write(np.zeros((200, 400, 3), np.uint8))
    vw.release()
    frames, _ = read_video(p, max_res=100)
    assert frames.shape[1:3] == (50, 100)


def test_write_accepts_frame_iterator_with_size(tmp_path):
    frames = (np.full((32, 48), i * 7, np.uint8) for i in range(6))
    out = write_gray_video(frames, tmp_path / "it.mp4", 24, size=(32, 48))
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    cap.release()


def test_probe_video(tmp_path):
    p = tmp_path / "p.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 48))
    for _ in range(5):
        vw.write(np.zeros((48, 64, 3), np.uint8))
    vw.release()
    from video2blockout.io import probe_video

    w, h, fps, n = probe_video(p)
    assert (w, h, n) == (64, 48, 5) and abs(fps - 25) < 0.01
