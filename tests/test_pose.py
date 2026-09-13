"""Pose and canny: the layout conversion, the renderer, tracking / smoothing / hysteresis,
and the whole pipeline with the fake pose backend — no model, no download."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from reshot import RunConfig, run
from reshot.backends.fake_pose import FakePoseBackend
from reshot.edges import canny_frame
from reshot.pose import PoseClip
from reshot.pose.skeleton import LAYOUT, VISIBLE_SCORE, body_bbox, render_frame, wholebody_to_openpose
from reshot.pose.tracking import OneEuro, apply_hysteresis, smooth, track


def _clip(path: Path, frames=12, w=200, h=120, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        vw.write(np.full((h, w, 3), (i * 10) % 256, np.uint8))
    vw.release()
    return path


# ── layout ────────────────────────────────────────────────────────────────────────


def test_wholebody_to_openpose_inserts_neck_and_reorders():
    kps = np.arange(133 * 2, dtype=np.float32).reshape(1, 133, 2)
    scs = np.full((1, 133), 0.9, np.float32)
    out_k, out_s = wholebody_to_openpose(kps, scs)
    assert out_k.shape == (1, 134, 2) and out_s.shape == (1, 134)
    # neck = mean of COCO shoulders 5 and 6, lands at OpenPose index 1
    assert np.allclose(out_k[0, 1], (kps[0, 5] + kps[0, 6]) / 2)
    # OpenPose 2 = right shoulder = COCO 6; OpenPose 8 = right hip = COCO 12
    assert np.allclose(out_k[0, 2], kps[0, 6]) and np.allclose(out_k[0, 8], kps[0, 12])
    # nose stays at 0; hands are untouched (shifted by one for the neck)
    assert np.allclose(out_k[0, 0], kps[0, 0]) and np.allclose(out_k[0, 92], kps[0, 91])
    assert LAYOUT["hand_right"] == [113, 134]


def test_neck_invisible_when_a_shoulder_is():
    kps = np.zeros((1, 133, 2), np.float32)
    scs = np.full((1, 133), 0.9, np.float32)
    scs[0, 5] = 0.1
    _, out_s = wholebody_to_openpose(kps, scs)
    assert out_s[0, 1] == 0.0


def test_empty_frame_round_trips():
    k, s = wholebody_to_openpose(np.zeros((0, 133, 2), np.float32), np.zeros((0, 133), np.float32))
    assert k.shape == (0, 134, 2) and s.shape == (0, 134)
    assert render_frame(k, s, 64, 96).sum() == 0


# ── renderer ──────────────────────────────────────────────────────────────────────


def test_render_draws_body_hands_and_optionally_face():
    clip = FakePoseBackend().infer(np.zeros((1, 240, 320, 3), np.uint8), 24.0)
    k, s = clip.keypoints[0], clip.scores[0]
    body_only = render_frame(k, s, 240, 320, hands=False, face=False)
    with_hands = render_frame(k, s, 240, 320, hands=True, face=False)
    with_face = render_frame(k, s, 240, 320, hands=True, face=True)
    assert body_only.dtype == np.uint8 and body_only.shape == (240, 320, 3)
    assert body_only.any()  # something was drawn
    assert (with_hands != body_only).any() and (with_face != with_hands).any()
    # OpenPose colour scheme: the neck→right-shoulder limb (index 0) is red
    assert (body_only[..., 0] > body_only[..., 1]).any()


def test_render_scales_keypoints_from_source_to_output_size():
    clip = FakePoseBackend().infer(np.zeros((1, 240, 320, 3), np.uint8), 24.0)
    k, s = clip.keypoints[0], clip.scores[0]
    small = render_frame(k, s, 120, 160, src_w=320, src_h=240)
    big = render_frame(k, s, 240, 320)

    # same drawing at half the size: the centroid of the drawn pixels lands at half the position
    def centroid(img):
        ys, xs = np.nonzero(img.max(-1))
        return xs.mean(), ys.mean()

    (bx, by), (sx, sy) = centroid(big), centroid(small)
    assert abs(bx / 2 - sx) < 3 and abs(by / 2 - sy) < 3


def test_hidden_joints_are_not_drawn():
    clip = FakePoseBackend().infer(np.zeros((1, 240, 320, 3), np.uint8), 24.0)
    k, s = clip.keypoints[0], clip.scores[0].copy()
    full = render_frame(k, s, 240, 320, hands=False)
    s[:, 8:14] = 0.0  # hide both legs
    legless = render_frame(k, s, 240, 320, hands=False)
    assert (legless > 0).sum() < (full > 0).sum()
    assert body_bbox(k[0], s[0]) is not None and body_bbox(k[0], np.zeros(134)) is None


# ── tracking / smoothing / hysteresis ─────────────────────────────────────────────


def test_track_keeps_ids_across_frames_and_swapped_order():
    clip = FakePoseBackend().infer(np.zeros((6, 240, 320, 3), np.uint8), 24.0)
    # swap the two people in frame 3: ids must follow the bodies, not the slots
    clip.keypoints[3] = clip.keypoints[3][::-1].copy()
    clip.scores[3] = clip.scores[3][::-1].copy()
    track(clip)
    assert [list(i) for i in clip.track_ids[:3]] == [[0, 1]] * 3
    assert list(clip.track_ids[3]) == [1, 0]
    assert list(clip.track_ids[4]) == [0, 1]


def test_track_starts_a_new_id_after_a_gap():
    clip = PoseClip(width=320, height=240, fps=24.0)
    person = FakePoseBackend().infer(np.zeros((1, 240, 320, 3), np.uint8), 24.0)
    k, s = person.keypoints[0][:1], person.scores[0][:1]
    for t in range(20):
        present = t < 5 or t >= 15  # gone for 10 frames > max_gap
        clip.keypoints.append(k if present else np.zeros((0, 134, 2), np.float32))
        clip.scores.append(s if present else np.zeros((0, 134), np.float32))
    track(clip, max_gap=8)
    assert clip.track_ids[0][0] == 0 and clip.track_ids[15][0] == 1


def test_one_euro_removes_jitter_and_follows_motion():
    rng = np.random.default_rng(0)
    f = OneEuro(24.0, min_cutoff=1.0, beta=0.5)
    still = np.array([100.0, 100.0])
    outs = [f(still + rng.normal(0, 2, 2)) for _ in range(48)]
    assert np.std(np.array(outs[24:]), axis=0).max() < 2.0  # less than the input noise
    f2 = OneEuro(24.0)
    xs = [f2(np.array([float(i * 20), 0.0])) for i in range(24)]  # 20 px/frame sprint
    assert abs(xs[-1][0] - 23 * 20) < 60  # follows fast motion with bounded lag


def test_smooth_keeps_shapes_and_changes_little():
    clip = FakePoseBackend().infer(np.zeros((12, 240, 320, 3), np.uint8), 24.0)
    raw = [k.copy() for k in clip.keypoints]
    smooth(clip)
    assert all(a.shape == b.shape for a, b in zip(raw, clip.keypoints, strict=True))
    assert clip.keypoints[0].tolist() == raw[0].tolist()  # first frame passes through
    assert np.abs(clip.keypoints[6] - raw[6]).max() < 0.2 * min(240, 320)


def test_hysteresis_holds_a_joint_that_dips_briefly():
    clip = FakePoseBackend().infer(np.zeros((8, 240, 320, 3), np.uint8), 24.0)
    assert clip.scores[3][0, 4] == pytest.approx(0.1)  # the fake's dropout frame
    clip.scores[3][0, 4] = 0.25  # between off (0.2) and on (0.3): should be held
    apply_hysteresis(clip)
    assert clip.scores[3][0, 4] > VISIBLE_SCORE
    clip2 = FakePoseBackend().infer(np.zeros((8, 240, 320, 3), np.uint8), 24.0)
    apply_hysteresis(clip2)
    assert clip2.scores[3][0, 4] == pytest.approx(0.1)  # 0.1 < off: really gone


# ── canny ─────────────────────────────────────────────────────────────────────────


def test_canny_frame_finds_an_edge_and_resizes():
    img = np.zeros((120, 200, 3), np.uint8)
    img[:, 100:] = 255
    e = canny_frame(img, 60, 96)
    assert e.shape == (60, 96) and e.dtype == np.uint8
    col = e.sum(axis=0)
    assert col.argmax() in range(44, 52)  # the edge sits at the middle column
    assert set(np.unique(e)) <= {0, 255}


# ── pipeline with the fake backends ───────────────────────────────────────────────


def test_run_pose_end_to_end(tmp_path):
    src = _clip(tmp_path / "src.mp4", frames=12, w=200, h=120)
    cfg = RunConfig(
        input=src,
        output=tmp_path / "pose.mp4",
        backend="fake",
        control="pose",
        target="h3",
        keypoints=tmp_path / "kp.json",
        metrics=tmp_path / "m.json",
    )
    res = run(cfg)
    assert (res.width, res.height) == (192, 96) and res.fps == 24
    assert res.output.exists() and res.output.stat().st_size > 500
    cap = cv2.VideoCapture(str(res.output))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (96, 192, 3) and frame.max() > 100  # coloured skeleton, not black
    kp = json.loads((tmp_path / "kp.json").read_text())
    assert kp["format"] == "reshot-pose/1" and len(kp["frames"]) == res.frames
    assert len(kp["frames"][0]["people"]) == 2 and len(kp["frames"][0]["people"][0]["keypoints"]) == 134
    assert {p["id"] for p in kp["frames"][5]["people"]} == {0, 1}
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["control"] == "pose" and m["people_per_frame_max"] == 2


def test_run_canny_end_to_end(tmp_path):
    src = _clip(tmp_path / "src.mp4", frames=6, w=200, h=120)
    res = run(RunConfig(input=src, output=tmp_path / "canny.mp4", control="canny", target="seedance"))
    assert (res.width, res.height) == (192, 112) and res.output.exists()
    assert res.metrics["control"] == "canny" and res.metrics["canny"] == [100, 200]


def test_plan_reads_at_output_size_for_pose_and_canny(tmp_path):
    from reshot import plan

    src = _clip(tmp_path / "src.mp4", frames=6, w=1920, h=1080)
    p_depth = plan(RunConfig(input=src, output=tmp_path / "o.mp4", backend="fake"))
    p_pose = plan(RunConfig(input=src, output=tmp_path / "o.mp4", backend="fake", control="pose"))
    assert p_depth.proc_h < 720  # depth: model resolution
    assert (p_pose.proc_w, p_pose.proc_h) == (1280, 720)  # pose: the output size
    assert p_pose.ram_estimate < p_depth.ram_estimate


def test_cli_multi_control_writes_one_file_each(tmp_path):
    from reshot.cli import main

    src = _clip(tmp_path / "src.mp4")
    out = tmp_path / "out"
    assert main([str(src), "-o", str(out), "--control", "depth,pose,canny", "--backend", "fake"]) == 0
    assert sorted(p.name for p in out.glob("*.mp4")) == ["src_canny.mp4", "src_depth.mp4", "src_pose.mp4"]


def test_cli_rejects_bad_control_and_canny(tmp_path):
    from reshot.cli import main

    src = _clip(tmp_path / "src.mp4")
    assert main([str(src), "-o", str(tmp_path / "o.mp4"), "--control", "hed", "--backend", "fake"]) == 2
    assert main([str(src), "-o", str(tmp_path / "o.mp4"), "--control", "canny", "--canny", "x"]) == 2
    from reshot.errors import InputError

    with pytest.raises(InputError):
        RunConfig(input=src, output=tmp_path / "o.mp4", control="canny", canny_low=300, canny_high=200)


def test_pose_detect_every_reaches_the_backend(tmp_path, monkeypatch):
    """The CLI flag must land in the backend constructor — it silently did not once."""
    import reshot.pipeline as pl

    seen = {}

    def fake_get_backend(name, **kw):
        seen.update(name=name, **kw)
        return FakePoseBackend()

    monkeypatch.setattr(pl, "get_backend", fake_get_backend)
    monkeypatch.delenv("RESHOT_FAKE_BACKEND", raising=False)  # test_web sets it at import time
    src = _clip(tmp_path / "src.mp4", frames=3)
    run(RunConfig(input=src, output=tmp_path / "o.mp4", control="pose", pose_detect_every=5))
    assert seen["name"] == "dwpose" and seen["detect_every"] == 5
