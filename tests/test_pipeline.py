"""End-to-end through `run()` with the fake backend: exercises planning, decoding,
presets, cropping, encoding, npz and metrics — everything except the model."""

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from reshot import RunConfig, RunResult, plan, run
from reshot.errors import InputError, RamBudgetError


def _clip(path: Path, frames=12, w=200, h=120, fps=30):
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        vw.write(np.full((h, w, 3), (i * 10) % 256, np.uint8))
    vw.release()
    return path


def test_run_end_to_end_with_h3_preset(tmp_path):
    src = _clip(tmp_path / "src.mp4")
    cfg = RunConfig(
        input=src,
        output=tmp_path / "out.mp4",
        backend="fake",
        target="h3",
        npz=tmp_path / "d.npz",
        metrics=tmp_path / "m.json",
    )
    res = run(cfg)
    assert isinstance(res, RunResult)
    # 200x120 → multiples of 32 → 192x96; 30 fps → 24 fps → 12 frames become ~10
    assert (res.width, res.height) == (192, 96)
    assert res.fps == 24 and 9 <= res.frames <= 10
    assert res.output.exists() and res.output.stat().st_size > 500
    z = np.load(tmp_path / "d.npz")
    assert z["depths"].shape[0] == res.frames
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["output_size"] == [192, 96] and m["config"]["target"] == "h3"


def test_plan_is_cheap_and_consistent(tmp_path):
    src = _clip(tmp_path / "src.mp4", frames=30, w=640, h=360)
    p = plan(RunConfig(input=src, output=tmp_path / "o.mp4", backend="fake"))
    assert (p.src_w, p.src_h, p.src_frames) == (640, 360, 30)
    assert p.proc_h == 360  # short side already ≤ 518: no downscale
    assert p.ram_estimate > 0 and p.ram_budget > 0


def test_ram_budget_refuses_then_force_allows(tmp_path, monkeypatch):
    import reshot.pipeline as pl

    src = _clip(tmp_path / "src.mp4")
    monkeypatch.setattr(pl, "memory_verdict", lambda *_: {"estimate": 10, "budget": 1, "ok": False, "max_frames_ok": 1})
    with pytest.raises(RamBudgetError) as ei:
        run(RunConfig(input=src, output=tmp_path / "o.mp4", backend="fake"))
    assert ei.value.exit_code == 3 and "--max-frames 1" in str(ei.value)
    run(RunConfig(input=src, output=tmp_path / "o.mp4", backend="fake", force=True))


def test_config_validation_is_upfront(tmp_path):
    with pytest.raises(InputError):
        RunConfig(input=tmp_path / "x.mp4", output=tmp_path / "o.mp4", target="nope")
    with pytest.raises(InputError):
        RunConfig(input=tmp_path / "x.mp4", output=tmp_path / "o.mp4", input_size=500)


def test_missing_input_is_a_user_error(tmp_path):
    with pytest.raises(InputError):
        run(RunConfig(input=tmp_path / "missing.mp4", output=tmp_path / "o.mp4", backend="fake"))


def test_cli_exit_codes(tmp_path):
    from reshot.cli import main

    src = _clip(tmp_path / "src.mp4")
    assert main([str(src), "-o", str(tmp_path / "o.mp4"), "--backend", "fake"]) == 0
    assert main([str(tmp_path / "nope.mp4"), "-o", str(tmp_path / "o.mp4"), "--backend", "fake"]) == 2
