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


def test_max_res_caps_output_not_inference(tmp_path):
    """`--max-res` must not shrink what the model sees (0.3.2 regression lock: a 320-px cap
    used to feed the model a thumbnail that it upsampled internally)."""
    src = _clip(tmp_path / "src.mp4", frames=4, w=1280, h=736)
    p = plan(RunConfig(input=src, output=tmp_path / "o.mp4", backend="fake", target="h3", max_res=320, quality="full"))
    assert p.proc_h == 518 and p.proc_w == 900  # model resolution
    p = plan(RunConfig(input=src, output=tmp_path / "o.mp4", backend="fake", target="h3", max_res=320))
    assert p.proc_h == 364  # default quality is fast
    assert (p.out_w, p.out_h) == (320, 160)  # output capped, multiple of 32


def test_cli_batch_loads_model_once_and_skips_bad_clip(tmp_path, monkeypatch):
    """`reshot a.mp4 b.mp4 missing.mp4 -o dir/`: one backend for the whole batch, per-clip
    outputs named after the inputs, a bad clip reported and skipped, exit code of the failure."""
    import reshot.pipeline as pl
    from reshot.cli import main

    a, b = _clip(tmp_path / "a.mp4"), _clip(tmp_path / "b.mp4", frames=8)
    calls = []
    real = pl.get_backend

    def counting(name, **kw):
        calls.append(name)
        return real(name, **kw)

    monkeypatch.setattr(pl, "get_backend", counting)
    out = tmp_path / "depth"
    code = main(
        [
            str(a),
            str(b),
            str(tmp_path / "missing.mp4"),
            "-o",
            str(out),
            "--backend",
            "fake",
            "--metrics",
            str(tmp_path / "m"),
        ]
    )
    assert code == 2  # InputError for the missing clip
    assert calls == ["fake"]  # one model for three clips
    assert (out / "a_depth.mp4").exists() and (out / "b_depth.mp4").exists()
    assert (tmp_path / "m" / "a.json").exists() and (tmp_path / "m" / "b.json").exists()
    assert not (out / "missing_depth.mp4").exists()


def test_quality_resolution(tmp_path, monkeypatch):
    """fast is the default; full is the user's call (with a warning on a small card);
    an explicit --input-size beats --quality."""
    import reshot.pipeline as pl

    def cfg(**kw):
        return RunConfig(input=tmp_path / "x.mp4", output=tmp_path / "y.mp4", backend="fake", **kw)

    steps = []

    class Rep:
        def step(self, tag, msg):
            steps.append(tag)

    monkeypatch.setattr(pl, "_cuda_total_bytes", lambda device: 8 * 2**30)
    assert pl.resolve_input_size(cfg(), "cuda", Rep()) == 364 and steps == []
    assert pl.resolve_input_size(cfg(quality="full"), "cuda", Rep()) == 518 and steps == ["warn"]
    assert pl.resolve_input_size(cfg(input_size=700), "cuda") == 700
    monkeypatch.setattr(pl, "_cuda_total_bytes", lambda device: 24 * 2**30)
    steps.clear()
    assert pl.resolve_input_size(cfg(quality="full"), "cuda", Rep()) == 518 and steps == []
    assert pl.resolve_input_size(cfg(), "cpu") == 364
    with pytest.raises(InputError):
        cfg(quality="auto")


def test_model_input_resolution_matches_the_vendored_resize():
    """The resolutions printed in the docs are the ones the model really sees."""
    from reshot.pipeline import model_input_resolution as mir

    assert mir(1280, 720, 364) == (644, 364) and mir(1280, 720, 518) == (924, 518)
    assert mir(720, 1280, 364) == (364, 644) and mir(720, 1280, 518) == (518, 924)
    assert mir(1024, 768, 364) == (490, 364) and mir(1024, 768, 518) == (686, 518)
    assert mir(1080, 1080, 518) == (518, 518)
