"""Planning is pure arithmetic — test it without a model."""

from reshot.planning import (
    HOST_RAM_BUDGET_RATIO,
    estimate_host_bytes,
    memory_verdict,
    processing_max_res,
)


def test_processing_res_puts_short_side_at_input_size():
    # 736×1280 portrait: short side 736 → 518 means long side 1280·518/736 = 900.87 → 900 (even)
    assert processing_max_res(736, 1280) == 900
    assert processing_max_res(1280, 736) == 900


def test_processing_res_never_upscales():
    assert processing_max_res(400, 300) == 400


def test_estimate_is_base_plus_linear_in_pixel_frames():
    base = estimate_host_bytes(0, 100, 100)
    one = estimate_host_bytes(1, 100, 100) - base
    assert estimate_host_bytes(2, 100, 100) - base == 2 * one
    assert estimate_host_bytes(1, 200, 100) - base == 2 * one


def test_estimate_matches_measured_4090_runs():
    """Calibration points from 2026-09-12 (see planning.py); the padded model must sit
    above every measurement but within ~25 % of it."""
    measured = [(48, 900, 518, 2.12), (294, 900, 518, 3.90), (96, 864, 496, 2.36), (289, 864, 496, 3.69)]
    for frames, w, h, gib in measured:
        est = estimate_host_bytes(frames, w, h) / 2**30
        assert gib <= est <= gib * 1.25, (frames, w, h, gib, est)


def test_verdict_refuses_over_budget_and_suggests_frames():
    ram = 8 * 2**30
    v = memory_verdict(10_000, 900, 518, ram_bytes=ram)
    assert not v["ok"]
    assert v["budget"] == int(ram * HOST_RAM_BUDGET_RATIO)
    # The suggested frame count itself must fit the budget
    assert estimate_host_bytes(v["max_frames_ok"], 900, 518) <= v["budget"]


def test_verdict_passes_small_job():
    assert memory_verdict(8, 320, 180, ram_bytes=8 * 2**30)["ok"]
