"""Planning is pure arithmetic — test it without a model."""

from video2blockout.planning import (
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


def test_estimate_scales_with_pixels_and_frames():
    assert estimate_host_bytes(2, 100, 100) == 2 * estimate_host_bytes(1, 100, 100)
    assert estimate_host_bytes(1, 200, 100) == 2 * estimate_host_bytes(1, 100, 100)


def test_verdict_refuses_over_budget_and_suggests_frames():
    ram = 8 * 2**30
    v = memory_verdict(10_000, 900, 518, ram_bytes=ram)
    assert not v["ok"]
    assert v["budget"] == int(ram * HOST_RAM_BUDGET_RATIO)
    # The suggested frame count itself must fit the budget
    assert estimate_host_bytes(v["max_frames_ok"], 900, 518) <= v["budget"]


def test_verdict_passes_small_job():
    assert memory_verdict(8, 320, 180, ram_bytes=8 * 2**30)["ok"]
