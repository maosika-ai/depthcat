"""Unit tests that need no model and no GPU."""

import numpy as np
import pytest

from reshot.postprocess import to_gray
from reshot.targets import TARGETS, center_crop, fit_dimensions


def test_global_normalisation_not_per_frame():
    # Frame 0 spans 0..1, frame 1 spans 0..10. Per-frame min/max would map both to
    # full range; global must keep frame 0 dark.
    d = np.zeros((2, 4, 4), np.float32)
    d[0] = np.linspace(0, 1, 16).reshape(4, 4)
    d[1] = np.linspace(0, 10, 16).reshape(4, 4)
    g = to_gray(d)
    assert g[0].max() <= 26  # 1/10 of 255
    assert g[1].max() == 255


def test_near_is_white_by_default_and_invert_flips():
    d = np.array([[[0.0, 5.0]]], np.float32)  # larger = closer (VDA convention)
    assert to_gray(d)[0, 0].tolist() == [0, 255]
    assert to_gray(d, invert=True)[0, 0].tolist() == [255, 0]


def test_clip_percent_ignores_hot_pixel():
    # A 0..1 ramp plus one hot pixel at 1000. Exact min/max lets the hot pixel crush the
    # ramp into the bottom few grey levels; a 2 % tail clip restores its full contrast.
    d = np.tile(np.linspace(0, 1, 100, dtype=np.float32), (1, 10, 1)).reshape(1, 10, 100)
    d[0, 0, 0] = 1000.0
    assert to_gray(d)[0, 5, 99] < 2
    assert to_gray(d, clip_percent=2)[0, 5, 99] >= 250


def test_flat_input_does_not_divide_by_zero():
    assert to_gray(np.ones((1, 2, 2), np.float32)).max() == 0


def test_to_gray_rejects_wrong_rank():
    with pytest.raises(ValueError):
        to_gray(np.zeros((2, 2), np.float32))


@pytest.mark.parametrize(
    "h,w,m,exp",
    [(1280, 736, 32, (1280, 736)), (1080, 1920, 32, (1056, 1920)), (7, 9, 2, (6, 8)), (10, 10, 32, (32, 32))],
)
def test_fit_dimensions_rounds_down_to_multiple(h, w, m, exp):
    assert fit_dimensions(h, w, m) == exp


def test_center_crop_is_centred():
    f = np.arange(6 * 6).reshape(1, 6, 6)
    c = center_crop(f, 4, 4)
    assert c.shape == (1, 4, 4)
    assert c[0, 0, 0] == 7 and c[0, -1, -1] == 28


def test_h3_preset_matches_published_constraints():
    t = TARGETS["h3"]
    assert (t.fps, t.multiple, t.max_seconds) == (24.0, 32, 15.0)


def test_seedance_preset():
    t = TARGETS["seedance"]
    assert (t.fps, t.multiple, t.max_seconds, t.min_pixels) == (24.0, 16, 15.0, 407_696)


def test_to_gray_matches_whole_array_reference_bit_for_bit():
    """to_gray converts frame by frame to save memory; the bytes must equal the
    straightforward whole-array formula for every option combination."""
    rng = np.random.default_rng(0)
    d = rng.random((6, 40, 64), dtype=np.float32) * 10

    def ref(x, invert=False, gamma=1.0):
        lo, hi = float(x.min()), float(x.max())
        n = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
        if gamma != 1.0:
            n = n ** float(gamma)
        if invert:
            n = 1.0 - n
        return (n * 255.0 + 0.5).astype(np.uint8)

    for kw in ({}, {"gamma": 1.6}, {"invert": True}, {"gamma": 0.8, "invert": True}):
        assert np.array_equal(to_gray(d, **kw), ref(d, **kw)), kw
