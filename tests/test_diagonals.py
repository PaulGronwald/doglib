import math

import pytest

from triplot import diagonals as d


def test_nice_decades_basic():
    vals = d._nice_decades(0.5, 50)
    assert 1.0 in vals and 10.0 in vals
    assert 2.0 in vals and 5.0 in vals
    assert 20.0 in vals and 50.0 in vals
    for v in vals:
        assert 0.5 <= v <= 50


def test_nice_decades_empty_for_invalid():
    assert d._nice_decades(-1, 1) == []
    assert d._nice_decades(10, 1) == []


def test_clip_slope_plus_one_full_viewport():
    # line log v = log f + 0 → passes through (1,1) and (10,10) in lin space
    res = d._clip_slope_line(0.0, +1, (0.0, 2.0), (0.0, 2.0))
    assert res is not None
    (x0, y0), (x1, y1) = res
    assert math.isclose(x0, 0.0) and math.isclose(y0, 0.0)
    assert math.isclose(x1, 2.0) and math.isclose(y1, 2.0)


def test_clip_slope_minus_one():
    # y = -x + 2  crosses (0,2) and (2,0)
    res = d._clip_slope_line(2.0, -1, (0.0, 2.0), (0.0, 2.0))
    assert res is not None
    pts = sorted(res)
    assert math.isclose(pts[0][0], 0.0) and math.isclose(pts[0][1], 2.0)
    assert math.isclose(pts[1][0], 2.0) and math.isclose(pts[1][1], 0.0)


def test_clip_returns_none_when_outside():
    # y = x + 100 never enters [0,1]x[0,1]
    assert d._clip_slope_line(100.0, +1, (0.0, 1.0), (0.0, 1.0)) is None


def test_displacement_segment_roundtrip():
    # d = 1 inch → v = 2π·f·1. At f=1 Hz, v=2π ≈ 6.28
    seg = d.displacement_segment(1.0, (1, 100), (0.1, 1000))
    assert seg is not None
    # both endpoints must satisfy v = 2π f d
    for f, v in [(seg.f0, seg.v0), (seg.f1, seg.v1)]:
        assert math.isclose(v, d.TWO_PI * f * 1.0, rel_tol=1e-9)


def test_acceleration_segment_roundtrip_imperial():
    # a = 1 g = 386.089 in/s² → v = a / (2π f)
    g = 386.089
    seg = d.acceleration_segment(1.0, (1, 100), (0.1, 1000), g_value=g)
    assert seg is not None
    for f, v in [(seg.f0, seg.v0), (seg.f1, seg.v1)]:
        assert math.isclose(v, g / (d.TWO_PI * f), rel_tol=1e-9)


def test_value_ranges_bracket_visible_lines():
    xlim, ylim = (1, 100), (0.1, 1000)
    d_lo, d_hi = d.displacement_value_range(xlim, ylim)
    a_lo, a_hi = d.acceleration_value_range(xlim, ylim)
    assert d_lo < d_hi
    assert a_lo < a_hi
    # picked values should be within [lo, hi]
    for v in d.pick_displacement_values(xlim, ylim):
        assert d_lo <= v <= d_hi
    for v in d.pick_acceleration_values(xlim, ylim, g_value=386.089):
        assert a_lo / 386.089 <= v <= a_hi / 386.089
