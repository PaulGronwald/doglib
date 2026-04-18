"""Property-based tests for diagonals math using hypothesis if available;
falls back to broad random sampling otherwise."""
from __future__ import annotations

import math
import random

import numpy as np
import pytest

from triplot import diagonals as d

try:
    from hypothesis import given, settings, strategies as st
    HAS_HYPOTHESIS = True
except Exception:
    HAS_HYPOTHESIS = False


# ---------------------------------------------------------------- helpers

def _valid_viewport(x0, x1, y0, y1):
    return x0 > 0 and x1 > x0 and y0 > 0 and y1 > y0


# ---------------------------------------------------------------- pure math


def test_displacement_segment_endpoints_always_satisfy_v_eq_2pifd():
    rng = random.Random(0)
    for _ in range(2000):
        x0 = 10 ** rng.uniform(-3, 3)
        x1 = x0 * 10 ** rng.uniform(0.5, 4)
        y0 = 10 ** rng.uniform(-4, 2)
        y1 = y0 * 10 ** rng.uniform(0.5, 4)
        d_val = 10 ** rng.uniform(-5, 5)
        seg = d.displacement_segment(d_val, (x0, x1), (y0, y1))
        if seg is None:
            continue
        for f, v in [(seg.f0, seg.v0), (seg.f1, seg.v1)]:
            assert math.isclose(v, d.TWO_PI * f * d_val, rel_tol=1e-9), \
                f"d={d_val} view=({x0},{x1},{y0},{y1}) pt=({f},{v})"
            # endpoints must lie inside viewport (with tiny numerical tolerance)
            tol = 1e-9
            assert x0 * (1 - tol) <= f <= x1 * (1 + tol)
            assert y0 * (1 - tol) <= v <= y1 * (1 + tol)


def test_acceleration_segment_endpoints_always_satisfy_a_eq_2pifv():
    rng = random.Random(1)
    g = 386.089
    for _ in range(2000):
        x0 = 10 ** rng.uniform(-3, 3)
        x1 = x0 * 10 ** rng.uniform(0.5, 4)
        y0 = 10 ** rng.uniform(-4, 2)
        y1 = y0 * 10 ** rng.uniform(0.5, 4)
        a_label = 10 ** rng.uniform(-5, 5)
        seg = d.acceleration_segment(a_label, (x0, x1), (y0, y1), g_value=g)
        if seg is None:
            continue
        a = a_label * g
        for f, v in [(seg.f0, seg.v0), (seg.f1, seg.v1)]:
            assert math.isclose(v, a / (d.TWO_PI * f), rel_tol=1e-9)
            tol = 1e-9
            assert x0 * (1 - tol) <= f <= x1 * (1 + tol)
            assert y0 * (1 - tol) <= v <= y1 * (1 + tol)


def test_picked_values_are_within_visible_range():
    rng = random.Random(2)
    for _ in range(200):
        x0 = 10 ** rng.uniform(-2, 2)
        x1 = x0 * 10 ** rng.uniform(0.5, 3)
        y0 = 10 ** rng.uniform(-3, 1)
        y1 = y0 * 10 ** rng.uniform(0.5, 3)
        d_lo, d_hi = d.displacement_value_range((x0, x1), (y0, y1))
        for v in d.pick_displacement_values((x0, x1), (y0, y1)):
            assert d_lo <= v <= d_hi
        a_lo, a_hi = d.acceleration_value_range((x0, x1), (y0, y1))
        for v in d.pick_acceleration_values((x0, x1), (y0, y1), g_value=1.0):
            assert a_lo <= v <= a_hi


def test_nice_decades_monotone_and_unique():
    rng = random.Random(3)
    for _ in range(500):
        lo = 10 ** rng.uniform(-5, 5)
        hi = lo * 10 ** rng.uniform(0.1, 6)
        vals = d._nice_decades(lo, hi)
        assert vals == sorted(vals)
        assert len(vals) == len(set(vals))


def test_clip_line_returns_two_distinct_points_or_none():
    rng = random.Random(4)
    for _ in range(1000):
        slope = rng.choice([+1, -1])
        b = rng.uniform(-20, 20)
        xmin, xmax = sorted([rng.uniform(-5, 5), rng.uniform(-5, 5)])
        if xmin == xmax: xmax += 1
        ymin, ymax = sorted([rng.uniform(-5, 5), rng.uniform(-5, 5)])
        if ymin == ymax: ymax += 1
        res = d._clip_slope_line(b, slope, (xmin, xmax), (ymin, ymax))
        if res is None:
            continue
        (x0, y0), (x1, y1) = res
        # endpoints on the line (within float tolerance)
        assert math.isclose(y0, slope * x0 + b, abs_tol=1e-9)
        assert math.isclose(y1, slope * x1 + b, abs_tol=1e-9)
        # both inside viewport
        assert xmin - 1e-9 <= x0 <= xmax + 1e-9
        assert xmin - 1e-9 <= x1 <= xmax + 1e-9
        assert ymin - 1e-9 <= y0 <= ymax + 1e-9
        assert ymin - 1e-9 <= y1 <= ymax + 1e-9
        # ordered
        assert x0 <= x1


def test_format_value_no_crashes():
    for v in [0, 0.001, 0.5, 1, 2, 5, 10, 100, 1e-6, 1e9, 1.2345, 2.718]:
        s = d.format_value(v)
        assert isinstance(s, str) and len(s) > 0


# ---------------------------------------------------------------- hypothesis (optional)

pytestmark_hypo = pytest.mark.skipif(not HAS_HYPOTHESIS, reason="hypothesis not installed")


if HAS_HYPOTHESIS:

    @settings(deadline=None, max_examples=200)
    @given(
        d_val=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False),
        log_x0=st.floats(min_value=-3, max_value=3),
        log_span_x=st.floats(min_value=0.5, max_value=4),
        log_y0=st.floats(min_value=-4, max_value=2),
        log_span_y=st.floats(min_value=0.5, max_value=4),
    )
    def test_hypo_displacement(d_val, log_x0, log_span_x, log_y0, log_span_y):
        x0 = 10 ** log_x0
        x1 = x0 * 10 ** log_span_x
        y0 = 10 ** log_y0
        y1 = y0 * 10 ** log_span_y
        seg = d.displacement_segment(d_val, (x0, x1), (y0, y1))
        if seg is None:
            return
        for f, v in [(seg.f0, seg.v0), (seg.f1, seg.v1)]:
            assert math.isclose(v, d.TWO_PI * f * d_val, rel_tol=1e-8)

    @settings(deadline=None, max_examples=200)
    @given(
        a_val=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False),
        log_x0=st.floats(min_value=-3, max_value=3),
        log_span_x=st.floats(min_value=0.5, max_value=4),
        log_y0=st.floats(min_value=-4, max_value=2),
        log_span_y=st.floats(min_value=0.5, max_value=4),
        g=st.sampled_from([1.0, 9.80665, 386.089]),
    )
    def test_hypo_acceleration(a_val, log_x0, log_span_x, log_y0, log_span_y, g):
        x0 = 10 ** log_x0; x1 = x0 * 10 ** log_span_x
        y0 = 10 ** log_y0; y1 = y0 * 10 ** log_span_y
        seg = d.acceleration_segment(a_val, (x0, x1), (y0, y1), g_value=g)
        if seg is None:
            return
        a = a_val * g
        for f, v in [(seg.f0, seg.v0), (seg.f1, seg.v1)]:
            assert math.isclose(v, a / (d.TWO_PI * f), rel_tol=1e-8)
