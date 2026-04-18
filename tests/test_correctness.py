"""Math-correctness verification against hand-computed reference values.

Checks the tripartite relations `v = 2*pi*f*d` and `a = 2*pi*f*v` against
known examples from Irvine's tripartite tutorial and textbook SRS problems.
"""
import math

import numpy as np
import pytest

from triplot import diagonals as d


TWO_PI = 2 * math.pi


def test_reference_point_100hz_1g_imperial():
    """Classic SRS teaching example:
    1 g acceleration at 100 Hz => pseudo-velocity = g / (2*pi*f).
    With g = 386.089 in/s^2: v = 386.089 / (2*pi*100) ~= 0.6144 in/s.
    """
    g_imp = 386.089
    v_expected = g_imp / (TWO_PI * 100.0)
    # Build a constant-accel segment for a = 1 g and check f=100 falls on it.
    seg = d.acceleration_segment(1.0, (1, 1000), (0.01, 100), g_value=g_imp)
    assert seg is not None
    # Point at f=100 on the line: v = a / (2*pi*f) = g / (2*pi*100)
    # We verify by interpolating along the log-log segment.
    f0, f1 = seg.f0, seg.f1
    v0, v1 = seg.v0, seg.v1
    t = (math.log10(100) - math.log10(f0)) / (math.log10(f1) - math.log10(f0))
    v_on_line = 10 ** (math.log10(v0) + t * (math.log10(v1) - math.log10(v0)))
    assert math.isclose(v_on_line, v_expected, rel_tol=1e-9), \
        f"expected {v_expected}, got {v_on_line}"


def test_reference_point_10hz_1in_imperial():
    """Constant-displacement 1 inch at 10 Hz => v = 2*pi*10*1 = 62.83 in/s."""
    v_expected = TWO_PI * 10.0 * 1.0
    seg = d.displacement_segment(1.0, (1, 1000), (0.1, 1000))
    assert seg is not None
    f0, f1 = seg.f0, seg.f1
    v0, v1 = seg.v0, seg.v1
    t = (math.log10(10) - math.log10(f0)) / (math.log10(f1) - math.log10(f0))
    v_on_line = 10 ** (math.log10(v0) + t * (math.log10(v1) - math.log10(v0)))
    assert math.isclose(v_on_line, v_expected, rel_tol=1e-9)


def test_reference_point_si_1ms2_at_1hz():
    """1 m/s^2 at 1 Hz => v = a / (2*pi) = 0.15915 m/s (SI, g_value=1)."""
    v_expected = 1.0 / TWO_PI
    seg = d.acceleration_segment(1.0, (0.1, 100), (0.001, 10), g_value=1.0)
    assert seg is not None
    f0, f1 = seg.f0, seg.f1
    v0, v1 = seg.v0, seg.v1
    t = (math.log10(1.0) - math.log10(f0)) / (math.log10(f1) - math.log10(f0))
    v_on_line = 10 ** (math.log10(v0) + t * (math.log10(v1) - math.log10(v0)))
    assert math.isclose(v_on_line, v_expected, rel_tol=1e-9)


def test_displacement_line_has_slope_plus_one_in_loglog():
    seg = d.displacement_segment(0.5, (1, 1000), (0.1, 100))
    assert seg is not None
    dy = math.log10(seg.v1) - math.log10(seg.v0)
    dx = math.log10(seg.f1) - math.log10(seg.f0)
    slope = dy / dx
    assert math.isclose(slope, 1.0, rel_tol=1e-9), f"got slope {slope}"


def test_acceleration_line_has_slope_minus_one_in_loglog():
    seg = d.acceleration_segment(1.0, (1, 1000), (0.01, 100), g_value=386.089)
    assert seg is not None
    dy = math.log10(seg.v1) - math.log10(seg.v0)
    dx = math.log10(seg.f1) - math.log10(seg.f0)
    slope = dy / dx
    assert math.isclose(slope, -1.0, rel_tol=1e-9), f"got slope {slope}"


def test_omega_is_2pi_times_freq():
    """Verify the library uses omega = 2*pi*f, not omega = f."""
    assert math.isclose(d.TWO_PI, 2 * math.pi, abs_tol=1e-15)
    # a = omega^2 * d => at f=1, a=1g, d should equal g / (2*pi)^2
    g = 386.089
    d_expected = g / (TWO_PI ** 2)
    # Find a point where the 1g accel line and 1Hz cross
    seg = d.acceleration_segment(1.0, (1, 10), (0.001, 100), g_value=g)
    assert seg is not None
    f0, f1 = seg.f0, seg.f1
    v0, v1 = seg.v0, seg.v1
    t = (math.log10(1.0) - math.log10(f0)) / (math.log10(f1) - math.log10(f0))
    v_at_1hz = 10 ** (math.log10(v0) + t * (math.log10(v1) - math.log10(v0)))
    d_at_1hz = v_at_1hz / (TWO_PI * 1.0)
    assert math.isclose(d_at_1hz, d_expected, rel_tol=1e-9)


def test_crossings_between_families_are_consistent():
    """At every (f, v) on a constant-d line, the corresponding a satisfies
    a = 2*pi*f*v and equals (2*pi*f)^2 * d. Test that both accel- and disp-
    lines through the same point agree."""
    xlim, ylim = (1, 1000), (0.01, 1000)
    d_val = 0.1
    seg_d = d.displacement_segment(d_val, xlim, ylim)
    assert seg_d is not None
    # midpoint
    log_f = 0.5 * (math.log10(seg_d.f0) + math.log10(seg_d.f1))
    log_v = 0.5 * (math.log10(seg_d.v0) + math.log10(seg_d.v1))
    f = 10 ** log_f
    v = 10 ** log_v
    a_implied = TWO_PI * f * v  # in raw accel units (in/s^2 for imperial)
    # Verify: this same point satisfies a = (2*pi*f)^2 * d
    a_from_d = (TWO_PI * f) ** 2 * d_val
    assert math.isclose(a_implied, a_from_d, rel_tol=1e-9)
