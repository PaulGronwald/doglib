"""Math for tripartite diagonal grids. No matplotlib dependency — pure numpy/math
so the decade picker and viewport clipping can be unit-tested in isolation.

Conventions (ω = 2πf):
    v = ω · d  =>  log10(v) = log10(f) + log10(2π·d)   (slope +1 in log-log)
    a = ω · v  =>  log10(v) = -log10(f) + log10(a / 2π) (slope -1 in log-log)

"Intercept" below is the y-value at f = 1 Hz on a log-log plot (i.e. the log10(v)
value when log10(f) = 0). For constant displacement d: intercept = log10(2π·d).
For constant acceleration a: intercept = log10(a / 2π).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class DiagSegment:
    """Line segment in data (freq, vel) space, already clipped to viewport."""
    value: float        # the constant (d or a) the line represents
    f0: float
    v0: float
    f1: float
    v1: float


def _nice_decades(lo: float, hi: float, subdivisions=(1.0, 2.0, 5.0)) -> list[float]:
    """Return "nice" values (1, 2, 5 × 10^k) covering [lo, hi]. lo/hi > 0."""
    if lo <= 0 or hi <= 0 or hi < lo:
        return []
    kmin = math.floor(math.log10(lo))
    kmax = math.ceil(math.log10(hi))
    vals = []
    for k in range(kmin - 1, kmax + 2):
        base = 10.0 ** k
        for s in subdivisions:
            v = s * base
            if lo <= v <= hi:
                vals.append(v)
    return vals


def _clip_slope_line(
    intercept_log: float,
    slope: int,
    xlim_log: tuple[float, float],
    ylim_log: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip y = slope*x + intercept_log to the log-space viewport rectangle.
    slope ∈ {+1, -1}. Returns ((x0, y0), (x1, y1)) in log space, or None if
    the line does not intersect the viewport.
    """
    xmin, xmax = xlim_log
    ymin, ymax = ylim_log
    # Candidate intersections with the 4 viewport edges.
    pts = []
    # Left edge x = xmin
    y = slope * xmin + intercept_log
    if ymin <= y <= ymax:
        pts.append((xmin, y))
    # Right edge x = xmax
    y = slope * xmax + intercept_log
    if ymin <= y <= ymax:
        pts.append((xmax, y))
    # Bottom edge y = ymin  =>  x = (ymin - b) / slope
    x = (ymin - intercept_log) / slope
    if xmin <= x <= xmax:
        pts.append((x, ymin))
    # Top edge y = ymax
    x = (ymax - intercept_log) / slope
    if xmin <= x <= xmax:
        pts.append((x, ymax))
    # De-duplicate (corner hits produce duplicates) and order.
    uniq = []
    for p in pts:
        if not any(math.isclose(p[0], q[0]) and math.isclose(p[1], q[1]) for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    uniq.sort()
    return uniq[0], uniq[-1]


def displacement_value_range(xlim, ylim) -> tuple[float, float]:
    """Min/max displacement values that can be visible in the viewport.
    d = v / (2π f). Extrema at viewport corners."""
    fmin, fmax = xlim
    vmin, vmax = ylim
    # d shrinks with f, grows with v
    d_lo = vmin / (TWO_PI * fmax)
    d_hi = vmax / (TWO_PI * fmin)
    return d_lo, d_hi


def acceleration_value_range(xlim, ylim) -> tuple[float, float]:
    """Min/max acceleration values in raw accel units (vel-unit/s)."""
    fmin, fmax = xlim
    vmin, vmax = ylim
    # a = 2π f v, grows with both
    a_lo = TWO_PI * fmin * vmin
    a_hi = TWO_PI * fmax * vmax
    return a_lo, a_hi


def pick_displacement_values(xlim, ylim, subdivisions=(1.0, 2.0, 5.0)) -> list[float]:
    d_lo, d_hi = displacement_value_range(xlim, ylim)
    return _nice_decades(d_lo, d_hi, subdivisions)


def pick_acceleration_values(xlim, ylim, g_value=1.0, subdivisions=(1.0, 2.0, 5.0)) -> list[float]:
    """Returns values in the label unit (g's if g_value != 1). Internally the
    line math uses raw (accel_in_vel_units_per_sec); we pick nice values in
    label units then convert back when drawing."""
    a_lo, a_hi = acceleration_value_range(xlim, ylim)
    a_lo_label = a_lo / g_value
    a_hi_label = a_hi / g_value
    return _nice_decades(a_lo_label, a_hi_label, subdivisions)


def displacement_segment(d: float, xlim, ylim) -> DiagSegment | None:
    """Clip the constant-d line to the viewport. Returns segment in linear data coords."""
    if d <= 0:
        return None
    intercept = math.log10(TWO_PI * d)
    clip = _clip_slope_line(
        intercept, +1,
        (math.log10(xlim[0]), math.log10(xlim[1])),
        (math.log10(ylim[0]), math.log10(ylim[1])),
    )
    if clip is None:
        return None
    (x0, y0), (x1, y1) = clip
    return DiagSegment(d, 10 ** x0, 10 ** y0, 10 ** x1, 10 ** y1)


def acceleration_segment(a_label: float, xlim, ylim, g_value=1.0) -> DiagSegment | None:
    """a_label is in display units (e.g. g); converted via g_value for math."""
    if a_label <= 0:
        return None
    a = a_label * g_value
    intercept = math.log10(a / TWO_PI)
    clip = _clip_slope_line(
        intercept, -1,
        (math.log10(xlim[0]), math.log10(xlim[1])),
        (math.log10(ylim[0]), math.log10(ylim[1])),
    )
    if clip is None:
        return None
    (x0, y0), (x1, y1) = clip
    return DiagSegment(a_label, 10 ** x0, 10 ** y0, 10 ** x1, 10 ** y1)


def format_value(v: float) -> str:
    """Short label: '0.01', '1', '100', '1e-4' for extremes."""
    if v == 0:
        return "0"
    av = abs(v)
    if av < 1e-3 or av >= 1e4:
        return f"{v:g}"
    if av >= 1:
        return f"{v:g}"
    return f"{v:g}"
