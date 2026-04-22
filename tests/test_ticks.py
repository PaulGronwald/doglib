"""Progressive tick picker — guarantees ≥2 values in any viewport, including
extreme narrow zooms and aspect-squished views. Before this was wired in,
the legacy ``_nice_decades`` would silently return an empty list any time
the viewport didn't straddle a {1,2,5}x10^k value, and all diagonal lines
would vanish — "rescaling stops at a specific level".
"""
import math

import pytest

from triplot import ticks, diagonals


def test_nice_values_decade_span_uses_coarse_ladder():
    """A comfortable 1-decade range should stick to the preferred {1,2,5}
    ladder — we don't want finer ladders kicking in and cluttering broad views."""
    vals = ticks.nice_values(1.0, 10.0, min_count=2, preferred=(1.0, 2.0, 5.0))
    assert 1.0 in vals and 2.0 in vals and 5.0 in vals and 10.0 in vals


def test_nice_values_narrow_zoom_progresses_ladder():
    """Narrow zoom inside a single decade: preferred ladder can't hit
    anything, so the picker must advance until min_count is met."""
    vals = ticks.nice_values(1.1, 1.4, min_count=2)
    assert len(vals) >= 2
    for v in vals:
        assert 1.1 <= v <= 1.4


def test_nice_values_ultra_narrow_sub_decade_fallback():
    """Viewport that doesn't touch any mantissa ladder at all — picker
    falls through to linear nice stepping."""
    vals = ticks.nice_values(1.23, 1.25, min_count=2)
    assert len(vals) >= 2
    for v in vals:
        assert 1.23 <= v <= 1.25


def test_nice_values_rejects_invalid():
    assert ticks.nice_values(0, 10) == []
    assert ticks.nice_values(-1, 10) == []
    assert ticks.nice_values(10, 1) == []


def test_pick_displacement_values_narrow_zoom_has_min_two():
    """The bug this test prevents: a narrow viewport where no
    constant-d line at mantissa {1,2,5}x10^k falls inside the range,
    causing the legacy picker to return zero lines."""
    vals = diagonals.pick_displacement_values(
        (100, 110), (10, 11), min_count=2,
    )
    assert len(vals) >= 2


def test_pick_acceleration_values_narrow_zoom_has_min_two():
    vals = diagonals.pick_acceleration_values(
        (100, 110), (10, 11), g_value=386.089, min_count=2,
    )
    assert len(vals) >= 2


def test_pick_with_aspect_squish_still_returns_lines():
    """Squished viewport — x-range tight, y-range wide — a realistic
    scenario when the user drags the figure very wide and narrow. Both
    axes must still produce gridlines."""
    xlim = (9.0, 11.0)
    ylim = (0.1, 1000.0)
    disp = diagonals.pick_displacement_values(xlim, ylim, min_count=2)
    accel = diagonals.pick_acceleration_values(xlim, ylim, g_value=386.089, min_count=2)
    assert len(disp) >= 2
    assert len(accel) >= 2


def test_overflow_pad_widens_range_symmetrically():
    lo, hi = ticks.overflow_pad(1.0, 10.0)
    assert lo < 1.0 and hi > 10.0
    assert math.isclose(
        math.log10(1.0) - math.log10(lo),
        math.log10(hi) - math.log10(10.0),
        rel_tol=1e-9,
    )


def test_overflow_pad_noop_for_invalid():
    # Unchanged when the caller hands in garbage — picker below will
    # reject it and return [].
    assert ticks.overflow_pad(-1, 10) == (-1, 10)


def test_decade_step_samples_every_nth_decade():
    """decade_step=5 emits 1e0, 1e5, 1e10 but not 1e1..1e4."""
    vals = ticks.nice_values(1.0, 1e10, preferred=(1.0,), decade_step=5)
    assert 1.0 in vals
    assert 1e5 in vals
    assert 1e10 in vals
    assert 10.0 not in vals
    assert 100.0 not in vals


def test_decade_step_anchored_to_absolute_decade():
    """Stepping is anchored at k=0 so the tick set is pan-stable: two
    overlapping viewports should emit the same ticks where they overlap,
    not shift ticks around as the window pans."""
    vals_a = ticks.nice_values(1.0, 1e10, preferred=(1.0,), decade_step=5)
    vals_b = ticks.nice_values(1e3, 1e13, preferred=(1.0,), decade_step=5)
    # Both ranges cover 1e5 and 1e10
    assert 1e5 in vals_a and 1e5 in vals_b
    assert 1e10 in vals_a and 1e10 in vals_b


def test_gridline_count_bounded_at_extreme_zoom():
    """The core must cap line count via decade_step once span > ~20 dec,
    so a 100-decade zoom doesn't render 100 gridlines per family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import triplot  # noqa: F401

    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1e-50, 1e50)
    ax.set_ylim(1e-50, 1e50)
    fig.canvas.draw()
    # 100-decade span hitting decade_step=5 -> ~20 ticks per family max
    for coll in (ax._disp_collection, ax._accel_collection):
        n = len(coll.get_segments())
        assert n <= 30, f"expected <=30 lines at 100-decade zoom, got {n}"
        assert n >= 2, f"expected >=2 lines at 100-decade zoom, got {n}"
    plt.close(fig)
