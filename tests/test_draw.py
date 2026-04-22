import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

import triplot  # noqa: F401


def _make():
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 100)
    return fig, ax


def test_draw_produces_diagonal_artists():
    fig, ax = _make()
    fig.canvas.draw()
    assert ax.diag_line_count > 0
    assert ax.diag_label_count > 0
    # Seismic style: many more lines than labels (e.g. 9 lines, 5 labels / decade)
    assert ax.diag_label_count <= ax.diag_line_count
    plt.close(fig)


def test_zoom_rebuilds_without_leak():
    fig, ax = _make()
    fig.canvas.draw()
    baseline_lines = ax.diag_line_count
    assert baseline_lines > 0

    for xhi in (500, 200, 2000, 10_000, 50):
        ax.set_xlim(1, xhi)
        fig.canvas.draw()

    # No leak: labels always a subset of lines, counts bounded
    assert ax.diag_label_count <= ax.diag_line_count
    assert ax.diag_line_count <= baseline_lines * 4
    plt.close(fig)


def test_grid_diagonal_toggle():
    fig, ax = _make()
    ax.grid_diagonal(False)
    fig.canvas.draw()
    assert ax.diag_line_count == 0
    assert ax.diag_label_count == 0

    ax.grid_diagonal(True)
    fig.canvas.draw()
    assert ax.diag_line_count > 0
    plt.close(fig)


def test_plot_data_works():
    fig, ax = _make()
    f = np.logspace(0, 3, 50)
    v = 10 / np.sqrt(1 + (f / 50) ** 2)
    line, = ax.plot(f, v)
    fig.canvas.draw()
    assert line in ax.lines
    plt.close(fig)


def test_si_units():
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite", "units": "SI"})
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.01, 10)
    fig.canvas.draw()
    assert "m/s" in ax.get_ylabel()
    plt.close(fig)


def test_fallback_labels_fill_edge_gap():
    """Viewport chosen so disp majors (d=0.001, 0.01) don't cross the top
    edge but do cross the right — without the fallback, such a line goes
    unlabeled. Only the *major* diagonals are labeled now (minors are
    unlabeled grid decoration), so we exercise disp at this zoom and
    leave accel geometry out of the assertion (accel majors at 10g /
    100g both cross right cleanly)."""
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(100, 300)
    ax.set_ylim(1, 30)
    fig.canvas.draw()

    disp_fb_visible = [a for a in ax._disp_fallback_labels if a.get_visible()]
    assert len(disp_fb_visible) > 0, "expected disp fallback labels in this viewport"
    for ann in disp_fb_visible:
        assert ann.get_text()
        assert "in" in ann.get_text()

    plt.close(fig)


def test_label_mode_edge_has_no_midpoint_labels():
    """label_mode='edge' (default) — rotated edge labels + fallback, no midpoints."""
    fig, ax = plt.subplots(
        subplot_kw={"projection": "tripartite", "label_mode": "edge"},
    )
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    assert len(ax._disp_labels) == 0
    assert len(ax._accel_labels) == 0
    # Edge labels still present
    assert sum(1 for a in ax._disp_top_labels if a.get_visible()) > 0
    plt.close(fig)


def test_label_mode_midpoint_has_no_edge_labels():
    """label_mode='midpoint' — labels only at segment midpoints, no edges."""
    fig, ax = plt.subplots(
        subplot_kw={"projection": "tripartite", "label_mode": "midpoint"},
    )
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    assert len(ax._disp_top_labels) == 0
    assert len(ax._accel_right_labels) == 0
    assert len(ax._disp_fallback_labels) == 0
    assert len(ax._accel_fallback_labels) == 0
    assert sum(1 for a in ax._disp_labels if a.get_visible()) > 0
    plt.close(fig)


def test_label_mode_is_binary():
    """Only 'edge' and 'midpoint' are valid — no 'both' / 'full' mode."""
    with pytest.raises(ValueError):
        plt.subplots(subplot_kw={"projection": "tripartite", "label_mode": "full"})
    with pytest.raises(ValueError):
        plt.subplots(subplot_kw={"projection": "tripartite", "label_mode": "both"})


def test_set_label_mode_switches_at_runtime():
    """Toggling label_mode after construction rebuilds the plot."""
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    # Default is 'edge' — midpoints suppressed
    assert len(ax._disp_labels) == 0

    ax.set_label_mode("midpoint")
    fig.canvas.draw()
    assert len(ax._disp_labels) > 0
    assert len(ax._disp_top_labels) == 0  # edges now suppressed

    ax.set_label_mode("edge")
    fig.canvas.draw()
    assert len(ax._disp_labels) == 0  # back to edge-only
    plt.close(fig)


def test_show_diag_titles_false_hides_callouts():
    """show_diag_titles=False removes the big centered 'Displacement' /
    'Acceleration' titles even for the seismic style (which shows them by
    default)."""
    fig, ax = plt.subplots(
        subplot_kw={"projection": "tripartite", "show_diag_titles": False},
    )
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    assert ax._disp_axis_title is None
    assert ax._accel_axis_title is None
    plt.close(fig)


def test_border_labels_are_rotated():
    """Border (edge) labels rotate to match their line's slope, not horizontal."""
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    rotations = [
        abs(a.get_rotation()) for a in ax._disp_top_labels if a.get_visible()
    ]
    assert len(rotations) > 0
    # Every rotated-on-line label should have a non-trivial rotation
    # (0° would mean horizontal — the bug this test catches).
    assert all(r > 10 for r in rotations)
    plt.close(fig)


def test_fallback_pool_does_not_leak():
    """Pan / zoom repeatedly — the fallback label pool must shrink when the
    view no longer needs as many, so stale artists don't accumulate."""
    fig, ax = _make()
    fig.canvas.draw()
    max_seen = len(ax._disp_fallback_labels) + len(ax._accel_fallback_labels)
    for xhi in (500, 2000, 100, 10_000):
        ax.set_xlim(1, xhi)
        fig.canvas.draw()
        max_seen = max(
            max_seen,
            len(ax._disp_fallback_labels) + len(ax._accel_fallback_labels),
        )
    # After a narrow re-zoom, counts should not exceed what was needed before.
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    final = len(ax._disp_fallback_labels) + len(ax._accel_fallback_labels)
    # Pools can be empty in some views; the key check is that they can shrink.
    assert final <= max_seen
    plt.close(fig)
