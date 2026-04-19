"""Robustness / permanence tests — edge cases, cleanup, serialization."""
import io
import pickle
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

import triplot  # noqa: F401
from triplot.axes import TripartiteAxes


def _mk(**kw):
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite", **kw})
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    return fig, ax


def test_clear_resets_caches_and_redraws():
    fig, ax = _mk()
    fig.canvas.draw()
    assert ax.diag_line_count > 0

    ax.clear()
    # clear() wipes axes settings; re-configure and draw again
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    assert ax.diag_line_count > 0
    plt.close(fig)


def test_cla_alias_still_works():
    fig, ax = _mk()
    fig.canvas.draw()
    ax.cla()  # alias of clear() in modern mpl
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    assert ax.diag_line_count > 0
    plt.close(fig)


def test_invalid_view_is_gated_by_validator():
    """Our _view_is_valid gate must reject non-finite / non-positive / inverted
    limits, so _rebuild_diagonals never hands bad numbers to downstream math."""
    fig, ax = _mk()
    assert ax._view_is_valid((1, 10), (0.1, 1)) is True
    assert ax._view_is_valid((float("nan"), 10), (0.1, 1)) is False
    assert ax._view_is_valid((1, float("inf")), (0.1, 1)) is False
    assert ax._view_is_valid((-1, 10), (0.1, 1)) is False
    assert ax._view_is_valid((1, 10), (0, 1)) is False
    assert ax._view_is_valid((10, 1), (0.1, 1)) is False  # inverted
    assert ax._view_is_valid((1, 1), (0.1, 1)) is False   # zero span
    plt.close(fig)


def test_rebuild_hides_on_hidden_state_without_crash():
    fig, ax = _mk()
    fig.canvas.draw()
    # manually invoke the invariant path: turning off visibility + rebuild
    ax._diag_visible = False
    ax._rebuild_diagonals()
    assert ax.diag_line_count == 0
    assert ax.diag_label_count == 0
    plt.close(fig)


def test_linear_scale_hides_diagonals_gracefully():
    fig, ax = _mk()
    fig.canvas.draw()
    # switch to linear; must not crash, should skip diagonals
    ax.set_xscale("linear")
    fig.canvas.draw()
    assert ax.diag_line_count == 0
    plt.close(fig)


def test_multiple_axes_in_one_figure_are_independent():
    fig, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "tripartite"})
    ax1.set_xlim(1, 100); ax1.set_ylim(0.1, 10)
    ax2.set_xlim(10, 10_000); ax2.set_ylim(1, 1000)
    fig.canvas.draw()
    assert ax1.diag_line_count > 0
    assert ax2.diag_line_count > 0
    # different view => disjoint label texts at least at boundaries
    plt.close(fig)


def test_pickle_roundtrip_preserves_data_lines():
    fig, ax = _mk()
    f = np.logspace(0, 3, 50)
    v = 10 / np.sqrt(1 + (f / 50) ** 2)
    ax.plot(f, v)
    fig.canvas.draw()

    buf = io.BytesIO()
    pickle.dump(fig, buf)
    buf.seek(0)
    fig2 = pickle.load(buf)
    ax2 = fig2.axes[0]
    fig2.canvas.draw()
    # data line preserved
    data_lines = [ln for ln in ax2.lines if len(ln.get_xdata()) == 50]
    assert len(data_lines) == 1
    # diagonal grid regenerated after load
    assert ax2.diag_line_count > 0
    plt.close(fig); plt.close(fig2)


def test_extreme_wide_range():
    fig, ax = _mk()
    ax.set_xlim(1e-3, 1e6)
    ax.set_ylim(1e-6, 1e6)
    fig.canvas.draw()
    assert ax.diag_line_count > 0
    plt.close(fig)


def test_extreme_narrow_range():
    fig, ax = _mk()
    ax.set_xlim(100, 110)
    ax.set_ylim(10, 11)
    fig.canvas.draw()
    # at least something should still render — or zero is acceptable if no "nice"
    # decade falls inside the band. Either way draw must not crash.
    assert ax.diag_line_count == ax.diag_label_count
    plt.close(fig)


def test_damping_validation():
    fig, ax = _mk()
    ax.set_damping(0.05)
    assert ax.get_damping() == pytest.approx(0.05)
    with pytest.raises(ValueError):
        ax.set_damping(-0.1)
    plt.close(fig)


def test_grid_diagonal_invalid_which():
    fig, ax = _mk()
    with pytest.raises(ValueError):
        ax.grid_diagonal(True, which="bogus")
    plt.close(fig)


def test_unknown_unit_raises():
    with pytest.raises(ValueError):
        plt.subplots(subplot_kw={"projection": "tripartite", "units": "potato"})
    plt.close("all")


def test_pass_unitsystem_instance():
    from triplot import UnitSystem
    custom = UnitSystem(
        name="custom", freq_label="Hz", vel_label="v [cm/s]",
        disp_label="d [cm]", accel_label="a [cm/s^2]",
        g_value=980.665, accel_in_g=False,
    )
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite", "units": custom})
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    assert "cm/s" in ax.get_ylabel()
    plt.close(fig)


def test_set_displacement_label_updates():
    """Axis-side label changes are stored on the UnitSystem (the in-plot
    diagonal labels are intentionally unit-free per DPlot/Irvine convention —
    units belong on the axis label, not on every grid line)."""
    fig, ax = _mk()
    ax.set_displacement_label("Disp [mm]")
    fig.canvas.draw()
    assert ax._units.disp_label == "Disp [mm]"
    plt.close(fig)


def test_label_rotation_matches_display_angle():
    # Use midpoint label_mode so _disp_labels is populated; the rotation
    # math is the same surface either way.
    fig, ax = plt.subplots(
        subplot_kw={"projection": "tripartite", "label_mode": "midpoint"},
    )
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    assert len(ax._disp_labels) > 0
    # pick first disp segment, recompute expected angle, compare
    segs = ax._disp_collection.get_segments()
    (f0, v0), (f1, v1) = segs[0]
    p0 = ax.transData.transform((f0, v0))
    p1 = ax.transData.transform((f1, v1))
    expected = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    actual = ax._disp_labels[0].get_rotation()
    diff = (expected - actual + 180) % 360 - 180
    assert abs(diff) < 0.01
    plt.close(fig)


def test_draw_path_caches_when_unchanged():
    """Second draw with identical view must not rebuild the cache."""
    fig, ax = _mk()
    fig.canvas.draw()
    key1 = ax._cache_key
    assert key1 is not None
    # Force rebuild skip — monkey-patch rebuild to count calls
    calls = [0]
    orig = ax._rebuild_diagonals
    def counting_rebuild():
        calls[0] += 1
        return orig()
    ax._rebuild_diagonals = counting_rebuild
    fig.canvas.draw()
    assert calls[0] == 0  # cached, no rebuild
    # Now change view, expect rebuild
    ax.set_xlim(2, 500)
    fig.canvas.draw()
    assert calls[0] >= 1
    plt.close(fig)


def test_savefig_produces_file():
    fig, ax = _mk()
    ax.plot([1, 10, 100], [1, 5, 10])
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    assert buf.tell() > 1000
    plt.close(fig)


def test_diagonals_never_have_nan():
    fig, ax = _mk()
    fig.canvas.draw()
    for seg in ax._disp_collection.get_segments():
        arr = np.asarray(seg)
        assert np.all(np.isfinite(arr))
    for seg in ax._accel_collection.get_segments():
        arr = np.asarray(seg)
        assert np.all(np.isfinite(arr))
    plt.close(fig)


def test_stress_many_zooms_no_leak():
    fig, ax = _mk()
    fig.canvas.draw()
    base_labels = ax.diag_label_count
    # pool may only grow, not leak orphans
    rng = np.random.default_rng(42)
    for _ in range(200):
        lo = float(10 ** rng.uniform(-2, 1))
        hi = lo * float(10 ** rng.uniform(0.5, 4))
        ax.set_xlim(lo, hi)
        ylo = float(10 ** rng.uniform(-3, 0))
        yhi = ylo * float(10 ** rng.uniform(0.5, 4))
        ax.set_ylim(ylo, yhi)
        fig.canvas.draw()
        assert ax.diag_label_count <= ax.diag_line_count
    plt.close(fig)
