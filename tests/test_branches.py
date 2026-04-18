"""Small tests to exercise remaining branches."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import triplot  # noqa: F401
from triplot.axes import _unit_from_label, TripartiteAxes


def _mk(**kw):
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite", **kw})
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    return fig, ax


def test_unit_from_label_no_brackets():
    assert _unit_from_label("no brackets here") == ""
    assert _unit_from_label("Vel [m/s]") == "m/s"
    assert _unit_from_label("") == ""


def test_subdivisions_major_minor_both():
    fig, ax = _mk()
    # default 'seismic' style => integer multipliers 1..9
    assert ax._subdivisions() == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    ax.grid_diagonal(True, which="major")
    assert ax._subdivisions() == (1.0,)

    ax.grid_diagonal(True, which="minor")
    assert ax._subdivisions() == (1.0, 2.0, 5.0)

    ax.grid_diagonal(True, which="both")
    for expected in (1.0, 2.0, 3.0, 5.0, 7.0, 9.0):
        assert expected in ax._subdivisions()
    fig.canvas.draw()
    assert ax.diag_line_count > 0
    plt.close(fig)


def test_set_acceleration_label_propagates():
    fig, ax = _mk(units="SI")
    ax.set_acceleration_label("Acc [ft/s2]")
    fig.canvas.draw()
    # Convention: diagonal labels are bare numeric; the unit lives on the
    # stored UnitSystem entry so callers / axis labels can use it.
    assert ax._units.accel_label == "Acc [ft/s2]"
    plt.close(fig)


def test_grid_diagonal_style_override_applies():
    fig, ax = _mk()
    ax.grid_diagonal(True, color="red", linewidth=2.0)
    fig.canvas.draw()
    coll = ax._disp_collection
    # LineCollection stores colors as RGBA array; first row red-ish
    c = coll.get_colors()[0]
    assert c[0] > 0.9 and c[1] < 0.2 and c[2] < 0.2
    lw = coll.get_linewidths()[0]
    assert lw == 2.0
    plt.close(fig)


def test_draw_exception_in_rebuild_emits_warning_and_hides(monkeypatch):
    import pytest
    fig, ax = _mk()
    def broken():
        raise RuntimeError("boom")
    monkeypatch.setattr(ax, "_rebuild_diagonals", broken)
    with pytest.warns(UserWarning, match="triplot"):
        fig.canvas.draw()
    plt.close(fig)


def test_getstate_setstate_manual():
    fig, ax = _mk()
    fig.canvas.draw()
    state = ax.__getstate__()
    assert state["_disp_collection"] is None
    assert state["_disp_labels"] == []
    assert state["_cache_key"] is None


def test_cache_signature_includes_bbox():
    fig, ax = _mk()
    fig.canvas.draw()
    sig1 = ax._cache_signature()
    # changing figure size should change bbox -> signature -> trigger rebuild
    fig.set_size_inches(10, 8)
    fig.canvas.draw()
    sig2 = ax._cache_signature()
    assert sig1 != sig2
    plt.close(fig)


def test_diagonals_never_exceed_viewport():
    fig, ax = _mk()
    xlim = (2.0, 500.0)
    ylim = (0.5, 50.0)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    fig.canvas.draw()
    for seg in ax._disp_collection.get_segments() + ax._accel_collection.get_segments():
        xs = [p[0] for p in seg]
        ys = [p[1] for p in seg]
        tol = 1e-6
        assert min(xs) >= xlim[0] * (1 - tol)
        assert max(xs) <= xlim[1] * (1 + tol)
        assert min(ys) >= ylim[0] * (1 - tol)
        assert max(ys) <= ylim[1] * (1 + tol)
    plt.close(fig)
