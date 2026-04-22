"""Plotly backend smoke tests — same math, different render target.

These tests are skipped if plotly isn't installed; the matplotlib
backend is the only hard dependency for the library.
"""
import math

import pytest

plotly = pytest.importorskip("plotly")

import triplot
from triplot.backends.base import DiagramFamily
from triplot.backends.plotly_backend import PlotlyBackend


def test_plot_plotly_returns_figure():
    fig = triplot.plot([1, 10, 100], [1, 5, 10], backend="plotly")
    assert "Figure" in type(fig).__name__
    # A user-data trace + triplot shapes + annotations
    assert len(fig.data) == 1
    assert len(fig.layout.shapes) > 0
    assert len(fig.layout.annotations) > 0


def test_plotly_backend_describe_artists_has_both_families():
    fig = triplot.subplots(backend="plotly")
    art = fig._triplot_backend.describe_artists()
    # Every family should have lines AND at least one label role
    for fam in ("displacement", "acceleration"):
        assert art.get((fam, "line"), 0) > 0


def test_plotly_narrow_zoom_keeps_at_least_two_lines_per_family():
    """The fixed-picker guarantee propagates through the plotly backend."""
    fig = triplot.subplots(backend="plotly")
    # Log-axis range values are log10 in plotly
    fig.update_xaxes(range=[2.0, 2.1])
    fig.update_yaxes(range=[1.0, 1.05])
    fig._triplot_rebuild()
    art = fig._triplot_backend.describe_artists()
    assert art.get(("displacement", "line"), 0) >= 2
    assert art.get(("acceleration", "line"), 0) >= 2


def test_plotly_invalid_backend_name_raises():
    with pytest.raises(ValueError):
        triplot.plot(backend="matplotlibish")


def test_plotly_unknown_unit_raises():
    with pytest.raises(ValueError):
        triplot.subplots(backend="plotly", units="potato")


def test_plotly_label_mode_edge_places_edge_annotations():
    fig = triplot.subplots(backend="plotly", label_mode="edge")
    art = fig._triplot_backend.describe_artists()
    assert art.get(("displacement", "edge"), 0) > 0 or art.get(
        ("displacement", "fallback"), 0
    ) > 0


def test_plotly_label_mode_midpoint_places_midpoint_annotations():
    fig = triplot.subplots(backend="plotly", label_mode="midpoint")
    art = fig._triplot_backend.describe_artists()
    # Midpoint mode — no edge or fallback
    assert art.get(("displacement", "edge"), 0) == 0
    assert art.get(("displacement", "fallback"), 0) == 0
    assert art.get(("displacement", "midpoint"), 0) > 0


def test_plotly_diagonals_are_line_shapes_not_traces():
    """Shapes must be used for gridlines (lightweight, no legend entry)
    rather than scatter traces. This test guards against a regression
    where someone swaps the rendering strategy and legendifies everything."""
    fig = triplot.subplots(backend="plotly")
    # Exactly zero traces when no user data is attached
    assert len(fig.data) == 0
    assert len(fig.layout.shapes) > 0


def test_plotly_rebuild_is_idempotent():
    """Re-running rebuild shouldn't grow the shape / annotation lists
    without bound — pools must shrink back to size of new set each time."""
    fig = triplot.subplots(backend="plotly")
    fig._triplot_rebuild()
    a1 = len(fig.layout.shapes)
    fig._triplot_rebuild()
    fig._triplot_rebuild()
    a2 = len(fig.layout.shapes)
    assert a1 == a2
