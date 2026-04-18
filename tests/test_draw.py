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
