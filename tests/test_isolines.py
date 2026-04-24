import matplotlib
matplotlib.use("Agg")

import math

import matplotlib.pyplot as plt

import triplot  # noqa: F401
from triplot import isolines as iso_mod


def _make_axes():
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1.0, 1000.0)
    ax.set_ylim(0.1, 100.0)
    return fig, ax


def test_velocity_crossing_uses_active_left_spine_coordinate():
    p = iso_mod._crossing_to_data(
        "vel", 10.0, "left", 10.0, 386.089, (10.0, 1000.0),
    )
    assert p == (10.0, 10.0)


def test_velocity_crossing_uses_active_right_spine_coordinate():
    p = iso_mod._crossing_to_data(
        "vel", 10.0, "right", 10.0, 386.089, (10.0, 1000.0),
    )
    assert p == (1000.0, 10.0)


def test_isoline_tick_segment_defaults_to_hidden_but_label_still_shows():
    fig, ax = _make_axes()
    line = ax.add_isoline("vel", 10.0, label="v=10")
    fig.canvas.draw()

    assert line.tick.get_visible() is False
    assert line.label is not None
    assert line.label.get_visible() is True

    plt.close(fig)


def test_velocity_tick_segment_attaches_to_left_edge_when_enabled():
    fig, ax = _make_axes()
    line = ax.add_isoline("vel", 10.0, draw_tick_segment=True)
    fig.canvas.draw()

    assert line.tick.get_visible() is True

    expected = ax.transData.transform((ax.get_xlim()[0], 10.0))
    x0 = float(line.tick.get_xdata()[0])
    y0 = float(line.tick.get_ydata()[0])
    assert math.isclose(x0, float(expected[0]), rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(y0, float(expected[1]), rel_tol=0.0, abs_tol=1e-6)

    ax.set_xlim(10.0, 1000.0)
    fig.canvas.draw()

    expected2 = ax.transData.transform((ax.get_xlim()[0], 10.0))
    x1 = float(line.tick.get_xdata()[0])
    y1 = float(line.tick.get_ydata()[0])
    assert math.isclose(x1, float(expected2[0]), rel_tol=0.0, abs_tol=1e-6)
    assert math.isclose(y1, float(expected2[1]), rel_tol=0.0, abs_tol=1e-6)

    plt.close(fig)
