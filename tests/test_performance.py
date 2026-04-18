"""Performance guardrails — these are loose upper bounds chosen to catch
regressions, not tuned for a specific CPU. If they fail, *something* slowed
down by an order of magnitude."""
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

import triplot  # noqa: F401


def _mk():
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    return fig, ax


def test_first_draw_under_500ms():
    fig, ax = _mk()
    t0 = time.perf_counter()
    fig.canvas.draw()
    dt = time.perf_counter() - t0
    assert dt < 0.5, f"first draw took {dt*1000:.1f} ms"
    plt.close(fig)


def test_cached_draw_much_faster_than_first():
    fig, ax = _mk()
    fig.canvas.draw()  # warm
    # unchanged view — subsequent draws should hit cache
    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        fig.canvas.draw()
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    # cached path should be well under 100 ms even on slow machines
    assert median < 0.2, f"cached draw median={median*1000:.1f} ms"
    plt.close(fig)


def test_zoom_draw_under_200ms_median():
    fig, ax = _mk()
    fig.canvas.draw()
    times = []
    for xhi in [100, 500, 2000, 10, 5000, 50, 500, 100]:
        ax.set_xlim(1, xhi)
        t0 = time.perf_counter()
        fig.canvas.draw()
        times.append(time.perf_counter() - t0)
    median = sorted(times)[len(times) // 2]
    assert median < 0.2, f"zoom draw median={median*1000:.1f} ms"
    plt.close(fig)


def test_no_rebuild_when_view_unchanged():
    """Cache hit: signature match => _rebuild_diagonals not invoked."""
    fig, ax = _mk()
    fig.canvas.draw()

    calls = [0]
    orig = ax._rebuild_diagonals
    def counting():
        calls[0] += 1
        return orig()
    ax._rebuild_diagonals = counting

    for _ in range(5):
        fig.canvas.draw()
    assert calls[0] == 0
    plt.close(fig)


def test_rebuild_invoked_exactly_once_per_view_change():
    fig, ax = _mk()
    fig.canvas.draw()
    calls = [0]
    orig = ax._rebuild_diagonals
    def counting():
        calls[0] += 1
        return orig()
    ax._rebuild_diagonals = counting

    for xhi in (500, 200, 2000):
        ax.set_xlim(1, xhi)
        fig.canvas.draw()
        fig.canvas.draw()  # same view — should NOT rebuild a second time

    assert calls[0] == 3, f"expected 3 rebuilds, got {calls[0]}"
    plt.close(fig)
