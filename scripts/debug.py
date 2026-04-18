"""Text-only smoke + benchmark harness for triplot.

Run:  python scripts/debug.py
Exit code 0 = all checks pass.

Stages:
    1. import / registration
    2. axes instantiation
    3. pure-math primitives
    4. first draw produces artists
    5. zoom cycles without leak
    6. grid toggle
    7. user data overlay
    8. SI units
    9. PNG render headless
   10. cache hit rate (permanence of caching layer)
   11. speed benchmarks (first / cached / zoom)
   12. stress: 500 random zooms, no leak
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_PASS = 0
_FAIL = 0


def check(label, cond, detail=""):
    global _PASS, _FAIL
    mark = "OK  " if cond else "FAIL"
    if cond:
        _PASS += 1
    else:
        _FAIL += 1
    msg = f"  [{mark}] {label}"
    if detail:
        msg += f"  -- {detail}"
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "replace").decode("ascii"))


def section(title):
    print(f"\n== {title} ==")


def bench(label, fn, n=10):
    """Return median seconds over n runs."""
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2], times[0], times[-1]


def main() -> int:
    print("triplot debug harness")
    print("=" * 60)

    # 1. Import + registration
    section("1. import + projection registration")
    try:
        import triplot
        check("import triplot", True, f"version {triplot.__version__}")
    except Exception as exc:
        check("import triplot", False, repr(exc))
        return 1

    names = matplotlib.projections.get_projection_names()
    check("'tripartite' in projection registry", "tripartite" in names,
          f"registry has {len(names)} projections")

    # 2. Axes instantiation
    section("2. axes instantiation")
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    check("class is TripartiteAxes", type(ax).__name__ == "TripartiteAxes")
    check("x scale is log", ax.get_xscale() == "log")
    check("y scale is log", ax.get_yscale() == "log")
    check("xlabel contains 'Frequency'", "Frequency" in ax.get_xlabel())
    check("ylabel contains 'Velocity'", "Velocity" in ax.get_ylabel())

    # 3. Diagonal math
    section("3. diagonal math (pure python)")
    from triplot import diagonals as d

    vals = d._nice_decades(0.5, 50.0)
    check("nice_decades(0.5, 50) non-empty", len(vals) > 0, f"{len(vals)} values")
    check("nice_decades within [lo, hi]",
          all(0.5 <= v <= 50 for v in vals))

    clip = d._clip_slope_line(0.0, +1, (0.0, 2.0), (0.0, 2.0))
    check("clip +45deg works", clip is not None)

    seg = d.displacement_segment(1.0, (1, 100), (0.1, 1000))
    check("displacement_segment works", seg is not None)
    if seg is not None:
        check("v = 2*pi*f*d holds at both endpoints",
              math.isclose(seg.v0, d.TWO_PI * seg.f0, rel_tol=1e-9)
              and math.isclose(seg.v1, d.TWO_PI * seg.f1, rel_tol=1e-9))

    # 4. Draw produces artists
    section("4. draw produces diagonal artists")
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    n_lines = ax.diag_line_count
    n_labels = ax.diag_label_count
    check("diagonal lines > 0", n_lines > 0, f"{n_lines} lines")
    check("diagonal labels > 0", n_labels > 0, f"{n_labels} labels")
    check("one label per line", n_lines == n_labels)

    # 5. Zoom cycles no leak
    section("5. zoom cycles (leak check)")
    counts = []
    for xhi in (500, 200, 5000, 50_000, 10, 1000):
        ax.set_xlim(1, xhi)
        fig.canvas.draw()
        counts.append((ax.diag_line_count, ax.diag_label_count))
    print(f"  counts: {counts}")
    check("lines == labels every zoom", all(l == t for l, t in counts))
    check("bounded artist counts", max(l for l, _ in counts) <= n_lines * 4)

    # 6. Toggle
    section("6. grid toggle")
    ax.grid_diagonal(False)
    fig.canvas.draw()
    check("off -> zero diagonals", ax.diag_line_count == 0)
    ax.grid_diagonal(True)
    fig.canvas.draw()
    check("on -> diagonals back", ax.diag_line_count > 0)

    # 7. User data plot
    section("7. user data plot")
    f = np.logspace(0, 3, 100)
    v = 10.0 / np.sqrt(1.0 + (f / 50.0) ** 2)
    line, = ax.plot(f, v, label="test")
    fig.canvas.draw()
    check("user line added", line in ax.lines)
    check("legend works", ax.legend() is not None)
    plt.close(fig)

    # 8. SI units
    section("8. SI units")
    fig, ax_si = plt.subplots(subplot_kw={"projection": "tripartite", "units": "SI"})
    ax_si.set_xlim(1, 1000); ax_si.set_ylim(0.01, 10)
    fig.canvas.draw()
    check("SI velocity label has m/s", "m/s" in ax_si.get_ylabel())
    check("SI accel labels use non-g unit",
          all("g" != t.get_text().split(" ")[-1] for t in ax_si._accel_labels))
    plt.close(fig)

    # 9. PNG render
    section("9. headless PNG render")
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    ax.plot(f, v, color="red"); ax.grid_diagonal(True)
    out = Path(__file__).with_name("_debug_render.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    check("PNG written", out.exists())
    if out.exists():
        check("PNG size reasonable", 10_000 < out.stat().st_size < 5_000_000,
              f"{out.stat().st_size} bytes")

    # 10. Cache hit rate
    section("10. cache layer (no redundant rebuild)")
    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
    fig.canvas.draw()
    rebuild_calls = [0]
    orig = ax._rebuild_diagonals
    def counting():
        rebuild_calls[0] += 1
        return orig()
    ax._rebuild_diagonals = counting
    for _ in range(20):
        fig.canvas.draw()
    check("20 unchanged draws => 0 rebuilds", rebuild_calls[0] == 0,
          f"rebuilds={rebuild_calls[0]}")
    # change view once
    ax.set_xlim(2, 500)
    fig.canvas.draw()
    fig.canvas.draw()
    check("1 view change => 1 rebuild", rebuild_calls[0] == 1,
          f"rebuilds={rebuild_calls[0]}")

    # 11. Speed benchmarks
    section("11. speed benchmarks (Agg backend)")

    def mkfig():
        f2, a2 = plt.subplots(subplot_kw={"projection": "tripartite"})
        a2.set_xlim(1, 1000); a2.set_ylim(0.1, 100)
        return f2, a2

    # first draw
    f2, a2 = mkfig()
    t0 = time.perf_counter()
    f2.canvas.draw()
    first_ms = (time.perf_counter() - t0) * 1000
    print(f"  first draw:      {first_ms:7.2f} ms")
    check("first draw < 500 ms", first_ms < 500)
    plt.close(f2)

    # cached (same view)
    f2, a2 = mkfig()
    f2.canvas.draw()  # warm
    med, lo, hi = bench("cached", f2.canvas.draw, n=20)
    print(f"  cached draw x20: median={med*1000:7.2f} ms  "
          f"(min {lo*1000:.2f}, max {hi*1000:.2f})")
    check("cached draw median < 80 ms", med < 0.08)
    plt.close(f2)

    # zoom (new view each time, inside tight loop)
    f2, a2 = mkfig()
    f2.canvas.draw()
    xhi_cycle = iter([])

    def zoom_draw():
        nonlocal xhi_cycle
        try:
            xhi = next(xhi_cycle)
        except StopIteration:
            xhi_cycle = iter([100, 500, 2000, 10_000, 50, 1000, 200])
            xhi = next(xhi_cycle)
        a2.set_xlim(1, xhi)
        f2.canvas.draw()

    med, lo, hi = bench("zoom", zoom_draw, n=20)
    print(f"  zoom draw x20:   median={med*1000:7.2f} ms  "
          f"(min {lo*1000:.2f}, max {hi*1000:.2f})")
    check("zoom draw median < 150 ms", med < 0.15)
    plt.close(f2)

    # 12. Stress
    section("12. stress (500 random zooms)")
    f2, a2 = mkfig()
    f2.canvas.draw()
    rng = np.random.default_rng(7)
    t0 = time.perf_counter()
    max_labels = 0
    for _ in range(500):
        lo = float(10 ** rng.uniform(-2, 1))
        hi = lo * float(10 ** rng.uniform(0.5, 4))
        a2.set_xlim(lo, hi)
        ylo = float(10 ** rng.uniform(-3, 0))
        yhi = ylo * float(10 ** rng.uniform(0.5, 4))
        a2.set_ylim(ylo, yhi)
        f2.canvas.draw()
        max_labels = max(max_labels, a2.diag_label_count)
    dt = time.perf_counter() - t0
    print(f"  500 random-view draws: {dt:.2f} s  "
          f"(avg {dt*1000/500:.2f} ms/draw, peak labels {max_labels})")
    check("stress avg < 150 ms/draw", dt / 500 < 0.15)
    check("final labels == lines", a2.diag_line_count == a2.diag_label_count)
    plt.close(f2)

    # Summary
    print("\n" + "=" * 60)
    print(f"PASS: {_PASS}    FAIL: {_FAIL}")
    if _FAIL:
        print("SOMETHING BROKE.")
        return 1
    print("ALL FINE. Package is fast and permanent.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
