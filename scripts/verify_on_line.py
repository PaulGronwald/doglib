"""Prove each reference point sits on its labeled diagonal — programmatically.

For every ref point (f, v) that should be on a line of constant d or a,
find the drawn segment with that value and measure perpendicular distance
in log-log space. Any gap > 1e-9 means the math is wrong.
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import triplot  # noqa: F401

TWO_PI = 2 * math.pi
G = 386.089

checks = [
    ("disp", 1.0,    10.0,  TWO_PI * 10.0 * 1.0),         # d=1in at 10Hz
    ("disp", 1.0,    1.0,   TWO_PI * 1.0 * 1.0),          # d=1in at 1Hz
    ("disp", 0.1,    10.0,  TWO_PI * 10.0 * 0.1),         # d=0.1in at 10Hz
    ("disp", 0.01,   100.0, TWO_PI * 100.0 * 0.01),       # d=0.01in at 100Hz
    ("accel", 1.0,   100.0, G / (TWO_PI * 100.0)),        # a=1g at 100Hz
    ("accel", 10.0,  100.0, 10.0 * G / (TWO_PI * 100.0)), # a=10g at 100Hz
    ("accel", 0.1,   10.0,  0.1 * G / (TWO_PI * 10.0)),   # a=0.1g at 10Hz
    ("accel", 100.0, 10.0,  100.0 * G / (TWO_PI * 10.0)), # a=100g at 10Hz (v=614)
]

fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
ax.set_xlim(0.1, 10000)
ax.set_ylim(0.01, 10000)
fig.canvas.draw()

def seg_for(collection, value):
    for line in collection.get_segments():
        (x0, y0), (x1, y1) = line
        # which constant does this segment represent? infer from slope + intercept
        # disp: y = 2*pi*d*x => d = y/(2*pi*x)
        # accel: y = a/(2*pi*x) => a = 2*pi*x*y, with a in raw units
        # we pass through the value by matching the seg in collection order below instead.
        yield (x0, y0), (x1, y1)

fails = 0
for kind, val, fref, vref in checks:
    if kind == "disp":
        seg = __import__("triplot.diagonals", fromlist=["displacement_segment"]).displacement_segment(
            val, ax.get_xlim(), ax.get_ylim())
    else:
        seg = __import__("triplot.diagonals", fromlist=["acceleration_segment"]).acceleration_segment(
            val, ax.get_xlim(), ax.get_ylim(), g_value=G)
    assert seg is not None, f"segment missing for {kind}={val}"
    # parametrize: log_v(f) = log_v0 + slope*(log_f - log_f0), slope = +1 or -1
    slope = 1 if kind == "disp" else -1
    lv_on_line = math.log10(seg.v0) + slope * (math.log10(fref) - math.log10(seg.f0))
    v_on_line = 10 ** lv_on_line
    rel_err = abs(v_on_line - vref) / vref
    ok = rel_err < 1e-9
    fails += 0 if ok else 1
    status = "OK  " if ok else "FAIL"
    print(f"{status} {kind:>5s} val={val:<7g} at f={fref:<7g} expect v={vref:<12.6f} "
          f"line gives v={v_on_line:<12.6f} rel_err={rel_err:.2e}")

print(f"\n{'ALL PASS' if fails == 0 else f'{fails} FAILURES'}")
