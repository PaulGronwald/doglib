"""Dump every label currently drawn, with its (f, v) position and the value
the label represents. Sanity check: the line passing through (f, v) with the
expected slope should have that constant."""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import triplot  # noqa: F401

TWO_PI = 2 * math.pi
G = 386.089

fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
ax.set_xlim(1, 1000); ax.set_ylim(0.1, 100)
fig.canvas.draw()

def inspect(label_list, segs, slope, kind, unit_factor=1.0):
    print(f"\n=== {kind} ({len(label_list)} labels) ===")
    for t, seg in zip(label_list, segs):
        f, v = t.get_position()
        text = t.get_text()
        # recover "what this line actually is" from geometry
        if slope == +1:      # disp: d = v / (2*pi*f)
            derived = v / (TWO_PI * f)
        else:                # accel: a = 2*pi*f*v in raw units; convert to g if needed
            derived = TWO_PI * f * v / unit_factor
        rel_err = abs(derived - seg.value) / seg.value
        ok = rel_err < 1e-9
        status = "OK  " if ok else "BAD "
        print(f"  {status} label='{text}' at (f={f:.4g}, v={v:.4g}) "
              f"stored={seg.value:.4g} derived={derived:.4g}")

from triplot import diagonals as d
subs = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
label_subs = (1.0, 2.0, 4.0, 6.0, 8.0)

disp_segs = [d.displacement_segment(v, ax.get_xlim(), ax.get_ylim())
             for v in d.pick_displacement_values(ax.get_xlim(), ax.get_ylim(), subs)]
disp_segs = [s for s in disp_segs if s is not None]
from triplot.axes import _is_label_value
disp_label_segs = [s for s in disp_segs if _is_label_value(s.value, label_subs)]

accel_segs = [d.acceleration_segment(v, ax.get_xlim(), ax.get_ylim(), g_value=G)
              for v in d.pick_acceleration_values(ax.get_xlim(), ax.get_ylim(), g_value=G, subdivisions=subs)]
accel_segs = [s for s in accel_segs if s is not None]
accel_label_segs = [s for s in accel_segs if _is_label_value(s.value, label_subs)]

inspect(ax._disp_labels, disp_label_segs, +1, "displacement (in)")
inspect(ax._accel_labels, accel_label_segs, -1, "acceleration (g)", unit_factor=G)
