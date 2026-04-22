"""Write one static HTML per zoom level to demonstrate that the picker
adapts gridline density across the full scalable range.

Produces ``plotly_zoom_*.html`` files for:
  (1) 3-decade default viewport   — full {1..9} gridlines
  (2) 1-decade zoom               — full density, narrower mantissa set
  (3) sub-decade narrow zoom      — progressive picker kicks in
  (4) 6-decade wide viewport      — density thinned to {1, 2, 5}
  (5) 12-decade ultra-wide        — {1, 3}, avoids clutter
  (6) 20-decade extreme           — one major per decade

Run::

    python examples/plotly_zoom_levels.py

This is the test for "zooming out used to flood the plot with gridlines".
Every file should render without visible clutter at its respective scale.
"""
import numpy as np
import plotly.io as pio

import triplot

pio.renderers.default = "browser"

CASES = [
    ("1_default_3dec",   (1.0, 1000.0),   (0.1, 100.0),   "3-decade default"),
    ("2_narrow_1dec",    (10.0, 100.0),   (1.0, 10.0),    "1-decade zoom"),
    ("3_subdecade",      (20.0, 25.0),    (5.0, 6.0),     "sub-decade narrow zoom"),
    ("4_wide_6dec",      (1e-3, 1e3),     (1e-4, 1e2),    "6-decade wide viewport"),
    ("5_ultrawide_12dec",(1e-6, 1e6),     (1e-6, 1e6),    "12-decade ultra-wide"),
    ("6_extreme_20dec",  (1e-10, 1e10),   (1e-10, 1e10),  "20-decade extreme"),
]

for tag, (xlo, xhi), (ylo, yhi), title in CASES:
    f = np.logspace(np.log10(xlo), np.log10(xhi), 400)
    pv = (yhi * 0.8) / np.sqrt(1 + (f / np.sqrt(xlo * xhi)) ** 2)
    fig = triplot.plot(f, pv, backend="plotly", figsize=(10, 8))
    fig.update_xaxes(range=[np.log10(xlo), np.log10(xhi)])
    fig.update_yaxes(range=[np.log10(ylo), np.log10(yhi)])
    fig._triplot_rebuild()
    fig.update_layout(title=f"triplot — {title}")
    path = f"plotly_zoom_{tag}.html"
    fig.write_html(path, include_plotlyjs="cdn")
    core = fig._triplot_core
    disp = len(core.last_disp_segments)
    accel = len(core.last_accel_segments)
    print(f"{tag:22s}  disp={disp:3d}  accel={accel:3d}  -> {path}")

print()
print("Open the HTML files in a browser. Gridline density should stay")
print("readable across all six zoom levels — that's the adaptive picker.")
