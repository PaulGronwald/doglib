# triplot

Matplotlib projection for **tripartite (four-coordinate) log-log plots** used in
shock response spectra, earthquake engineering, and vibration analysis.

Frequency on X, pseudo-velocity on Y, constant-displacement lines at +45°,
constant-acceleration lines at −45°. Grid regenerates dynamically on zoom/pan —
no precomputed fixed range.

## Install

```bash
pip install -e .
```

## Use

Shortest path:

```python
import numpy as np
import matplotlib.pyplot as plt
import triplot

freq = np.logspace(0, 3, 200)
pv = 10 / (1 + (freq / 50) ** 2) ** 0.5

fig, ax = triplot.plot(freq, pv)          # one-liner
plt.show()
```

Long form (equivalent):

```python
fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
ax.plot(freq, pv)
```

## Configuration knobs

All pass through as `subplot_kw={"projection": "tripartite", ...}` or as
kwargs to `triplot.plot()` / `triplot.subplots()`.

| knob               | values                   | default   | effect                                                      |
|--------------------|--------------------------|-----------|-------------------------------------------------------------|
| `units`            | `"imperial"` / `"SI"`    | imperial  | Axis units (in/s vs m/s, g vs m/s²)                         |
| `style`            | `"seismic"`/`"dplot"`/`"shock"` | seismic | Grid density + label density preset                     |
| `aspect`           | `"equal"` / `"auto"`     | equal     | `"auto"` lets the plot be squished asymmetrically           |
| `label_mode`       | `"edge"` / `"midpoint"`  | edge      | Where diagonal-line labels appear — never both              |
| `show_diag_titles` | `True`/`False`/`None`    | None (style-dep) | Big centered "Displacement (in)" / "Acceleration (g)" |

Runtime setters: `ax.set_label_mode(...)`, `ax.set_show_diag_titles(...)`.
Changes invalidate the draw cache so the view redraws on the next frame.

### Label modes

`label_mode="edge"` (default) places one rotated label per in-view line at
its edge crossing, outside the spine by default. Lines that miss the
designated edge (top for displacement, right for acceleration) get a
fallback label inside the plot, on the line, with a white glyph stroke so
the text reads over the grid without erasing it.

`label_mode="midpoint"` labels each line at its midpoint inside the plot
instead (the classic tripartite look).

### UX in edge mode

When `label_mode="edge"`:

- `ax.set_title(...)` auto-adds extra `pad` (the top margin holds disp
  labels) AND enables matplotlib's constrained layout so the extra height
  is reserved in the figure grid — the title won't overflow into subplots
  above.
- `ax.legend()` defaults to `loc="upper left"` — the only uncluttered
  corner when edge labels are drawn on the other three sides.
- Pass `pad=...` / `loc=...` explicitly to opt out of the auto-behavior.

## Interactive 6-panel gallery

Opens a TkAgg/Qt window with six different configurations (seismic, DPlot,
SI units, extreme-range, narrow-zoom, minimalist shock). Scroll wheel = zoom
at cursor, left-drag = pan. Blit-accelerated so all six panels stay
responsive.

```bash
python scripts/gallery_viewer.py
```

A static version that writes `examples/gallery.png` instead:

```bash
python examples/gallery.py
```
