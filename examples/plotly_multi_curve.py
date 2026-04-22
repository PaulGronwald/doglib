"""Multiple SRS curves on one tripartite plot (plotly backend).

Demonstrates that user data traces layer cleanly over the diagonal grid:
triplot's gridlines / labels are ``layout.shapes`` / ``layout.annotations``
with ``layer='below'``, so any number of scatter traces draw on top
without interfering.

Run::

    python examples/plotly_multi_curve.py
"""
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import triplot

pio.renderers.default = "browser"


def srs(f, fn, zeta):
    r = f / fn
    return (fn * 0.5) / np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)


f = np.logspace(-1, 3, 500)

# Build an empty triplot figure, then add user traces on top — this is the
# analogue of 'fig, ax = triplot.subplots(); ax.plot(...)' in matplotlib.
fig = triplot.subplots(backend="plotly", style="dplot", figsize=(10, 8))
fig.update_xaxes(range=[-1, 3])
fig.update_yaxes(range=[-3, 1])

for fn, zeta, color, name in [
    (3.0,   0.01, "#c0392b", "3 Hz, 1% damping"),
    (3.0,   0.05, "#e67e22", "3 Hz, 5% damping"),
    (10.0,  0.05, "#27ae60", "10 Hz, 5% damping"),
    (30.0,  0.05, "#2980b9", "30 Hz, 5% damping"),
    (100.0, 0.05, "#8e44ad", "100 Hz, 5% damping"),
]:
    fig.add_trace(go.Scatter(
        x=f.tolist(), y=srs(f, fn, zeta).tolist(),
        mode="lines", line=dict(color=color, width=2),
        name=name,
    ))

# Rebuild so the diagonals refresh against whatever range the traces set,
# then show. Triplot's core doesn't auto-listen to trace additions on a
# plain Figure (only FigureWidget relayout triggers), so we call rebuild
# explicitly after layering data.
fig._triplot_rebuild()
fig.update_layout(
    title="triplot — five SRS curves, DPlot style, plotly backend",
    showlegend=True,
    legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
)
fig.show()
