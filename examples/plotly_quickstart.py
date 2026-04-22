"""Plotly quickstart — single tripartite plot, opened in the browser.

Run::

    python examples/plotly_quickstart.py

Plotly's default renderer when no Jupyter kernel is attached is
``'browser'`` — calling ``fig.show()`` writes a temp HTML file and opens
your default browser to view it. The diagonals, labels, and edge ticks
are all rendered server-side by the triplot core at current axis limits;
the resulting HTML is a self-contained snapshot.

Note: the HTML export is static in the sense that our Python picker
never reruns on the client — plotly.js will pan/zoom the existing shapes
but won't refine tick subdivisions at extreme zoom. For live refinement
use ``examples/plotly_dash_viewer.py`` (Dash app) or a Jupyter notebook
with ``interactive=True`` (see ``plotly_interactive_widget.py``).
"""
import numpy as np
import plotly.io as pio

import triplot

# Force the browser renderer so fig.show() opens a browser tab even when
# running outside a notebook. Without this, plotly falls back to a
# non-interactive mime bundle and silently does nothing in terminal Python.
pio.renderers.default = "browser"

# Classic shock response spectrum — two-pole behaviour with corner freq 30 Hz.
f = np.logspace(0, 3, 400)
pv = 50 / np.sqrt(1 + (f / 30) ** 2) * np.sqrt(1 + (f / 300) ** 2)

fig = triplot.plot(f, pv, backend="plotly", figsize=(10, 8))
fig.update_layout(title="triplot quickstart — SRS in plotly (browser renderer)")
fig.show()
