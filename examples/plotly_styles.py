"""Compare the three tripartite styles side by side in plotly.

Opens three browser tabs (one per style). The same shock-response curve
is drawn in ``'seismic'`` (dense {1..9} gridlines + {1,2,4,6,8} labels),
``'dplot'`` (dense gridlines + {1,5} labels only), and ``'shock'`` (one
sparse line per decade — minimalist industrial look).

Run::

    python examples/plotly_styles.py
"""
import numpy as np
import plotly.io as pio

import triplot

pio.renderers.default = "browser"

f = np.logspace(0, 3, 400)
# Representative SRS with realistic roll-off
pv = 50 / np.sqrt(1 + (f / 30) ** 2) * np.sqrt(1 + (f / 300) ** 2)

for style in ("seismic", "dplot", "shock"):
    fig = triplot.plot(
        f, pv, backend="plotly", style=style, figsize=(10, 8),
    )
    fig.update_layout(
        title=f"triplot — style={style!r}",
    )
    fig.show()
