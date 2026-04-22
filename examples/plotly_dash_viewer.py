"""Dash viewer — live tick-refinement as you zoom the browser plot.

``triplot.plot(backend='plotly')`` returns a static go.Figure: plotly.js
pans and zooms client-side, but our Python picker never reruns so
diagonals don't refine when you zoom in 100x or zoom out 10 decades.
This viewer wires a server-side Dash callback: every relayout event
(pan / scroll / rect-zoom / reset) ships the new axis range back,
``TripartiteCore.rebuild()`` runs, and the updated shapes + annotations
are pushed to the client.

Run::

    pip install dash           # not a core triplot dependency
    python examples/plotly_dash_viewer.py

Then open http://127.0.0.1:8050 in any browser. Drag to pan, scroll to
zoom, box-select from the toolbar for rect-zoom, and double-click to
reset. Each triggers a Python rebuild — gridline density adapts to the
visible decade span and narrow zooms refine down through progressive
subdivisions.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from dash import Dash, Input, Output, dcc, html

import triplot
from triplot.backends.plotly_backend import PlotlyBackend
from triplot.core import TripartiteCore
from triplot.units import resolve as resolve_units


# Module-scope state: Dash callbacks need the same core / backend / figure
# across invocations so updates mutate the same shapes instead of stacking
# duplicates. This is the single-user debug-viewer pattern — fine for a
# local tool, would need per-session state for a real multi-user app.

f = np.logspace(0, 3, 400)
pv = 50 / np.sqrt(1 + (f / 30) ** 2) * np.sqrt(1 + (f / 300) ** 2)

fig = go.Figure()
fig.add_trace(go.Scatter(x=f.tolist(), y=pv.tolist(), mode="lines",
                         name="PV", line=dict(color="#2c3e50", width=2)))
fig.update_layout(
    width=1000, height=750,
    margin=dict(l=70, r=90, t=50, b=60),
    plot_bgcolor="white",
    dragmode="pan",
    showlegend=False,
)
fig.update_xaxes(type="log", range=[0.0, 3.0], title_text="Frequency [Hz]",
                 showgrid=True, gridcolor="#DDD")
fig.update_yaxes(type="log", range=[-1.0, 2.0], title_text="Pseudo-Velocity [in/s]",
                 showgrid=True, gridcolor="#DDD")

core = TripartiteCore(units=resolve_units("imperial"), style="seismic", label_mode="edge")
backend = PlotlyBackend(fig, log_x=True, log_y=True)
# Drop scaleanchor — it clamps the aspect and fights interactive zoom on
# log axes, same as matplotlib's adjustable='box' vs 'datalim' trade-off.
fig.update_yaxes(scaleanchor=None)
core.rebuild(backend)


app = Dash(__name__)
app.layout = html.Div(
    style={"fontFamily": "system-ui, sans-serif", "padding": "12px"},
    children=[
        html.H2("triplot — Dash viewer (live tick refinement on zoom)"),
        html.Div(
            "Drag = pan · scroll = zoom · toolbar box-zoom = rect-zoom · "
            "double-click = reset. Every layout change re-runs the Python "
            "picker so gridline density tracks the visible decade span.",
            style={"color": "#555", "marginBottom": "8px", "fontSize": "13px"},
        ),
        dcc.Graph(id="triplot-graph", figure=fig, config={"scrollZoom": True}),
    ],
)


def _extract_range(relayout_data, axis: str):
    """Pull the new axis range out of a plotly relayout event. Plotly
    ships axis changes as either ``{axis.range: [lo, hi]}`` (single key
    for combined updates) or ``{axis.range[0]: lo, axis.range[1]: hi}``
    (two keys for individual updates) depending on the interaction."""
    if relayout_data is None:
        return None
    full = relayout_data.get(f"{axis}.range")
    if isinstance(full, (list, tuple)) and len(full) == 2:
        return float(full[0]), float(full[1])
    lo = relayout_data.get(f"{axis}.range[0]")
    hi = relayout_data.get(f"{axis}.range[1]")
    if lo is not None and hi is not None:
        return float(lo), float(hi)
    return None


@app.callback(
    Output("triplot-graph", "figure"),
    Input("triplot-graph", "relayoutData"),
    prevent_initial_call=True,
)
def _on_relayout(relayout_data):
    if relayout_data is None:
        return fig.to_dict()

    if relayout_data.get("xaxis.autorange") or relayout_data.get("yaxis.autorange"):
        fig.update_xaxes(range=[0.0, 3.0])
        fig.update_yaxes(range=[-1.0, 2.0])
    else:
        xrange = _extract_range(relayout_data, "xaxis")
        yrange = _extract_range(relayout_data, "yaxis")
        if xrange is not None:
            fig.update_xaxes(range=list(xrange))
        if yrange is not None:
            fig.update_yaxes(range=list(yrange))
    core.rebuild(backend)
    # Return a plain dict — returning the same Figure object makes Dash
    # diff against itself (no props-change detected) and the client keeps
    # the stale shapes. Dict forces a full re-send.
    return fig.to_dict()


if __name__ == "__main__":
    # debug=False — the auto-reloader re-imports the module, which creates
    # a fresh fig + backend on every reload and defeats the shared-state
    # pattern. Restart manually after edits.
    app.run(debug=False, host="127.0.0.1", port=8050)
