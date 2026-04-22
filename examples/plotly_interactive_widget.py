"""FigureWidget interactive demo — for use inside Jupyter notebooks.

In a notebook::

    %run examples/plotly_interactive_widget.py

This returns a ``plotly.graph_objects.FigureWidget``. The widget's
``on_relayout`` event is wired to the core's rebuild callback, so
zooming / panning in the rendered cell automatically refreshes the
diagonals.

Requires ``anywidget`` (plotly 6+ pulled out its ipywidgets glue into a
separate package). Install with::

    pip install anywidget

Run outside a notebook and you'll just see diagnostic prints — the
widget needs ipywidgets + a kernel to display. For a live-rescale
browser experience without a notebook, use ``plotly_dash_viewer.py``
instead.
"""
import numpy as np

import triplot


def build_widget():
    f = np.logspace(0, 3, 400)
    pv = 50 / np.sqrt(1 + (f / 30) ** 2) * np.sqrt(1 + (f / 300) ** 2)

    # interactive=True returns a FigureWidget — on_relayout wires to
    # core.rebuild() so zoom / pan refresh the diagonals in-place.
    fig = triplot.plot(
        f, pv, backend="plotly", interactive=True, figsize=(10, 8),
    )
    fig.update_layout(title="triplot FigureWidget — zoom / pan to refresh diagonals")
    return fig


if __name__ == "__main__":
    try:
        fig = build_widget()
    except ImportError as exc:
        print(f"Widget requires anywidget: {exc}")
        print("Install with: pip install anywidget")
        raise SystemExit(1)
    print(f"Built {type(fig).__name__}")
    print(f"  shapes:      {len(fig.layout.shapes)}")
    print(f"  annotations: {len(fig.layout.annotations)}")
    print(f"  traces:      {len(fig.data)}")
    print()
    print("Display this widget inside a Jupyter cell to interact with it,")
    print("or call build_widget() from an ipykernel-backed REPL.")
