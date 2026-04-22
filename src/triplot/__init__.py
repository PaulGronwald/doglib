"""triplot — tripartite (shock response spectrum) plots with a pluggable
rendering backend.

Importing the package registers the 'tripartite' projection with
matplotlib (so ``plt.subplots(subplot_kw={"projection": "tripartite"})``
works). The high-level :func:`plot` / :func:`subplots` helpers accept a
``backend=`` argument that picks matplotlib or plotly at execution
time — same core, same math, same visual output, different render
target.

Quick start (matplotlib, the default)::

    import triplot
    fig, ax = triplot.plot(freq, pv)                         # mpl
    fig = triplot.plot(freq, pv, backend="plotly")           # plotly go.Figure

Interactive plotly in a notebook::

    fig = triplot.plot(freq, pv, backend="plotly", interactive=True)
    fig   # FigureWidget — diagonals re-snap on zoom / pan
"""
from matplotlib.projections import register_projection

from .axes import TripartiteAxes
from .units import IMPERIAL, SI, UnitSystem

register_projection(TripartiteAxes)

_VALID_BACKENDS = ("matplotlib", "mpl", "plotly")


def _mpl_subplots(*, figsize=None, **axes_kwargs):
    import matplotlib.pyplot as plt
    kwargs = {}
    if figsize is not None:
        kwargs["figsize"] = figsize
    return plt.subplots(
        subplot_kw={"projection": "tripartite", **axes_kwargs}, **kwargs,
    )


def _plotly_figure(*, figsize=None, interactive=False, **core_kwargs):
    """Build a plotly tripartite figure.

    Returns ``(fig, core, backend)`` so the caller can attach data
    traces before or after the first rebuild. Set ``interactive=True`` to
    get a ``FigureWidget`` with relayout-wired rescale callbacks (needs
    ipywidgets / running in a notebook).
    """
    import plotly.graph_objects as go
    from .core import TripartiteCore
    from .backends.plotly_backend import PlotlyBackend
    from .units import resolve as _resolve_units

    units = _resolve_units(core_kwargs.pop("units", None))
    style = core_kwargs.pop("style", "seismic")
    if style not in ("seismic", "shock", "dplot"):
        raise ValueError("style must be 'seismic', 'shock', or 'dplot'")
    label_mode = core_kwargs.pop("label_mode", "edge")
    if label_mode not in ("edge", "midpoint"):
        raise ValueError("label_mode must be 'edge' or 'midpoint'")
    show_diag_titles = core_kwargs.pop("show_diag_titles", None)
    aspect = core_kwargs.pop("aspect", "equal")

    width, height = (700, 500)
    if figsize is not None:
        # Matplotlib figsize is inches; plotly wants pixels. 100 dpi gives
        # a close visual match without the plot feeling tiny.
        width, height = int(figsize[0] * 100), int(figsize[1] * 100)

    fig_cls = go.FigureWidget if interactive else go.Figure
    fig = fig_cls()
    fig.update_layout(
        width=width, height=height,
        margin=dict(l=70, r=90, t=50, b=60),
        showlegend=False,
        plot_bgcolor="white",
    )
    fig.update_xaxes(title_text=units.freq_label, showgrid=True, gridcolor="#CCC")
    fig.update_yaxes(title_text=units.vel_label, showgrid=True, gridcolor="#CCC")

    core = TripartiteCore(
        units=units,
        style=style,
        label_mode=label_mode,
        show_diag_titles=show_diag_titles,
    )
    backend = PlotlyBackend(fig, log_x=True, log_y=True)

    # Aspect: plotly's scaleanchor trick already enforces equal log-space
    # ratios; disable when the user asks for 'auto'.
    if aspect == "auto":
        fig.update_yaxes(scaleanchor=None)

    # Initial seed — give the figure a sensible range so the first rebuild
    # has a valid viewport (plotly's autorange is None until a trace lands).
    fig.update_xaxes(range=[0.0, 3.0])   # log10 — i.e. [1, 1000]
    fig.update_yaxes(range=[-1.0, 2.0])  # log10 — i.e. [0.1, 100]

    def _rebuild():
        core.rebuild(backend)

    backend.connect_rescale(_rebuild)
    _rebuild()

    # Stash references for later introspection / test use.
    fig._triplot_core = core
    fig._triplot_backend = backend
    fig._triplot_rebuild = _rebuild

    return fig


def subplots(*, figsize=None, backend="matplotlib", **axes_kwargs):
    """Create a tripartite figure with no data attached.

    ``backend='matplotlib'`` (default) returns ``(fig, ax)``. ``backend='plotly'``
    returns the plotly ``Figure`` or ``FigureWidget`` (no Axes — plotly
    layouts don't separate figure from axes the way matplotlib does).
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")
    if backend == "plotly":
        return _plotly_figure(figsize=figsize, **axes_kwargs)
    return _mpl_subplots(figsize=figsize, **axes_kwargs)


def plot(freq=None, pv=None, *, figsize=None, backend="matplotlib", **axes_kwargs):
    """Create a tripartite plot and optionally draw ``(freq, pv)`` on it.

    ``backend='matplotlib'``: returns ``(fig, ax)`` — the legacy behaviour.
    ``backend='plotly'``: returns the plotly Figure / FigureWidget.

    ``interactive=True`` (plotly only) returns a ``FigureWidget`` that
    re-snaps its diagonal grid on zoom / pan / resize — the ticks and
    overflow labels are themselves indicators, not data, so they rescale
    automatically in response to axis range changes.
    """
    if backend not in _VALID_BACKENDS:
        raise ValueError(f"backend must be one of {_VALID_BACKENDS}, got {backend!r}")

    if backend == "plotly":
        interactive = axes_kwargs.pop("interactive", False)
        fig = _plotly_figure(figsize=figsize, interactive=interactive, **axes_kwargs)
        if freq is not None and pv is not None:
            import plotly.graph_objects as go
            fig.add_trace(go.Scatter(x=list(freq), y=list(pv), mode="lines"))
            # Re-run rebuild so the range update for the new trace picks up
            # fresh diagonal placements.
            fig._triplot_rebuild()
        return fig

    fig, ax = _mpl_subplots(figsize=figsize, **axes_kwargs)
    if freq is not None and pv is not None:
        ax.plot(freq, pv)
    return fig, ax


__all__ = [
    "TripartiteAxes", "UnitSystem", "IMPERIAL", "SI",
    "plot", "subplots",
]
__version__ = "0.1.0"
