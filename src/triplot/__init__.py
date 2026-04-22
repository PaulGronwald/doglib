"""triplot — matplotlib tripartite (shock response spectrum) projection.

Importing the package registers the 'tripartite' projection with
matplotlib, so::

    fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})

works out of the box. The high-level :func:`plot` / :func:`subplots`
helpers save the boilerplate.

A previous revision of this module shipped a pluggable plotly backend
alongside matplotlib. It was stashed to branch ``archive/plotly-backend``
while the rendering quality / update performance get sorted out —
see also ``src/triplot/backends/base.py`` which keeps the backend
abstraction in place so the work can resume without redoing the
refactor.

Quick start::

    import triplot
    fig, ax = triplot.plot(freq, pv)
    # or:
    fig, ax = triplot.subplots(units="SI")
    ax.plot(freq, pv)
"""
from matplotlib.projections import register_projection

from .axes import TripartiteAxes
from .units import IMPERIAL, SI, UnitSystem

register_projection(TripartiteAxes)


def subplots(*, figsize=None, **axes_kwargs):
    """Create a figure with a single tripartite Axes.

    Thin wrapper around ``plt.subplots(subplot_kw={"projection": "tripartite", ...})``.
    All keyword args are forwarded to :class:`TripartiteAxes` — e.g.
    ``units="SI"``, ``style="dplot"``, ``label_mode="midpoint"``,
    ``aspect="equal"``.

    Returns ``(fig, ax)``.
    """
    import matplotlib.pyplot as plt
    kwargs = {}
    if figsize is not None:
        kwargs["figsize"] = figsize
    return plt.subplots(
        subplot_kw={"projection": "tripartite", **axes_kwargs}, **kwargs,
    )


def plot(freq=None, pv=None, *, figsize=None, **axes_kwargs):
    """Create a tripartite plot and optionally draw ``(freq, pv)`` on it.

    The shortest path from arrays to a rendered tripartite plot::

        fig, ax = triplot.plot(freq, pv)
        plt.show()

    When ``freq`` and ``pv`` are supplied, the initial viewport is
    auto-fit to the data with a small log-space margin — the user sees
    the curve filling the plot on first render instead of the default
    ``[1, 1000] × [0.1, 100]`` window. Subsequent pan/zoom proceed
    from this viewport; the autoscale only runs once at creation.

    Passing ``freq=None`` / ``pv=None`` is equivalent to :func:`subplots`
    — useful when you want to add multiple curves later.

    Returns ``(fig, ax)``.
    """
    fig, ax = subplots(figsize=figsize, **axes_kwargs)
    if freq is not None and pv is not None:
        ax.plot(freq, pv)
        _fit_viewport_to_data(ax, freq, pv)
    return fig, ax


def _fit_viewport_to_data(ax, freq, pv) -> None:
    """Set initial xlim / ylim to cover ``(freq, pv)`` with a small
    log-space margin. Runs once at plot creation; pan/zoom afterwards
    can take the viewport anywhere without being reset."""
    import math
    try:
        fmin = min(float(v) for v in freq if v > 0)
        fmax = max(float(v) for v in freq if v > 0)
        vmin = min(float(v) for v in pv if v > 0)
        vmax = max(float(v) for v in pv if v > 0)
    except (ValueError, TypeError):
        return  # empty / non-positive — leave default viewport in place
    if not (fmin < fmax and vmin < vmax):
        return
    # Margin of ~10% of the data span in log space so endpoints aren't
    # glued to the spines.
    mx = max(0.05 * (math.log10(fmax) - math.log10(fmin)), 0.05)
    my = max(0.05 * (math.log10(vmax) - math.log10(vmin)), 0.05)
    ax.set_xlim(10 ** (math.log10(fmin) - mx), 10 ** (math.log10(fmax) + mx))
    ax.set_ylim(10 ** (math.log10(vmin) - my), 10 ** (math.log10(vmax) + my))


__all__ = [
    "TripartiteAxes", "UnitSystem", "IMPERIAL", "SI",
    "plot", "subplots",
]
__version__ = "0.1.0"
