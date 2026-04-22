"""Pluggable rendering backends for triplot.

Pick a backend with ``triplot.plot(backend='matplotlib')`` or
``triplot.plot(backend='plotly')``. The shared abstractions live in
:mod:`triplot.backends.base`; each concrete backend in its own module.
"""
from .base import Backend, LabelItem, TickItem, DiagramFamily

__all__ = ["Backend", "LabelItem", "TickItem", "DiagramFamily"]
