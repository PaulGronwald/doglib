"""Pluggable rendering backends for triplot.

Currently ships with a matplotlib backend only. The plotly backend was
stashed to ``archive/plotly-backend`` until its update performance +
render fidelity catch up with the matplotlib path; the shared
abstractions here (:class:`Backend`, :class:`LabelItem`, :class:`TickItem`,
:class:`DiagramFamily`) are preserved so a future plotly revive doesn't
need another refactor.
"""
from .base import Backend, LabelItem, TickItem, DiagramFamily

__all__ = ["Backend", "LabelItem", "TickItem", "DiagramFamily"]
