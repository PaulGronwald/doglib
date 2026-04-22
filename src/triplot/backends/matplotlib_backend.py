"""Matplotlib backend for the tripartite core.

Wraps a :class:`matplotlib.axes.Axes` and translates core draw requests
into artist pool updates — :class:`LineCollection` for diagonal
gridlines, :class:`Annotation` for rotated labels, :class:`Line2D` for
edge ticks. Artist pools grow / shrink in place so repeated zoom never
leaks.

Interactivity: the backend wires ``xlim_changed`` and ``ylim_changed``
callbacks on the wrapped Axes so pan / zoom triggers a core rebuild.
Matplotlib already calls ``Axes.draw()`` on resize, so a draw-hook
fallback catches aspect changes and DPI changes too.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib import patheffects
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text
from matplotlib.transforms import blended_transform_factory

from .base import Backend, BackendStyle, DiagramFamily, LabelItem, TickItem


_COLLECTION_KEYMAP = {"color": "colors", "linewidth": "linewidths", "linestyle": "linestyles"}


def _to_collection_kwargs(style: dict) -> dict:
    out = {}
    for k, v in style.items():
        out[_COLLECTION_KEYMAP.get(k, k)] = v
    return out


_HA_MAP = {"left": "left", "center": "center", "right": "right"}
_VA_MAP = {
    "top": "top", "center": "center", "bottom": "bottom",
    "center_baseline": "center_baseline", "baseline": "baseline",
}

_LABEL_HALO_BBOX = dict(
    boxstyle="square,pad=0.10", facecolor="white", edgecolor="none",
)
_AXIS_TITLE_BBOX = dict(
    boxstyle="square,pad=0.25", facecolor="white", edgecolor="none",
)
_INSIDE_LABEL_STROKE = [
    patheffects.withStroke(linewidth=2.0, foreground="white"),
    patheffects.Normal(),
]


def _label_kwargs_from_style(style: dict) -> dict:
    """Translate core style keys into matplotlib Text kwargs."""
    out = {}
    for k, v in style.items():
        if k == "halign":
            out["ha"] = v
        elif k == "valign":
            out["va"] = v
        else:
            out[k] = v
    return out


class MatplotlibBackend(Backend):
    """Backend adapter for a matplotlib Axes.

    Holds all per-(family, role) artist pools on itself — keeps the
    wrapped Axes clean and makes teardown deterministic.
    """

    def __init__(self, ax):
        self._ax = ax
        self._collections: dict[DiagramFamily, LineCollection | None] = {
            DiagramFamily.DISPLACEMENT: None,
            DiagramFamily.ACCELERATION: None,
        }
        # {(family, role): list[Text|Annotation]}
        self._label_pools: dict[tuple[DiagramFamily, str], list] = {}
        # {family: list[Line2D]}
        self._tick_pools: dict[DiagramFamily, list] = {
            DiagramFamily.DISPLACEMENT: [],
            DiagramFamily.ACCELERATION: [],
        }
        self._style = BackendStyle()
        self._rescale_cb = None
        self._cb_ids: list[int] = []

    # ---- viewport --------------------------------------------------------

    def get_xlim(self) -> tuple[float, float]:
        return tuple(self._ax.get_xlim())

    def get_ylim(self) -> tuple[float, float]:
        return tuple(self._ax.get_ylim())

    def is_log_log(self) -> bool:
        return self._ax.get_xscale() == "log" and self._ax.get_yscale() == "log"

    def data_to_pixel(self, points):
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)
        return self._ax.transData.transform(arr)

    def measure_text_width_pt(self, text: str, style_key: str = "label") -> float:
        fig = self._ax.figure
        try:
            renderer = fig.canvas.get_renderer()
        except Exception:
            return 30.0
        # Quick throwaway Text artist — not added to axes, just measured.
        tmp = Text(0, 0, text, **_label_kwargs_from_style(self._style.label))
        tmp.set_figure(fig)
        try:
            bbox = tmp.get_window_extent(renderer=renderer)
            return bbox.width * 72.0 / fig.dpi
        except Exception:
            return 30.0

    # ---- style ----------------------------------------------------------

    def apply_style(self, style: BackendStyle) -> None:
        self._style = style

    # ---- lines ----------------------------------------------------------

    def _ensure_collection(self, family: DiagramFamily) -> LineCollection:
        coll = self._collections[family]
        if coll is None:
            coll = LineCollection([])
            coll.set_clip_on(True)
            self._ax.add_collection(coll)
            self._collections[family] = coll
        return coll

    def set_lines(self, family, segments, per_line_style=None):
        coll = self._ensure_collection(family)
        coll.set_segments([[p0, p1] for p0, p1 in segments])
        if per_line_style is None or not per_line_style:
            # Uniform style from major_line bucket
            coll.set(**_to_collection_kwargs(self._style.major_line or {}))
            return
        lws = [float(s.get("linewidth", 0.5)) for s in per_line_style]
        cols = [s.get("color", "0.45") for s in per_line_style]
        lss = [s.get("linestyle", "-") for s in per_line_style]
        coll.set_linewidths(lws)
        coll.set_colors(cols)
        try:
            coll.set_linestyle(lss)
        except Exception:
            pass

    # ---- labels ---------------------------------------------------------

    def _pool(self, family: DiagramFamily, role: str) -> list:
        key = (family, role)
        if key not in self._label_pools:
            self._label_pools[key] = []
        return self._label_pools[key]

    def _new_label(self, role: str) -> Annotation:
        style_dict = (
            self._style.axis_title if role == "axis_title" else self._style.label
        )
        ann = self._ax.annotate(
            "", xy=(0, 0), xycoords="data",
            xytext=(0, 0), textcoords="offset points",
            annotation_clip=False, **_label_kwargs_from_style(style_dict),
        )
        ann.set_rotation_mode("anchor")
        # Keep label artists out of the layout engine's margin calculation.
        # Otherwise constrained_layout re-measures our edge labels every draw
        # and the axes rectangle shrinks / grows as label count changes with
        # zoom — the exact "graph area gets smaller when I zoom" complaint.
        try:
            ann.set_in_layout(False)
        except AttributeError:
            pass
        return ann

    def _shrink(self, pool: list, keep: int) -> None:
        while len(pool) > keep:
            ann = pool.pop()
            try:
                ann.remove()
            except Exception:
                pass

    def set_labels(self, family, role, items):
        pool = self._pool(family, role)
        while len(pool) < len(items):
            pool.append(self._new_label(role))
        self._shrink(pool, len(items))

        for ann, item in zip(pool, items):
            ann.set_text(item.text)
            ann.xy = item.anchor
            ann.set_position(item.offset_pt)
            ann.set_rotation(item.rotation_deg)
            ann.set_rotation_mode("anchor")
            ann.set_ha(_HA_MAP.get(item.halign, "center"))
            ann.set_va(_VA_MAP.get(item.valign, "center"))
            if item.stroke:
                ann.set_bbox(None)
                ann.set_path_effects(list(_INSIDE_LABEL_STROKE))
            else:
                bbox = _AXIS_TITLE_BBOX if role == "axis_title" else _LABEL_HALO_BBOX
                ann.set_bbox(dict(bbox))
                ann.set_path_effects([])
            ann.set_visible(True)

    # ---- ticks ----------------------------------------------------------

    def _make_tick(self, edge: str) -> Line2D:
        if edge == "top":
            trans = blended_transform_factory(self._ax.transData, self._ax.transAxes)
            line = Line2D([0.0, 0.0], [1.0, 1.006], transform=trans,
                          color="0.15", linewidth=0.7)
        elif edge == "right":
            trans = blended_transform_factory(self._ax.transAxes, self._ax.transData)
            line = Line2D([1.0, 1.006], [0.0, 0.0], transform=trans,
                          color="0.15", linewidth=0.7)
        elif edge == "bottom":
            trans = blended_transform_factory(self._ax.transData, self._ax.transAxes)
            line = Line2D([0.0, 0.0], [0.0, -0.006], transform=trans,
                          color="0.15", linewidth=0.7)
        else:  # left
            trans = blended_transform_factory(self._ax.transAxes, self._ax.transData)
            line = Line2D([0.0, -0.006], [0.0, 0.0], transform=trans,
                          color="0.15", linewidth=0.7)
        line.set_clip_on(False)
        try:
            line.set_in_layout(False)
        except AttributeError:
            pass
        self._ax.add_artist(line)
        return line

    def set_ticks(self, family, items):
        pool = self._tick_pools[family]
        while len(pool) < len(items):
            pool.append(self._make_tick(items[len(pool)].edge if items else "top"))
        self._shrink(pool, len(items))

        for line, item in zip(pool, items):
            if item.edge == "top" or item.edge == "bottom":
                line.set_xdata([float(item.position), float(item.position)])
            else:
                line.set_ydata([float(item.position), float(item.position)])
            line.set_visible(True)

    # ---- interactivity --------------------------------------------------

    def connect_rescale(self, callback) -> None:
        self._rescale_cb = callback

        def _on_lim(_ax):
            # Fired on any xlim/ylim change — interactive zoom, pan, programmatic set_*.
            try:
                callback()
            except Exception:
                pass

        cb1 = self._ax.callbacks.connect("xlim_changed", _on_lim)
        cb2 = self._ax.callbacks.connect("ylim_changed", _on_lim)
        self._cb_ids.extend([cb1, cb2])

    def request_redraw(self) -> None:
        # Matplotlib's draw loop calls Axes.draw already; we just mark stale
        # so the change gets picked up at the next canvas.draw().
        self._ax.stale = True

    # ---- introspection --------------------------------------------------

    def describe_artists(self) -> dict:
        out = {}
        for fam, coll in self._collections.items():
            out[(fam.value, "lines")] = (
                len(coll.get_segments()) if coll is not None else 0
            )
        for (fam, role), pool in self._label_pools.items():
            out[(fam.value, role)] = sum(1 for a in pool if a.get_visible())
        for fam, pool in self._tick_pools.items():
            out[(fam.value, "tick")] = sum(1 for a in pool if a.get_visible())
        return out

    # ---- teardown -------------------------------------------------------

    def teardown(self) -> None:
        for coll in self._collections.values():
            if coll is not None:
                try:
                    coll.remove()
                except Exception:
                    pass
        for pool in self._label_pools.values():
            for a in pool:
                try:
                    a.remove()
                except Exception:
                    pass
        for pool in self._tick_pools.values():
            for a in pool:
                try:
                    a.remove()
                except Exception:
                    pass
        self._collections = {
            DiagramFamily.DISPLACEMENT: None,
            DiagramFamily.ACCELERATION: None,
        }
        self._label_pools = {}
        self._tick_pools = {
            DiagramFamily.DISPLACEMENT: [],
            DiagramFamily.ACCELERATION: [],
        }
