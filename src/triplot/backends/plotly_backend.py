"""Plotly backend for the tripartite core.

Renders diagonal gridlines as ``layout.shapes`` (line shapes, cheap to
update) and edge / midpoint / fallback labels as ``layout.annotations``
(which natively support ``textangle``). Edge ticks use short line shapes
anchored with ``xref='x' / yref='paper'`` (or vice versa) so they sit
flush against the spine without tracking data coords for the outside end.

Interactivity: when the backend is attached to a
:class:`plotly.graph_objects.FigureWidget`, we connect an ``on_relayout``
callback that reads the new axis ranges from the layout event and
invokes the rescale callback. Plain ``Figure`` objects get a static
render (they're used for offline HTML export); the core rebuilds once at
plot time and produces a snapshot that matches whatever range / aspect
is in effect.

Why shapes + annotations instead of scatter traces? Traces would be
fine for straight lines, but every single isoline would become its own
trace — plotly builds a legend entry and hover target per trace, and
that balloons the figure size and clutters the UI. Shapes stay in the
layout and render as cheap decorations that don't pollute the legend.
"""
from __future__ import annotations

import math

import numpy as np

from .base import Backend, BackendStyle, DiagramFamily, LabelItem, TickItem


def _to_css_color(c):
    """Accept matplotlib-style greyscale strings ('0.45'), RGB tuples,
    and already-valid CSS strings; return something plotly will accept."""
    if c is None:
        return "#666666"
    if isinstance(c, (tuple, list)):
        if len(c) == 3:
            r, g, b = c
            return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"
        if len(c) == 4:
            r, g, b, a = c
            return f"rgba({int(r*255)},{int(g*255)},{int(b*255)},{a})"
    if isinstance(c, str):
        # Matplotlib greyscale fraction like '0.45' or '0.65' — convert to
        # hex. Plotly rejects bare numeric strings.
        try:
            f = float(c)
            if 0.0 <= f <= 1.0:
                v = int(round(f * 255))
                return f"rgb({v},{v},{v})"
        except ValueError:
            pass
        return c
    return str(c)


_HA_MAP = {"left": "left", "center": "center", "right": "right"}
_VA_MAP = {
    "top": "top", "center": "middle", "bottom": "bottom",
    "center_baseline": "middle", "baseline": "middle",
}

# Used to tag shapes / annotations so this backend can find-and-replace its
# own artists without touching user-added shapes. Plotly's shape/annotation
# lists are flat, so a naming convention is the cleanest way.
_NAME_PREFIX = "triplot:"


def _shape_name(family: DiagramFamily, index: int, kind: str = "line") -> str:
    return f"{_NAME_PREFIX}{family.value}:{kind}:{index}"


def _anno_name(family: DiagramFamily, role: str, index: int) -> str:
    return f"{_NAME_PREFIX}{family.value}:{role}:{index}"


class PlotlyBackend(Backend):
    """Backend adapter for a plotly ``go.Figure`` or ``go.FigureWidget``.

    The wrapped figure owns its own layout; this backend mutates
    ``fig.layout.shapes`` and ``fig.layout.annotations`` in-place. Tagged
    names keep the triplot-owned items identifiable on subsequent
    rebuilds, so user-added shapes / annotations are left alone.
    """

    def __init__(self, fig, *, log_x: bool = True, log_y: bool = True):
        self._fig = fig
        self._log_x = log_x
        self._log_y = log_y
        self._style = BackendStyle()
        self._rescale_cb = None
        # Per-family shape/annotation index tracking — tells us which slots
        # in fig.layout.shapes / annotations are triplot's so updates
        # overwrite without duplicating.
        self._owned_shape_ids: set[str] = set()
        self._owned_anno_ids: set[str] = set()

        self._configure_axes()

    def _configure_axes(self) -> None:
        # Log axes + square aspect — matches the matplotlib projection.
        self._fig.update_xaxes(type="log" if self._log_x else "linear")
        self._fig.update_yaxes(type="log" if self._log_y else "linear")
        # Locked aspect ratio in log space for the "equal" look. The
        # plotly equivalent of matplotlib's aspect='equal' on log-log is
        # scaleanchor + scaleratio on the y axis.
        self._fig.update_yaxes(scaleanchor="x", scaleratio=1.0)

    # ---- viewport --------------------------------------------------------

    def _axis_range(self, axis: str) -> tuple[float, float]:
        layout = self._fig.layout
        ax = getattr(layout, axis)
        rng = getattr(ax, "range", None)
        if rng is None or rng[0] is None or rng[1] is None:
            # Ranges on log axes are stored as log10 values in plotly.
            # autorange=True means we haven't been told yet; fall back to
            # a sensible default matching the matplotlib default window.
            if axis == "xaxis":
                return (1.0, 1000.0)
            return (0.1, 100.0)
        lo, hi = rng
        if ax.type == "log":
            return (10.0 ** lo, 10.0 ** hi)
        return (float(lo), float(hi))

    def get_xlim(self) -> tuple[float, float]:
        return self._axis_range("xaxis")

    def get_ylim(self) -> tuple[float, float]:
        return self._axis_range("yaxis")

    def is_log_log(self) -> bool:
        return (
            self._fig.layout.xaxis.type == "log"
            and self._fig.layout.yaxis.type == "log"
        )

    # ---- transforms ------------------------------------------------------

    def data_to_pixel(self, points):
        """Approximate data->pixel transform using current axis ranges and
        the figure's pixel size. Plotly doesn't expose a canonical
        transData like matplotlib does, so we reconstruct it from layout
        dimensions and domain fractions. Good enough for rotation + density
        math — the core only uses pixel distances for tie-breaking, not
        for absolute placement.
        """
        arr = np.asarray(points, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, 2)

        width = float(getattr(self._fig.layout, "width", None) or 700.0)
        height = float(getattr(self._fig.layout, "height", None) or 500.0)

        xlim = self.get_xlim()
        ylim = self.get_ylim()
        log_x = self.is_log_log()  # both or neither
        log_y = log_x

        if log_x:
            lx0, lx1 = math.log10(xlim[0]), math.log10(xlim[1])
            x_frac = (np.log10(arr[:, 0]) - lx0) / max(lx1 - lx0, 1e-30)
        else:
            x_frac = (arr[:, 0] - xlim[0]) / max(xlim[1] - xlim[0], 1e-30)
        if log_y:
            ly0, ly1 = math.log10(ylim[0]), math.log10(ylim[1])
            y_frac = (np.log10(arr[:, 1]) - ly0) / max(ly1 - ly0, 1e-30)
        else:
            y_frac = (arr[:, 1] - ylim[0]) / max(ylim[1] - ylim[0], 1e-30)

        # Pixel y grows downward in plotly convention; match matplotlib
        # (upward) since the core treats them symmetrically via arctan2.
        px = x_frac * width
        py = y_frac * height
        return np.column_stack([px, py])

    def measure_text_width_pt(self, text: str, style_key: str = "label") -> float:
        """Plotly can't measure offline. Return a reasonable estimate
        from font size and character count — good enough for along-line
        offset math (the core only needs half-width to keep the bbox
        centered)."""
        style = self._style.label if style_key == "label" else self._style.axis_title
        size = float(style.get("fontsize", 7))
        # 0.55 em per char is a rough sans-serif average in points.
        return len(text) * size * 0.55

    # ---- style ----------------------------------------------------------

    def apply_style(self, style: BackendStyle) -> None:
        self._style = style

    # ---- shape / annotation bookkeeping ---------------------------------

    def _prune_owned(self) -> None:
        """Remove only triplot-owned shapes + annotations from the figure
        layout; user-added decorations are preserved."""
        shapes = tuple(self._fig.layout.shapes or ())
        self._fig.layout.shapes = tuple(
            s for s in shapes if not (s.name and s.name.startswith(_NAME_PREFIX))
        )
        annos = tuple(self._fig.layout.annotations or ())
        self._fig.layout.annotations = tuple(
            a for a in annos if not (a.name and a.name.startswith(_NAME_PREFIX))
        )
        self._owned_shape_ids.clear()
        self._owned_anno_ids.clear()

    def _add_shape(self, shape_dict: dict) -> None:
        self._fig.add_shape(**shape_dict)
        self._owned_shape_ids.add(shape_dict.get("name", ""))

    def _add_annotation(self, anno_dict: dict) -> None:
        self._fig.add_annotation(**anno_dict)
        self._owned_anno_ids.add(anno_dict.get("name", ""))

    # ---- drawing --------------------------------------------------------

    def _reset_family_lines(self, family: DiagramFamily) -> None:
        prefix = f"{_NAME_PREFIX}{family.value}:line:"
        shapes = tuple(self._fig.layout.shapes or ())
        self._fig.layout.shapes = tuple(
            s for s in shapes if not (s.name and s.name.startswith(prefix))
        )

    def _reset_family_ticks(self, family: DiagramFamily) -> None:
        prefix = f"{_NAME_PREFIX}{family.value}:tick:"
        shapes = tuple(self._fig.layout.shapes or ())
        self._fig.layout.shapes = tuple(
            s for s in shapes if not (s.name and s.name.startswith(prefix))
        )

    def _reset_family_role(self, family: DiagramFamily, role: str) -> None:
        prefix = f"{_NAME_PREFIX}{family.value}:{role}:"
        annos = tuple(self._fig.layout.annotations or ())
        self._fig.layout.annotations = tuple(
            a for a in annos if not (a.name and a.name.startswith(prefix))
        )

    def set_lines(self, family, segments, per_line_style=None):
        self._reset_family_lines(family)
        for i, (p0, p1) in enumerate(segments):
            style = (
                (per_line_style[i] if per_line_style else {})
                or self._style.major_line
                or {}
            )
            color = _to_css_color(style.get("color", "#666666"))
            lw = float(style.get("linewidth", 0.6))
            dash_map = {"-": "solid", "--": "dash", ":": "dot", "-.": "dashdot"}
            dash = dash_map.get(style.get("linestyle", "-"), "solid")
            self._add_shape(dict(
                type="line",
                xref="x", yref="y",
                x0=p0[0], y0=p0[1], x1=p1[0], y1=p1[1],
                line=dict(color=color, width=lw, dash=dash),
                layer="below",
                name=_shape_name(family, i, "line"),
            ))

    def set_labels(self, family, role, items):
        self._reset_family_role(family, role)
        style = (
            self._style.axis_title if role == "axis_title" else self._style.label
        )
        for i, item in enumerate(items):
            font = dict(
                size=float(style.get("fontsize", 7)),
                color=_to_css_color(style.get("color", "#222")),
                family=style.get("family", "serif"),
            )
            # Offset in points — plotly annotations natively support
            # pixel offsets via ``xshift`` / ``yshift`` (points = px at
            # 72 DPI which is plotly's default).
            xshift, yshift = item.offset_pt
            ha = _HA_MAP.get(item.halign, "center")
            va = _VA_MAP.get(item.valign, "middle")
            bgcolor = "rgba(255,255,255,0.8)" if not item.stroke else "rgba(0,0,0,0)"
            bordercolor = "rgba(0,0,0,0)"

            self._add_annotation(dict(
                name=_anno_name(family, role, i),
                x=item.anchor[0], y=item.anchor[1],
                xref="x", yref="y",
                text=item.text,
                showarrow=False,
                textangle=-float(item.rotation_deg),  # plotly rotates CW
                font=font,
                xanchor=ha,
                yanchor=va,
                xshift=xshift,
                yshift=yshift,
                bgcolor=bgcolor,
                bordercolor=bordercolor,
                borderpad=2,
            ))

    def set_ticks(self, family, items):
        self._reset_family_ticks(family)
        for i, item in enumerate(items):
            length = item.length_axes  # axis fraction
            if item.edge == "top":
                self._add_shape(dict(
                    type="line",
                    xref="x", yref="paper",
                    x0=item.position, y0=1.0,
                    x1=item.position, y1=1.0 + length,
                    line=dict(color="#222", width=0.8),
                    name=_shape_name(family, i, "tick"),
                ))
            elif item.edge == "right":
                self._add_shape(dict(
                    type="line",
                    xref="paper", yref="y",
                    x0=1.0, y0=item.position,
                    x1=1.0 + length, y1=item.position,
                    line=dict(color="#222", width=0.8),
                    name=_shape_name(family, i, "tick"),
                ))
            elif item.edge == "bottom":
                self._add_shape(dict(
                    type="line",
                    xref="x", yref="paper",
                    x0=item.position, y0=0.0,
                    x1=item.position, y1=-length,
                    line=dict(color="#222", width=0.8),
                    name=_shape_name(family, i, "tick"),
                ))
            else:  # left
                self._add_shape(dict(
                    type="line",
                    xref="paper", yref="y",
                    x0=0.0, y0=item.position,
                    x1=-length, y1=item.position,
                    line=dict(color="#222", width=0.8),
                    name=_shape_name(family, i, "tick"),
                ))

    # ---- interactivity --------------------------------------------------

    def connect_rescale(self, callback) -> None:
        """Wire a relayout listener — fires when the user zooms, pans, or
        resizes. Only works for FigureWidget (which has on_relayout); plain
        Figures render once and ignore rescale."""
        self._rescale_cb = callback
        on_relayout = getattr(self._fig, "on_relayout", None) or getattr(
            self._fig, "observe", None
        )
        if on_relayout is None:
            return

        def _handler(changes, *args, **kwargs):
            try:
                callback()
            except Exception:
                pass

        try:
            self._fig.on_relayout(_handler)  # FigureWidget
        except Exception:
            # Not a FigureWidget — silently no-op. User's rebuild on demand
            # via manual ``triplot_rebuild(fig)`` still works.
            pass

    def request_redraw(self) -> None:
        # FigureWidget updates push automatically via traitlets. Plain
        # Figures re-render on next show/to_html. Nothing to do here.
        pass

    # ---- introspection --------------------------------------------------

    def describe_artists(self) -> dict:
        shapes = tuple(self._fig.layout.shapes or ())
        annos = tuple(self._fig.layout.annotations or ())
        out: dict = {}
        for s in shapes:
            if not (s.name and s.name.startswith(_NAME_PREFIX)):
                continue
            # name format: triplot:<family>:<kind>:<idx>
            parts = s.name[len(_NAME_PREFIX):].split(":")
            if len(parts) >= 2:
                key = (parts[0], parts[1])
                out[key] = out.get(key, 0) + 1
        for a in annos:
            if not (a.name and a.name.startswith(_NAME_PREFIX)):
                continue
            parts = a.name[len(_NAME_PREFIX):].split(":")
            if len(parts) >= 2:
                key = (parts[0], parts[1])
                out[key] = out.get(key, 0) + 1
        return out
