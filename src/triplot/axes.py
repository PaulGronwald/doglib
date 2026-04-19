"""TripartiteAxes — matplotlib Axes subclass with dynamic +/-45 degree diagonal
grids for tripartite (shock response spectrum) plots.

Performance:
    * One LineCollection per diagonal family (displacement / acceleration)
      instead of many Line2D instances — far cheaper to render.
    * A (xlim, ylim, style, bbox) cache key gates rebuild: identical view =
      zero work inside draw() beyond the signature comparison.
    * Label midpoints + rotations are computed with batched numpy transforms.

Permanence:
    * cla() + __setstate__ invalidate caches so the axes survives clear / pickle.
    * Non-log scales or non-positive / non-finite limits short-circuit cleanly
      (diagonals are hidden but no exception escapes draw()).
    * Label pool grows / shrinks in place — no artist leak under repeated zoom.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from matplotlib import patheffects
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Annotation, Text
from matplotlib.transforms import blended_transform_factory

from . import diagonals as _diag
from .units import UnitSystem, resolve as _resolve_units


def _unit_from_label(label: str) -> str:
    """Extract bracketed unit from a label like 'Displacement [in]' -> 'in'."""
    if "[" in label and "]" in label:
        return label.split("[", 1)[1].rsplit("]", 1)[0]
    return ""


def _plain_tick_formatter(x, pos):
    """Module-level tick formatter so pickle can reach it."""
    return f"{x:g}"


_DEFAULT_LINE_STYLE = dict(color="0.45", linewidth=0.55, linestyle="-", alpha=1.0)
_MAJOR_LINE_STYLE = dict(color="0.25", linewidth=0.85, linestyle="-", alpha=1.0)
_MINOR_LINE_STYLE = dict(color="0.65", linewidth=0.35, linestyle="-", alpha=1.0)
_DEFAULT_LABEL_STYLE = dict(
    color="0.15", fontsize=7, ha="center", va="center",
    family="serif",
)
_EDGE_LABEL_HALO_BBOX = dict(
    boxstyle="square,pad=0.12", facecolor="white", edgecolor="none",
)
_LABEL_HALO_BBOX = dict(
    boxstyle="square,pad=0.10", facecolor="white", edgecolor="none",
)
# Stroke effect for labels drawn OVER the diagonal grid (inside the plot):
# a thin white outline around each glyph keeps text readable over complex
# backgrounds without erasing a rectangular region the way a bbox halo does.
_INSIDE_LABEL_STROKE = [
    patheffects.withStroke(linewidth=2.0, foreground="white"),
    patheffects.Normal(),
]
_AXIS_TITLE_STYLE = dict(
    color="0.05", fontsize=10, ha="center", va="center",
    family="serif", fontstyle="italic", fontweight="bold",
)
_AXIS_TITLE_BBOX = dict(
    boxstyle="square,pad=0.25", facecolor="white", edgecolor="none",
)


def _is_label_value(v: float, label_subs: tuple[float, ...]) -> bool:
    """Return True if v == multiplier * 10^k for some integer k and multiplier
    in the label subset. Uses log10 + rounding — tolerant of tiny float drift."""
    if v <= 0:
        return False
    lv = math.log10(v)
    # Find the leading decade k and the mantissa m (1 <= m < 10).
    k = math.floor(lv + 1e-9)
    m = v / 10 ** k
    return any(abs(m - s) < 1e-6 for s in label_subs)


def _line_style_to_collection_kwargs(style: dict) -> dict:
    """Translate Line2D-flavoured kwargs to LineCollection kwargs."""
    mapping = {"color": "colors", "linewidth": "linewidths", "linestyle": "linestyles"}
    out = {}
    for k, v in style.items():
        out[mapping.get(k, k)] = v
    return out


class TripartiteAxes(Axes):
    """Log-log frequency / pseudo-velocity axes with dynamic diagonal grids."""

    name = "tripartite"

    # ------------------------------------------------------------------ init

    def __init__(
        self,
        *args,
        units=None,
        style="seismic",
        aspect="equal",
        label_mode="edge",
        show_diag_titles=None,
        **kwargs,
    ):
        self._units: UnitSystem = _resolve_units(units)
        self._diag_visible = True
        # style: 'seismic' (Frazee/Newmark-Hall), 'shock' (sparse), or 'dplot'
        # (DPlot/commercial: dense grid, 1/5 labels, no in-plot axis titles).
        if style not in ("seismic", "shock", "dplot"):
            raise ValueError("style must be 'seismic', 'shock', or 'dplot'")
        self._style = style
        # aspect: 'equal' keeps one decade of x == one decade of y (true 45°
        # diagonals). 'auto' lets the axes stretch independently in x/y so the
        # plot can be squished into a wide or tall box. Label rotation is
        # computed in display space either way, so the diagonals stay
        # visually aligned with their labels at any aspect.
        # Stored under a non-reserved name — matplotlib's Axes owns self._aspect.
        self._tri_aspect = aspect
        # label_mode: binary choice for where diagonal-isoline labels appear.
        #   'edge'     — rotated labels at each line's edge crossing (outside
        #                by default; flipped inside on overflow). Every in-view
        #                line gets labeled once; clean interior. (default)
        #   'midpoint' — labels at each line's midpoint inside the plot.
        #                Classic tripartite look; no edge labels.
        # Never both together — redundant clutter.
        if label_mode not in ("edge", "midpoint"):
            raise ValueError("label_mode must be 'edge' or 'midpoint'")
        self._label_mode = label_mode
        # show_diag_titles: the big centered 'Displacement (in)' / 'Acceleration (g)'
        # axis titles drawn across the middle of the plot. None means
        # style-dependent default (on for seismic, off otherwise).
        self._show_diag_titles = show_diag_titles
        self._diag_which = "default"
        self._diag_line_style = dict(_DEFAULT_LINE_STYLE)
        self._major_line_style = dict(_MAJOR_LINE_STYLE)
        self._minor_line_style = dict(_MINOR_LINE_STYLE)
        self._tiered_default = True
        self._diag_label_style = dict(_DEFAULT_LABEL_STYLE)
        self._damping: float | None = None

        self._cache_key = None
        self._disp_collection: LineCollection | None = None
        self._accel_collection: LineCollection | None = None
        self._disp_labels: list[Text] = []
        self._accel_labels: list[Text] = []
        self._disp_axis_title: Text | None = None
        self._accel_axis_title: Text | None = None
        self._disp_top_labels: list[Text] = []
        self._disp_top_ticks: list[Line2D] = []
        self._accel_right_labels: list[Text] = []
        self._accel_right_ticks: list[Line2D] = []
        # Fallback labels for segments that miss the designated edge. Placed
        # ON the line, rotated to match the slope, offset ALONG the line into
        # the plot — a user never sees an unlabeled in-view isoline.
        self._disp_fallback_labels: list[Annotation] = []
        self._accel_fallback_labels: list[Annotation] = []

        super().__init__(*args, **kwargs)

        self.set_xscale("log")
        self.set_yscale("log")

        x0, x1 = self.get_xlim()
        if x0 <= 0 or x1 / max(x0, 1e-12) < 20:
            self.set_xlim(1, 1000)
            self.set_ylim(0.1, 100)

        self.set_xlabel(self._units.freq_label)
        self.set_ylabel(self._units.vel_label)
        self.grid(True, which="both", linestyle="-", linewidth=0.4, color="0.8")

        # Plain-number log tick labels ("100", "0.001" not "10^2" or "0.00") —
        # FuncFormatter with %g handles both very small and very large cleanly.
        from matplotlib.ticker import FuncFormatter, LogLocator
        for axis in (self.xaxis, self.yaxis):
            axis.set_major_formatter(FuncFormatter(_plain_tick_formatter))
            axis.set_major_locator(LogLocator(base=10, numticks=12))
            axis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10), numticks=60))

        # Aspect: 'equal' in log space gives true 45° diagonals; 'auto' lets
        # the plot be squished. Label rotation is computed in display pixels
        # (not data space), so labels follow the actual on-screen slope
        # regardless of aspect.
        self.set_aspect(self._tri_aspect, adjustable="box")

    # ------------------------------------------------------------------ API

    def set_displacement_label(self, label: str) -> None:
        self._units = UnitSystem(
            self._units.name, self._units.freq_label, self._units.vel_label,
            label, self._units.accel_label, self._units.g_value, self._units.accel_in_g,
        )
        self._invalidate_cache()

    def set_acceleration_label(self, label: str) -> None:
        self._units = UnitSystem(
            self._units.name, self._units.freq_label, self._units.vel_label,
            self._units.disp_label, label, self._units.g_value, self._units.accel_in_g,
        )
        self._invalidate_cache()

    def set_label_mode(self, mode: str) -> None:
        """Switch labeling style at runtime. ``'edge'`` or ``'midpoint'``."""
        if mode not in ("edge", "midpoint"):
            raise ValueError("mode must be 'edge' or 'midpoint'")
        self._label_mode = mode
        self._invalidate_cache()

    def set_show_diag_titles(self, show: bool | None) -> None:
        """Enable / disable the big centered 'Displacement' / 'Acceleration'
        callouts across the middle of the plot. ``None`` restores the
        style-dependent default (on for seismic, off for other styles)."""
        self._show_diag_titles = show
        self._invalidate_cache()

    def set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        """Override: in edge label_mode, the top margin holds rotated disp
        labels. Push the title up with extra ``pad`` so it doesn't collide,
        and enable constrained_layout so the extra height is reserved in the
        figure grid — otherwise the title would overflow into any subplot
        above. Both effects are suppressed when the user passes ``pad=``
        explicitly or already has a layout engine active.
        """
        if pad is None and self._label_mode == "edge":
            # ~30 pt clears the outer offset (7 pt) + rotated label height.
            pad = 30.0
            # Enable constrained_layout on the root Figure if no engine is
            # active, so the extra title height is reserved in the subplot
            # grid (otherwise the title overflows into the axes above).
            root = self.figure
            while root is not None and not hasattr(root, "set_layout_engine"):
                root = getattr(root, "figure", None)  # walk up from SubFigure
            set_le = getattr(root, "set_layout_engine", None)
            get_le = getattr(root, "get_layout_engine", None)
            if set_le is not None and get_le is not None and get_le() is None:
                try:
                    set_le("constrained")
                except Exception:  # noqa: BLE001
                    pass  # non-fatal; user can set it themselves
        return super().set_title(label, fontdict=fontdict, loc=loc, pad=pad, y=y, **kwargs)

    def legend(self, *args, **kwargs):
        """Override: in edge label_mode, the top-right margin carries disp
        edge labels and the right margin carries accel edge labels, so the
        matplotlib default ``loc='best'`` often picks a cluttered corner.
        Default to ``'upper left'`` — the only uncluttered corner when edge
        labels are drawn. Users passing ``loc=...`` explicitly are unaffected.
        """
        if "loc" not in kwargs and self._label_mode == "edge":
            # Only set the default when the caller didn't also pass loc as
            # a positional string (legend(handles, labels, loc, ...)).
            has_pos_loc = any(isinstance(a, str) for a in args[:4])
            if not has_pos_loc:
                kwargs["loc"] = "upper left"
        return super().legend(*args, **kwargs)

    def grid_diagonal(self, visible: bool = True, which: str = "major", **kwargs) -> None:
        """Toggle diagonal displacement / acceleration grid.

        which : 'major', 'minor', or 'both'.
        kwargs may override color / linewidth / linestyle / alpha.
        """
        if which not in ("major", "minor", "both"):
            raise ValueError("which must be 'major', 'minor', or 'both'")
        self._diag_visible = bool(visible)
        self._diag_which = which
        if kwargs:
            self._tiered_default = False
            self._diag_line_style.update(kwargs)
        if self._disp_collection is not None:
            coll_kw = _line_style_to_collection_kwargs(self._diag_line_style)
            for coll in (self._disp_collection, self._accel_collection):
                coll.set(**coll_kw)
        self._invalidate_cache()

    def set_diag_style(
        self,
        major_linewidth: float | None = None,
        major_color: str | None = None,
        minor_linewidth: float | None = None,
        minor_color: str | None = None,
        label_fontsize: float | None = None,
        label_color: str | None = None,
    ) -> None:
        """Tune tiered diagonal grid appearance per-axes.

        Major = decade-boundary lines (mantissa == 1).
        Minor = intermediate subdivisions (mantissa 2..9).
        Keeps tiered mode active. For a uniform grid, call
        grid_diagonal(True, color=..., linewidth=...) instead.
        """
        if major_linewidth is not None:
            self._major_line_style["linewidth"] = float(major_linewidth)
        if major_color is not None:
            self._major_line_style["color"] = major_color
        if minor_linewidth is not None:
            self._minor_line_style["linewidth"] = float(minor_linewidth)
        if minor_color is not None:
            self._minor_line_style["color"] = minor_color
        if label_fontsize is not None:
            self._diag_label_style["fontsize"] = float(label_fontsize)
            for t in self._disp_labels + self._accel_labels:
                t.set_fontsize(float(label_fontsize))
        if label_color is not None:
            self._diag_label_style["color"] = label_color
            for t in self._disp_labels + self._accel_labels:
                t.set_color(label_color)
        self._tiered_default = True
        self._invalidate_cache()

    def _tier_arrays(self, segs):
        """Per-segment (linewidths, colors) for tiered rendering."""
        maj, mnr = self._major_line_style, self._minor_line_style
        lws, cols = [], []
        for s in segs:
            m = int(round(s.value / 10 ** math.floor(math.log10(s.value))))
            style = maj if m == 1 else mnr
            lws.append(style["linewidth"])
            cols.append(style["color"])
        return lws, cols

    def set_damping(self, ratio: float) -> None:
        if ratio < 0:
            raise ValueError("damping ratio must be >= 0")
        self._damping = float(ratio)

    def get_damping(self) -> float | None:
        return self._damping

    # ---- combined view accessors (useful for tests / debug) -------------

    @property
    def diag_line_count(self) -> int:
        """Number of currently visible diagonal gridlines (across both families)."""
        n = 0
        for coll in (self._disp_collection, self._accel_collection):
            if coll is not None:
                n += len(coll.get_segments())
        return n

    @property
    def diag_label_count(self) -> int:
        return (
            len(self._disp_labels) + len(self._accel_labels)
            + len(self._disp_fallback_labels) + len(self._accel_fallback_labels)
        )

    @property
    def _diag_artists(self) -> list:
        """Flat list of internal diagonal artists. Kept for test/debug use."""
        out = []
        for coll in (self._disp_collection, self._accel_collection):
            if coll is not None and len(coll.get_segments()) > 0:
                out.append(coll)
        out.extend(self._disp_labels)
        out.extend(self._accel_labels)
        out.extend(self._disp_fallback_labels)
        out.extend(self._accel_fallback_labels)
        return out

    # ------------------------------------------------------------------ core

    def _subdivisions(self) -> tuple[float, ...]:
        """Which multipliers get *gridlines* drawn within each decade."""
        if self._diag_which == "default":
            # seismic / dplot: dense grid 1..9. shock: one per decade.
            if self._style in ("seismic", "dplot"):
                return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
            return (1.0,)
        if self._diag_which == "major":
            return (1.0,)
        if self._diag_which == "minor":
            return (1.0, 2.0, 5.0)
        # 'both' / 'full'
        return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

    def _label_subdivisions(self) -> tuple[float, ...]:
        """Which multipliers get a *text label*."""
        if self._style == "seismic":
            return (1.0, 2.0, 4.0, 6.0, 8.0)
        if self._style == "dplot":
            # DPlot convention: labels only at m=1 and m=5 per decade.
            return (1.0, 5.0)
        return self._subdivisions()

    def _invalidate_cache(self) -> None:
        self._cache_key = None
        self.stale = True

    def _ensure_collections(self) -> None:
        if self._disp_collection is None:
            kw = _line_style_to_collection_kwargs(self._diag_line_style)
            self._disp_collection = LineCollection([], **kw)
            self._disp_collection.set_clip_on(True)
            self.add_collection(self._disp_collection)
        if self._accel_collection is None:
            kw = _line_style_to_collection_kwargs(self._diag_line_style)
            self._accel_collection = LineCollection([], **kw)
            self._accel_collection.set_clip_on(True)
            self.add_collection(self._accel_collection)

    def _shrink_label_pool(self, pool: list[Text], keep: int) -> None:
        while len(pool) > keep:
            txt = pool.pop()
            try:
                txt.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass

    def _update_labels(
        self,
        pool: list[Text],
        segments: list,
        unit_suffix: str = "",
    ) -> None:
        """Place numeric rotated labels (with optional unit suffix) at the
        midpoint of each segment. Density-filtered in pixel space so labels
        never crowd regardless of zoom / panel size."""
        if len(segments) == 0:
            self._shrink_label_pool(pool, 0)
            return

        ends_a = np.array([[s.f0, s.v0] for s in segments], dtype=float)
        ends_b = np.array([[s.f1, s.v1] for s in segments], dtype=float)

        trans = self.transData.transform
        pa = trans(ends_a)
        pb = trans(ends_b)
        dx = pb[:, 0] - pa[:, 0]
        dy = pb[:, 1] - pa[:, 1]
        angles = np.degrees(np.arctan2(dy, dx))
        angles = np.where(angles > 90, angles - 180, angles)
        angles = np.where(angles < -90, angles + 180, angles)

        # Midpoint in log space = geometric mean in linear space.
        f_mid = np.sqrt(ends_a[:, 0] * ends_b[:, 0])
        v_mid = np.sqrt(ends_a[:, 1] * ends_b[:, 1])
        mid_pix = trans(np.column_stack([f_mid, v_mid]))

        # Density filter: walk segments in order, keep if pixel gap to last
        # kept midpoint >= MIN_PIX. Major (m=1) labels always kept — they are
        # the decade anchors. Segments already arrive sorted by value, so
        # geometric midpoints progress monotonically along the diagonal.
        MIN_PIX = 55.0
        keep: list[int] = []
        last_xy = None
        for i, s in enumerate(segments):
            m = int(round(s.value / 10 ** math.floor(math.log10(s.value))))
            is_major = (m == 1)
            if last_xy is None or is_major:
                keep.append(i)
                last_xy = mid_pix[i]
                continue
            gap = math.hypot(mid_pix[i, 0] - last_xy[0], mid_pix[i, 1] - last_xy[1])
            if gap >= MIN_PIX:
                keep.append(i)
                last_xy = mid_pix[i]

        n = len(keep)
        while len(pool) < n:
            t = Text(0, 0, "", rotation_mode="anchor", **self._diag_label_style)
            t.set_bbox(dict(_LABEL_HALO_BBOX))
            self.add_artist(t)
            pool.append(t)
        self._shrink_label_pool(pool, n)

        suffix = f" {unit_suffix}" if unit_suffix else ""
        for t, idx in zip(pool, keep):
            s = segments[idx]
            t.set_position((float(f_mid[idx]), float(v_mid[idx])))
            t.set_text(_diag.format_value(s.value) + suffix)
            t.set_rotation(float(angles[idx]))
            t.set_visible(True)

    def _update_axis_title(
        self,
        attr: str,
        text: str,
        segments: list,
    ) -> None:
        """Place a rotated in-plot axis title ('Displacement (in)' / 'Acceleration (g)')
        along the middle visible diagonal of its family."""
        current = getattr(self, attr)
        if len(segments) < 2:
            if current is not None:
                try: current.remove()
                except Exception: pass
                setattr(self, attr, None)
            return

        # middle segment (in log-value order), near plot center
        mid_seg = segments[len(segments) // 2]
        fm = math.sqrt(mid_seg.f0 * mid_seg.f1)
        vm = math.sqrt(mid_seg.v0 * mid_seg.v1)
        p0 = self.transData.transform((mid_seg.f0, mid_seg.v0))
        p1 = self.transData.transform((mid_seg.f1, mid_seg.v1))
        angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        if angle > 90: angle -= 180
        if angle < -90: angle += 180

        if current is None:
            txt = Text(fm, vm, text, rotation=angle, rotation_mode="anchor",
                       **_AXIS_TITLE_STYLE)
            txt.set_bbox(dict(_AXIS_TITLE_BBOX))
            self.add_artist(txt)
            setattr(self, attr, txt)
        else:
            current.set_position((fm, vm))
            current.set_text(text)
            current.set_rotation(angle)
            current.set_visible(True)

    def _hide_all(self) -> None:
        if self._disp_collection is not None:
            self._disp_collection.set_segments([])
        if self._accel_collection is not None:
            self._accel_collection.set_segments([])
        self._shrink_label_pool(self._disp_labels, 0)
        self._shrink_label_pool(self._accel_labels, 0)
        self._shrink_label_pool(self._disp_top_labels, 0)
        self._shrink_label_pool(self._accel_right_labels, 0)
        self._shrink_label_pool(self._disp_top_ticks, 0)
        self._shrink_label_pool(self._accel_right_ticks, 0)
        self._shrink_label_pool(self._disp_fallback_labels, 0)
        self._shrink_label_pool(self._accel_fallback_labels, 0)
        for attr in ("_disp_axis_title", "_accel_axis_title"):
            t = getattr(self, attr, None)
            if t is not None:
                try: t.remove()
                except Exception: pass
                setattr(self, attr, None)

    def _make_edge_tick(self, side: str) -> Line2D:
        """Short tick line protruding OUTWARD from the top (or right) spine.

        Pairs with the rotated edge labels that sit outside the plot — the
        tick marks where the isoline meets the spine; the label hangs in the
        margin beyond. Length is 0.6% of the axes extent (a few pixels on
        typical figure sizes).
        """
        if side == "top":
            trans = blended_transform_factory(self.transData, self.transAxes)
            line = Line2D([0.0, 0.0], [1.0, 1.006], transform=trans,
                          color="0.15", linewidth=0.7)
        else:
            trans = blended_transform_factory(self.transAxes, self.transData)
            line = Line2D([1.0, 1.006], [0.0, 0.0], transform=trans,
                          color="0.15", linewidth=0.7)
        line.set_clip_on(False)
        self.add_artist(line)
        return line

    def _make_edge_label(self, side: str) -> Annotation:
        """Rotated numeric label placed ON the line at the edge crossing.

        Uses ``textcoords='offset points'`` so the along-line inward offset
        (set in ``_update_edge``) survives DPI / resize without recomputation.
        """
        ann = self.annotate(
            "", xy=(0, 0), xycoords="data",
            xytext=(0, 0), textcoords="offset points",
            annotation_clip=False, **self._diag_label_style,
        )
        ann.set_rotation_mode("anchor")
        ann.set_bbox(dict(_LABEL_HALO_BBOX))
        return ann

    def _update_edge(
        self,
        label_pool: list[Annotation],
        tick_pool: list[Line2D],
        segments: list,
        side: str,
        unit_suffix: str = "",
    ) -> None:
        """Place rotated ON-line labels at the designated edge crossing.

        For each segment whose line crosses the designated edge (``side``),
        anchor an Annotation at the crossing point in data coords, rotate it
        to match the line's display-space slope, and offset it ALONG the
        line by (half rendered width + 2 pt) so the text box stays inside
        the frame while overlaying the line. A short tick mark is drawn
        from the spine inward as a visual anchor.

        Segments that miss the designated edge are handled by
        :meth:`_update_edge_fallback`.

        Top (displacement): constant-d line v = 2π d f hits y = ymax at
        f = ymax / (2π d).
        Right (acceleration): constant-a line v = a / (2π f) hits x = xmax at
        v = a / (2π xmax).
        """
        TWO_PI = 2.0 * math.pi
        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()
        g = self._units.g_value if self._units.accel_in_g else 1.0

        crossings: list[tuple[object, float, float]] = []  # (segment, fe, ve)
        for s in segments:
            if s.value <= 0:
                continue
            if side == "top":
                f = ymax / (TWO_PI * s.value)
                if xmin <= f <= xmax:
                    crossings.append((s, f, ymax))
            else:
                a = s.value * g
                v = a / (TWO_PI * xmax)
                if ymin <= v <= ymax:
                    crossings.append((s, xmax, v))

        if not crossings:
            self._shrink_label_pool(label_pool, 0)
            self._shrink_label_pool(tick_pool, 0)
            return

        # Density filter along the edge in pixel space: keep all majors
        # (mantissa == 1) and drop minors that crowd within MIN_PIX of the
        # last kept entry. Same philosophy as _update_labels.
        trans = self.transData.transform
        if side == "top":
            pix = trans(np.array([[c[1], ymax] for c in crossings]))[:, 0]
        else:
            pix = trans(np.array([[xmax, c[2]] for c in crossings]))[:, 1]

        MIN_PIX = 35.0
        keep_idx: list[int] = []
        last = None
        for i, (s, _, _) in enumerate(crossings):
            m = int(round(s.value / 10 ** math.floor(math.log10(s.value))))
            is_major = (m == 1)
            if last is None or is_major or abs(pix[i] - last) >= MIN_PIX:
                keep_idx.append(i)
                last = pix[i]

        kept = [crossings[i] for i in keep_idx]
        n = len(kept)
        while len(label_pool) < n:
            label_pool.append(self._make_edge_label(side))
        while len(tick_pool) < n:
            tick_pool.append(self._make_edge_tick(side))
        self._shrink_label_pool(label_pool, n)
        self._shrink_label_pool(tick_pool, n)

        # Placement: anchor at edge crossing, then two styles —
        #   OUTSIDE (default): perpendicular offset past the tick mark, with
        #     ha chosen so the rotated text body swings outward along the
        #     line extension (not both ways from a centered anchor).
        #   INSIDE (overflow): along-line offset inward by half the rendered
        #     label width so the label sits CENTERED on the line, with its
        #     bounding box entirely clear of the spine.
        outer_offset_pt = 7.0  # past the ~4pt tick into the margin
        inner_pad_pt = 2.0     # extra safety on top of half-width

        def _log_frac(v, lo, hi):
            if v <= 0 or lo <= 0 or hi <= 0:
                return 0.5
            return (math.log10(v) - math.log10(lo)) / max(
                math.log10(hi) - math.log10(lo), 1e-30
            )

        renderer = None
        try:
            renderer = self.figure.canvas.get_renderer()
        except Exception:
            renderer = None

        edge_normals = {"top": (0.0, 1.0), "right": (1.0, 0.0)}
        ux, uy = edge_normals[side]

        suffix = f" {unit_suffix}" if unit_suffix else ""
        for ann, tick_line, (s, fe, ve) in zip(label_pool, tick_pool, kept):
            p0 = trans((s.f0, s.v0))
            p1 = trans((s.f1, s.v1))
            angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            if angle > 90: angle -= 180
            if angle < -90: angle += 180

            overflow_inside = False
            if side == "top":
                if _log_frac(fe, xmin, xmax) > 0.97:
                    overflow_inside = True
            else:
                if _log_frac(ve, ymin, ymax) < 0.03:
                    overflow_inside = True

            ar = math.radians(angle)
            lx, ly = math.cos(ar), math.sin(ar)

            ann.set_text(_diag.format_value(s.value) + suffix)
            ann.xy = (float(fe), float(ve))
            ann.set_rotation_mode("anchor")
            ann.set_rotation(float(angle))
            ann.set_va("center_baseline")

            if overflow_inside:
                # ON the line — along-line inward offset by half rendered width.
                # ha='center' keeps the label centered on the line; inward offset
                # pushes the whole bbox off the spine so it doesn't clip.
                if lx * (-ux) + ly * (-uy) >= 0:
                    along_x, along_y = lx, ly
                else:
                    along_x, along_y = -lx, -ly
                ann.set_ha("center")
                # Measure AFTER setting text/rotation for an accurate width.
                half_w_pt = 15.0
                if renderer is not None:
                    try:
                        bbox = ann.get_window_extent(renderer=renderer)
                        half_w_pt = (bbox.width * 72.0 / self.figure.dpi) / 2
                    except Exception:
                        pass
                d_pt = half_w_pt + inner_pad_pt
                ann.set_position((along_x * d_pt, along_y * d_pt))
                ann.set_bbox(None)
                # White stroke around glyphs so text stays readable over the
                # diagonal grid without a bbox rectangle erasing it.
                ann.set_path_effects(list(_INSIDE_LABEL_STROKE))
            else:
                # Outside past the tick — perpendicular offset, ha on the
                # NEAR side so the rotated text body swings outward along
                # the line (+x rotates to +outward, or -x rotates to +outward,
                # whichever matches).
                if lx * ux + ly * uy >= 0:
                    ha = "left"   # +x rotates outward
                else:
                    ha = "right"  # -x rotates outward
                ann.set_ha(ha)
                ann.set_position((ux * outer_offset_pt, uy * outer_offset_pt))
                ann.set_bbox(dict(_LABEL_HALO_BBOX))
                ann.set_path_effects([])  # no stroke needed in the white margin

            ann.set_visible(True)

            if side == "top":
                tick_line.set_xdata([float(fe), float(fe)])
            else:
                tick_line.set_ydata([float(ve), float(ve)])
            tick_line.set_visible(True)

    def _update_edge_fallback(
        self,
        label_pool: list[Annotation],
        segments: list,
        designated_edge: str,
        unit_suffix: str = "",
    ) -> None:
        """Place labels INSIDE the plot for segments that miss the designated
        edge (``'top'`` for disp, ``'right'`` for accel).

        For a disp line (slope +1): if its crossing at v=ymax falls outside
        [xmin, xmax], try the right edge instead. For an accel line (slope -1):
        if its crossing at f=xmax falls outside [ymin, ymax], try the bottom.

        Placement follows the 4cp-graph pattern: anchor at the fallback-edge
        crossing, offset perpendicular inward by a few points, ha/va set so
        rotation around the anchor sweeps the text body away from the spine.
        No halo bbox — the label sits over the grid without erasing it.
        """
        TWO_PI = 2.0 * math.pi
        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()
        g = self._units.g_value if self._units.accel_in_g else 1.0

        # (segment, f_edge, v_edge, fallback_edge)
        misses: list[tuple[object, float, float, str]] = []
        for s in segments:
            if s.value <= 0:
                continue
            if designated_edge == "top":
                f_at_top = ymax / (TWO_PI * s.value)
                if xmin <= f_at_top <= xmax:
                    continue
                v_at_right = TWO_PI * s.value * xmax
                if ymin <= v_at_right <= ymax:
                    misses.append((s, xmax, v_at_right, "right"))
            else:  # "right" — accel
                a = s.value * g
                v_at_right = a / (TWO_PI * xmax)
                if ymin <= v_at_right <= ymax:
                    continue
                f_at_bot = a / (TWO_PI * ymin)
                if xmin <= f_at_bot <= xmax:
                    misses.append((s, f_at_bot, ymin, "bottom"))

        n = len(misses)
        while len(label_pool) < n:
            ann = self.annotate(
                "", xy=(0, 0), xycoords="data",
                xytext=(0, 0), textcoords="offset points",
                annotation_clip=False, **self._diag_label_style,
            )
            ann.set_rotation_mode("anchor")
            label_pool.append(ann)
        self._shrink_label_pool(label_pool, n)
        if n == 0:
            return

        suffix = f" {unit_suffix}" if unit_suffix else ""
        trans = self.transData.transform
        edge_normals = {
            "right": (1.0, 0.0), "left": (-1.0, 0.0),
            "top": (0.0, 1.0),   "bottom": (0.0, -1.0),
        }
        renderer = None
        try:
            renderer = self.figure.canvas.get_renderer()
        except Exception:
            renderer = None
        inner_pad_pt = 2.0

        for ann, (s, fe, ve, fb_edge) in zip(label_pool, misses):
            p0 = trans((s.f0, s.v0))
            p1 = trans((s.f1, s.v1))
            angle = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
            if angle > 90: angle -= 180
            if angle < -90: angle += 180

            # Along-line inward direction (opposite of fb_edge's outward normal).
            ux, uy = edge_normals[fb_edge]
            ar = math.radians(angle)
            lx, ly = math.cos(ar), math.sin(ar)
            if lx * (-ux) + ly * (-uy) >= 0:
                along_x, along_y = lx, ly
            else:
                along_x, along_y = -lx, -ly

            ann.set_text(_diag.format_value(s.value) + suffix)
            ann.xy = (float(fe), float(ve))
            ann.set_rotation_mode("anchor")
            ann.set_rotation(float(angle))
            ann.set_ha("center")
            ann.set_va("center_baseline")
            ann.set_bbox(None)
            # White stroke around glyphs — readable over the grid without
            # erasing a rectangular background region.
            ann.set_path_effects(list(_INSIDE_LABEL_STROKE))

            half_w_pt = 15.0
            if renderer is not None:
                try:
                    bbox = ann.get_window_extent(renderer=renderer)
                    half_w_pt = (bbox.width * 72.0 / self.figure.dpi) / 2
                except Exception:
                    pass
            d_pt = half_w_pt + inner_pad_pt
            ann.set_position((along_x * d_pt, along_y * d_pt))
            ann.set_visible(True)

    def _view_is_valid(self, xlim, ylim) -> bool:
        if not (all(np.isfinite(xlim)) and all(np.isfinite(ylim))):
            return False
        if min(xlim) <= 0 or min(ylim) <= 0:
            return False
        if xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
            return False
        return True

    def _rebuild_diagonals(self) -> None:
        self._ensure_collections()

        if not self._diag_visible:
            self._hide_all()
            return

        if self.get_xscale() != "log" or self.get_yscale() != "log":
            self._hide_all()
            return

        xlim = self.get_xlim()
        ylim = self.get_ylim()
        if not self._view_is_valid(xlim, ylim):
            self._hide_all()
            return

        subs = self._subdivisions()
        label_subs = self._label_subdivisions()
        g = self._units.g_value if self._units.accel_in_g else 1.0

        disp_segs = []
        for v in _diag.pick_displacement_values(xlim, ylim, subs):
            seg = _diag.displacement_segment(v, xlim, ylim)
            if seg is not None:
                disp_segs.append(seg)

        accel_segs = []
        for v in _diag.pick_acceleration_values(xlim, ylim, g_value=g, subdivisions=subs):
            seg = _diag.acceleration_segment(v, xlim, ylim, g_value=g)
            if seg is not None:
                accel_segs.append(seg)

        self._disp_collection.set_segments(
            [[(s.f0, s.v0), (s.f1, s.v1)] for s in disp_segs]
        )
        self._accel_collection.set_segments(
            [[(s.f0, s.v0), (s.f1, s.v1)] for s in accel_segs]
        )

        if self._tiered_default:
            lws, cols = self._tier_arrays(disp_segs)
            self._disp_collection.set_linewidths(lws)
            self._disp_collection.set_colors(cols)
            lws, cols = self._tier_arrays(accel_segs)
            self._accel_collection.set_linewidths(lws)
            self._accel_collection.set_colors(cols)
        else:
            kw = _line_style_to_collection_kwargs(self._diag_line_style)
            self._disp_collection.set(**kw)
            self._accel_collection.set(**kw)

        # Sparse labels: only the subset defined by _label_subdivisions, so the
        # plot doesn't become a wall of numbers when the gridlines are dense.
        disp_label_segs = [s for s in disp_segs if _is_label_value(s.value, label_subs)]
        accel_label_segs = [s for s in accel_segs if _is_label_value(s.value, label_subs)]
        d_unit = _unit_from_label(self._units.disp_label) or ""
        a_unit = "g" if self._units.accel_in_g else _unit_from_label(self._units.accel_label)

        want_midpoint = self._label_mode == "midpoint"
        want_border = self._label_mode == "edge"

        if want_midpoint:
            self._update_labels(self._disp_labels, disp_label_segs, unit_suffix=d_unit)
            self._update_labels(self._accel_labels, accel_label_segs, unit_suffix=a_unit)
        else:
            self._shrink_label_pool(self._disp_labels, 0)
            self._shrink_label_pool(self._accel_labels, 0)

        if want_border:
            self._update_edge(
                self._disp_top_labels, self._disp_top_ticks,
                disp_label_segs, "top", unit_suffix=d_unit,
            )
            self._update_edge(
                self._accel_right_labels, self._accel_right_ticks,
                accel_label_segs, "right", unit_suffix=a_unit,
            )
            # Fallback: every label-eligible segment that missed its designated
            # edge gets a rotated label on the line inside the plot.
            self._update_edge_fallback(
                self._disp_fallback_labels, disp_label_segs, "top", unit_suffix=d_unit,
            )
            self._update_edge_fallback(
                self._accel_fallback_labels, accel_label_segs, "right", unit_suffix=a_unit,
            )
        else:
            self._shrink_label_pool(self._disp_top_labels, 0)
            self._shrink_label_pool(self._accel_right_labels, 0)
            self._shrink_label_pool(self._disp_top_ticks, 0)
            self._shrink_label_pool(self._accel_right_ticks, 0)
            self._shrink_label_pool(self._disp_fallback_labels, 0)
            self._shrink_label_pool(self._accel_fallback_labels, 0)

        # Diagonal axis titles — explicit override wins, else style default
        # (on for seismic, off for others — matches historical behavior).
        titles_on = (
            self._show_diag_titles
            if self._show_diag_titles is not None
            else (self._style == "seismic")
        )
        if titles_on:
            self._update_axis_title(
                "_disp_axis_title",
                f"Displacement ({d_unit})" if d_unit else "Displacement",
                disp_label_segs,
            )
            self._update_axis_title(
                "_accel_axis_title",
                f"Acceleration ({a_unit})" if a_unit else "Acceleration",
                accel_label_segs,
            )
        else:
            for attr in ("_disp_axis_title", "_accel_axis_title"):
                t = getattr(self, attr, None)
                if t is not None:
                    try: t.remove()
                    except Exception: pass
                    setattr(self, attr, None)

    # ------------------------------------------------------------------ cache

    def _cache_signature(self):
        try:
            bbox = tuple(self.bbox.bounds)
        except Exception:
            bbox = (0.0, 0.0, 0.0, 0.0)
        return (
            tuple(self.get_xlim()),
            tuple(self.get_ylim()),
            self.get_xscale(),
            self.get_yscale(),
            self._diag_visible,
            self._diag_which,
            self._tiered_default,
            tuple(sorted(self._diag_line_style.items())),
            tuple(sorted(self._major_line_style.items())),
            tuple(sorted(self._minor_line_style.items())),
            tuple(sorted(self._diag_label_style.items())),
            self._units.name,
            self._units.accel_in_g,
            self._units.g_value,
            self._units.disp_label,
            self._units.accel_label,
            self._label_mode,
            self._show_diag_titles,
            bbox,
        )

    # ------------------------------------------------------------------ draw / lifecycle

    def draw(self, renderer):
        self._unstale_viewLim()
        try:
            sig = self._cache_signature()
        except Exception:
            sig = None
        if sig is None or sig != self._cache_key:
            try:
                self._rebuild_diagonals()
                self._cache_key = sig
            except Exception as exc:  # noqa: BLE001  — never break draw()
                warnings.warn(f"triplot: diagonal rebuild failed: {exc!r}", stacklevel=2)
                self._hide_all()
                self._cache_key = None
        super().draw(renderer)

    def clear(self):
        # Base Axes.clear() calls self.cla() which ultimately wipes artists —
        # do that first, then reset our caches. Resetting before super() would
        # be fine too, but this ordering keeps the attribute set consistent if
        # super() raises partway through.
        result = super().clear()
        self._cache_key = None
        self._disp_collection = None
        self._accel_collection = None
        self._disp_labels = []
        self._accel_labels = []
        self._disp_top_labels = []
        self._disp_top_ticks = []
        self._accel_right_labels = []
        self._accel_right_ticks = []
        self._disp_fallback_labels = []
        self._accel_fallback_labels = []
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop live artists — they aren't picklable across all backends; rebuild
        # on next draw after unpickling.
        state["_disp_collection"] = None
        state["_accel_collection"] = None
        state["_disp_labels"] = []
        state["_accel_labels"] = []
        state["_disp_top_labels"] = []
        state["_disp_top_ticks"] = []
        state["_accel_right_labels"] = []
        state["_accel_right_ticks"] = []
        state["_disp_fallback_labels"] = []
        state["_accel_fallback_labels"] = []
        state["_cache_key"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
