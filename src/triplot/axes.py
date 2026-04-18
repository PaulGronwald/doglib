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
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.text import Text
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

    def __init__(self, *args, units=None, style="seismic", **kwargs):
        self._units: UnitSystem = _resolve_units(units)
        self._diag_visible = True
        # style: 'seismic' (Frazee/Newmark-Hall), 'shock' (sparse), or 'dplot'
        # (DPlot/commercial: dense grid, 1/5 labels, no in-plot axis titles).
        if style not in ("seismic", "shock", "dplot"):
            raise ValueError("style must be 'seismic', 'shock', or 'dplot'")
        self._style = style
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

        # Equal aspect in log space so one decade of x == one decade of y in
        # display pixels -> +45 degree diagonals really look 45 degrees.
        self.set_aspect("equal", adjustable="box")

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
        return len(self._disp_labels) + len(self._accel_labels)

    @property
    def _diag_artists(self) -> list:
        """Flat list of internal diagonal artists. Kept for test/debug use."""
        out = []
        for coll in (self._disp_collection, self._accel_collection):
            if coll is not None and len(coll.get_segments()) > 0:
                out.append(coll)
        out.extend(self._disp_labels)
        out.extend(self._accel_labels)
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
        for attr in ("_disp_axis_title", "_accel_axis_title"):
            t = getattr(self, attr, None)
            if t is not None:
                try: t.remove()
                except Exception: pass
                setattr(self, attr, None)

    def _make_edge_tick(self, side: str) -> Line2D:
        """Short tick line protruding inward from the top (or right) spine."""
        if side == "top":
            trans = blended_transform_factory(self.transData, self.transAxes)
            line = Line2D([0.0, 0.0], [1.0, 0.985], transform=trans,
                          color="0.15", linewidth=0.7)
        else:
            trans = blended_transform_factory(self.transAxes, self.transData)
            line = Line2D([1.0, 0.985], [0.0, 0.0], transform=trans,
                          color="0.15", linewidth=0.7)
        line.set_clip_on(False)
        self.add_artist(line)
        return line

    def _make_edge_label(self, side: str) -> Text:
        """Numeric label placed just inside the top (or right) spine."""
        if side == "top":
            trans = blended_transform_factory(self.transData, self.transAxes)
            t = Text(0.0, 0.975, "", transform=trans,
                     ha="center", va="top",
                     fontsize=7, color="0.15", family="serif")
        else:
            trans = blended_transform_factory(self.transAxes, self.transData)
            t = Text(0.975, 0.0, "", transform=trans,
                     ha="right", va="center",
                     fontsize=7, color="0.15", family="serif")
        t.set_bbox(dict(_EDGE_LABEL_HALO_BBOX))
        self.add_artist(t)
        return t

    def _update_edge(
        self,
        label_pool: list[Text],
        tick_pool: list[Line2D],
        segments: list,
        side: str,
        unit_suffix: str = "",
    ) -> None:
        """Place disp labels on top edge / accel labels on right edge.

        Top (displacement): constant-d line v = 2π d f hits y = ymax at
        f = ymax / (2π d).
        Right (acceleration): constant-a line v = a / (2π f) hits x = xmax at
        v = a / (2π xmax).  a = s.value * g_value when acceleration is shown in g.
        """
        TWO_PI = 2.0 * math.pi
        xmin, xmax = self.get_xlim()
        ymin, ymax = self.get_ylim()
        g = self._units.g_value if self._units.accel_in_g else 1.0

        positions: list[tuple[float, float]] = []  # (pos_along_edge, raw value)
        for s in segments:
            if side == "top":
                f = ymax / (TWO_PI * s.value)
                if xmin <= f <= xmax:
                    positions.append((f, s.value))
            else:
                a = s.value * g
                v = a / (TWO_PI * xmax)
                if ymin <= v <= ymax:
                    positions.append((v, s.value))

        if not positions:
            self._shrink_label_pool(label_pool, 0)
            self._shrink_label_pool(tick_pool, 0)
            return

        # Pixel-distance density filter along the edge.  Same philosophy as
        # diagonal-label filter: keep all majors (m == 1), drop minors that
        # crowd within MIN_PIX of the last kept label.
        trans = self.transData.transform
        if side == "top":
            pix = trans(np.array([[p, ymax] for p, _ in positions]))[:, 0]
        else:
            pix = trans(np.array([[xmax, p] for p, _ in positions]))[:, 1]

        MIN_PIX = 35.0
        keep: list[tuple[float, float]] = []
        last = None
        for i, (pos, val) in enumerate(positions):
            m = int(round(val / 10 ** math.floor(math.log10(val))))
            is_major = (m == 1)
            if last is None or is_major or abs(pix[i] - last) >= MIN_PIX:
                keep.append((pos, val))
                last = pix[i]

        n = len(keep)
        while len(label_pool) < n:
            label_pool.append(self._make_edge_label(side))
        while len(tick_pool) < n:
            tick_pool.append(self._make_edge_tick(side))
        self._shrink_label_pool(label_pool, n)
        self._shrink_label_pool(tick_pool, n)

        suffix = f" {unit_suffix}" if unit_suffix else ""
        for t, line, (pos, val) in zip(label_pool, tick_pool, keep):
            text = _diag.format_value(val) + suffix
            if side == "top":
                t.set_position((float(pos), 0.975))
                line.set_xdata([float(pos), float(pos)])
            else:
                t.set_position((0.975, float(pos)))
                line.set_ydata([float(pos), float(pos)])
            t.set_text(text)
            t.set_visible(True)
            line.set_visible(True)

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
        self._update_labels(self._disp_labels, disp_label_segs, unit_suffix=d_unit)
        self._update_labels(self._accel_labels, accel_label_segs, unit_suffix=a_unit)
        self._update_edge(
            self._disp_top_labels, self._disp_top_ticks,
            disp_label_segs, "top", unit_suffix=d_unit,
        )
        self._update_edge(
            self._accel_right_labels, self._accel_right_ticks,
            accel_label_segs, "right", unit_suffix=a_unit,
        )

        # Diagonal axis titles (seismic style only — dplot identifies family via label unit).
        if self._style == "seismic":
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
        state["_cache_key"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
