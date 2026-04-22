"""Backend-agnostic coordinator for tripartite plots.

``TripartiteCore`` owns the *semantic* state of a tripartite plot — units,
style, label_mode, damping, diagonal visibility — and, given a
:class:`triplot.backends.base.Backend`, drives a rebuild pass that picks
nice values, clips segments, selects labels, filters density in pixel
space, and emits backend draw calls. All maths lives here; nothing
imports matplotlib or plotly.

Why extract this from ``TripartiteAxes``? Originally the Axes subclass
computed segments and called LineCollection / Annotation / Line2D
methods inline. Porting to a second backend would have meant duplicating
hundreds of lines of math. Splitting the semantic / math layer from the
rendering layer lets both backends share the same tick-picking,
density-filtering, and overflow-handling logic, and guarantees the two
stay visually identical.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from . import diagonals as _diag
from . import ticks as _ticks
from .backends.base import (
    Backend,
    BackendStyle,
    DiagramFamily,
    LabelItem,
    TickItem,
)
from .units import UnitSystem, resolve as _resolve_units


# Default style dicts — mirror what the matplotlib implementation had baked in.
# Backends translate keys as they see fit.
DEFAULT_LINE_STYLE = dict(color="0.45", linewidth=0.55, linestyle="-", alpha=1.0)
DEFAULT_MAJOR_LINE_STYLE = dict(color="0.25", linewidth=0.85, linestyle="-", alpha=1.0)
DEFAULT_MINOR_LINE_STYLE = dict(color="0.65", linewidth=0.35, linestyle="-", alpha=1.0)
DEFAULT_LABEL_STYLE = dict(
    color="0.15", fontsize=7, halign="center", valign="center", family="serif",
)
DEFAULT_AXIS_TITLE_STYLE = dict(
    color="0.05", fontsize=10, halign="center", valign="center",
    family="serif", fontstyle="italic", fontweight="bold",
)

# Minimum on-screen spacing between kept labels in pixel units. Values were
# chosen visually on default figure sizes — tuned on matplotlib; plotly
# renders at the same DPI unit so the same numbers work.
MIN_PIX_MIDPOINT = 55.0
MIN_PIX_EDGE = 35.0

# "Within ~3% of the spine in log coords" = treat edge label as overflow
# and flip it onto the line inside. Same threshold on both backends.
OVERFLOW_LOG_FRAC = 0.03


def _is_label_value(v: float, label_subs: tuple[float, ...]) -> bool:
    """Return True if v == m * 10^k for some integer k and m in
    ``label_subs`` (within floating-point tolerance)."""
    if v <= 0:
        return False
    lv = math.log10(v)
    k = math.floor(lv + 1e-9)
    m = v / 10 ** k
    return any(abs(m - s) < 1e-6 for s in label_subs)


def _mantissa_int(v: float) -> int:
    return int(round(v / 10 ** math.floor(math.log10(v))))


def _unit_from_label(label: str) -> str:
    if "[" in label and "]" in label:
        return label.split("[", 1)[1].rsplit("]", 1)[0]
    return ""


@dataclass
class TripartiteCore:
    """Backend-agnostic state + rebuild logic for a tripartite plot.

    The core is cheap to construct, holds no artists, and can be attached
    to any backend at any time. Every rebuild recomputes from the backend's
    current viewport — so a zoom / resize callback simply invokes
    :meth:`rebuild`.
    """

    units: UnitSystem = field(default_factory=lambda: _resolve_units("imperial"))
    style: str = "seismic"
    label_mode: str = "edge"
    show_diag_titles: object = None  # None = style-dependent default
    diag_visible: bool = True
    diag_which: str = "default"
    tiered_default: bool = True

    line_style: dict = field(default_factory=lambda: dict(DEFAULT_LINE_STYLE))
    major_style: dict = field(default_factory=lambda: dict(DEFAULT_MAJOR_LINE_STYLE))
    minor_style: dict = field(default_factory=lambda: dict(DEFAULT_MINOR_LINE_STYLE))
    label_style: dict = field(default_factory=lambda: dict(DEFAULT_LABEL_STYLE))
    axis_title_style: dict = field(default_factory=lambda: dict(DEFAULT_AXIS_TITLE_STYLE))

    damping: float | None = None

    # Populated by rebuild for backends / tests to introspect.
    last_disp_segments: list = field(default_factory=list, repr=False)
    last_accel_segments: list = field(default_factory=list, repr=False)

    # ---- validation helpers ---------------------------------------------

    def _check_label_mode(self, mode: str) -> None:
        if mode not in ("edge", "midpoint"):
            raise ValueError("label_mode must be 'edge' or 'midpoint'")

    def _check_style(self, style: str) -> None:
        if style not in ("seismic", "shock", "dplot"):
            raise ValueError("style must be 'seismic', 'shock', or 'dplot'")

    def set_label_mode(self, mode: str) -> None:
        self._check_label_mode(mode)
        self.label_mode = mode

    def set_style(self, style: str) -> None:
        self._check_style(style)
        self.style = style

    # ---- style subsets --------------------------------------------------

    def subdivisions(self, span_decades: float | None = None) -> tuple[float, ...]:
        """Mantissas that get gridlines within each (sampled) decade.

        Paired with :meth:`decade_step` to drive the full adaptive
        gridline policy. This method picks *density within a decade*;
        ``decade_step`` picks *which decades to sample* when the span is
        large enough that even 1-per-decade would crowd the plot.

        Thresholds (per decade span) are tuned so visual density on a
        ~700-1000px axes stays roughly constant regardless of zoom:

        | span           | mantissa ladder           | ticks/decade |
        | -------------- | ------------------------- | ------------ |
        | <= 2           | ``{1..9}``                | 9            |
        | <= 4           | ``{1, 2, 3, 5, 7}``       | 5            |
        | <= 8           | ``{1, 2, 5}``             | 3            |
        | <= 15          | ``{1, 3}``                | 2            |
        | > 15           | ``{1}``                   | 1            |

        Above ~20-decade spans :meth:`decade_step` kicks in and starts
        skipping decades themselves, so a 100-decade viewport doesn't
        emit 100 "1-per-decade" ticks.

        User overrides via ``grid_diagonal(which=...)`` are always
        respected verbatim — we only adapt the default tier.
        """
        if self.diag_which != "default":
            if self.diag_which == "major":
                return (1.0,)
            if self.diag_which == "minor":
                return (1.0, 2.0, 5.0)
            return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

        if self.style == "shock":
            # shock style is always sparse regardless of zoom
            return (1.0,)

        if span_decades is None:
            return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

        # Span is measured on the *effective sampled* decade count — i.e.
        # span / decade_step. Callers should pass (raw_span / step) so
        # the mantissa ladder maps to the number of decades that will
        # actually be visited, not the raw viewport width. The core does
        # this coupling in rebuild().
        if span_decades <= 2.0:
            return (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        if span_decades <= 4.0:
            return (1.0, 2.0, 3.0, 5.0, 7.0)
        if span_decades <= 8.0:
            return (1.0, 2.0, 5.0)
        if span_decades <= 15.0:
            return (1.0, 3.0)
        return (1.0,)

    def decade_step(self, span_decades: float) -> int:
        """Sample every Nth decade when the viewport is wide enough that
        one-per-decade would crowd the plot. Combines with
        :meth:`subdivisions` to cap visual tick density at ~15-20 per
        axis regardless of how far the user zooms out.

        | raw span (decades) | decade_step | ticks-per-axis target  |
        | ------------------ | ----------- | ---------------------- |
        | <= 20              | 1           | up to ~20              |
        | <= 40              | 2           | ~10-20                 |
        | <= 100             | 5           | ~8-20                  |
        | <= 200             | 10          | ~10-20                 |
        | > 200              | 25          | bounded                |

        The 25 upper bound is the "we'll handle 10^300 some other day"
        escape hatch — past this the viewport is floating-point-sketchy
        anyway, but we at least don't render hundreds of lines.
        """
        if span_decades <= 20.0:
            return 1
        if span_decades <= 40.0:
            return 2
        if span_decades <= 100.0:
            return 5
        if span_decades <= 200.0:
            return 10
        return 25

    def label_subdivisions(self) -> tuple[float, ...]:
        """Which mantissas get numeric labels."""
        if self.style == "seismic":
            return (1.0, 2.0, 4.0, 6.0, 8.0)
        if self.style == "dplot":
            return (1.0, 5.0)
        return self.subdivisions()

    def _g_value(self) -> float:
        return self.units.g_value if self.units.accel_in_g else 1.0

    def _unit_suffix(self, family: DiagramFamily) -> str:
        if family is DiagramFamily.DISPLACEMENT:
            return _unit_from_label(self.units.disp_label) or ""
        return "g" if self.units.accel_in_g else _unit_from_label(self.units.accel_label)

    # ---- rebuild --------------------------------------------------------

    def build_backend_style(self) -> BackendStyle:
        return BackendStyle(
            major_line=dict(self.major_style),
            minor_line=dict(self.minor_style),
            label=dict(self.label_style),
            axis_title=dict(self.axis_title_style),
        )

    def rebuild(self, backend: Backend) -> None:
        """Re-derive every diagonal, label, and tick from the backend's
        current viewport and push the results via backend.set_*."""
        backend.apply_style(self.build_backend_style())

        if not self.diag_visible or not backend.is_log_log():
            self._emit_empty(backend)
            return

        xlim = backend.get_xlim()
        ylim = backend.get_ylim()
        if not self._viewport_valid(xlim, ylim):
            self._emit_empty(backend)
            return

        g = self._g_value()

        # Split each family into majors + minors using the same policy as
        # the X/Y axis gridlines (see ticks.major_minor_split). Majors are
        # the anchors — always ``10^k`` values, bounded to ~5-8 visible
        # across any viewport via nice decade steps. Minors fill the
        # space between majors logically in log space.
        #
        # Only majors carry edge labels / midpoint labels / ticks —
        # minors are decoration only, matching the X/Y axis convention.
        d_lo, d_hi = _diag.displacement_value_range(xlim, ylim)
        a_lo, a_hi = _diag.acceleration_value_range(xlim, ylim)

        disp_maj_vals, disp_min_vals = _ticks.major_minor_split(d_lo, d_hi)
        accel_maj_vals, accel_min_vals = _ticks.major_minor_split(
            a_lo / g, a_hi / g,
        )

        # Guarantee at least 2 majors even in ultra-narrow zooms — fall
        # back to the progressive mantissa picker when the decade-only
        # split is insufficient (e.g. sub-decade viewports).
        if len(disp_maj_vals) < 2:
            disp_maj_vals = _diag.pick_displacement_values(
                xlim, ylim, min_count=2, include_overflow=False,
            )
            disp_min_vals = []
        if len(accel_maj_vals) < 2:
            accel_maj_vals = _diag.pick_acceleration_values(
                xlim, ylim, g_value=g, min_count=2, include_overflow=False,
            )
            accel_min_vals = []

        disp_maj_segs = _clip_values(disp_maj_vals, xlim, ylim, "disp", g)
        disp_min_segs = _clip_values(disp_min_vals, xlim, ylim, "disp", g)
        accel_maj_segs = _clip_values(accel_maj_vals, xlim, ylim, "accel", g)
        accel_min_segs = _clip_values(accel_min_vals, xlim, ylim, "accel", g)

        # Union for legacy introspection (``last_*_segments``, tests).
        disp_segs = disp_maj_segs + disp_min_segs
        accel_segs = accel_maj_segs + accel_min_segs
        self.last_disp_segments = disp_segs
        self.last_accel_segments = accel_segs

        # Render: majors + minors as one per-family collection, styled
        # per segment so the backend can draw them in one pass.
        self._emit_lines_tiered(
            backend, DiagramFamily.DISPLACEMENT, disp_maj_segs, disp_min_segs,
        )
        self._emit_lines_tiered(
            backend, DiagramFamily.ACCELERATION, accel_maj_segs, accel_min_segs,
        )

        # Only majors get labels — the matching convention with the X/Y
        # axis gridlines, and what the user expects: minors are logical
        # in-between decorations with no numeric identity.
        disp_label_segs = disp_maj_segs
        accel_label_segs = accel_maj_segs

        want_mid = self.label_mode == "midpoint"
        want_edge = self.label_mode == "edge"

        if want_mid:
            self._emit_midpoint_labels(backend, DiagramFamily.DISPLACEMENT, disp_label_segs)
            self._emit_midpoint_labels(backend, DiagramFamily.ACCELERATION, accel_label_segs)
        else:
            backend.set_labels(DiagramFamily.DISPLACEMENT, "midpoint", [])
            backend.set_labels(DiagramFamily.ACCELERATION, "midpoint", [])

        if want_edge:
            self._emit_edge(backend, DiagramFamily.DISPLACEMENT, disp_label_segs,
                            side="top", xlim=xlim, ylim=ylim, g=g)
            self._emit_edge(backend, DiagramFamily.ACCELERATION, accel_label_segs,
                            side="right", xlim=xlim, ylim=ylim, g=g)
        else:
            for fam in (DiagramFamily.DISPLACEMENT, DiagramFamily.ACCELERATION):
                backend.set_labels(fam, "edge", [])
                backend.set_labels(fam, "fallback", [])
                backend.set_ticks(fam, [])

        titles_on = (
            self.show_diag_titles
            if self.show_diag_titles is not None
            else (self.style == "seismic")
        )
        if titles_on:
            self._emit_axis_title(backend, DiagramFamily.DISPLACEMENT, disp_label_segs)
            self._emit_axis_title(backend, DiagramFamily.ACCELERATION, accel_label_segs)
        else:
            backend.set_labels(DiagramFamily.DISPLACEMENT, "axis_title", [])
            backend.set_labels(DiagramFamily.ACCELERATION, "axis_title", [])

        backend.request_redraw()

    # ---- emit helpers ---------------------------------------------------

    def _emit_empty(self, backend: Backend) -> None:
        for fam in (DiagramFamily.DISPLACEMENT, DiagramFamily.ACCELERATION):
            backend.set_lines(fam, [], None)
            for role in ("midpoint", "edge", "fallback", "axis_title"):
                backend.set_labels(fam, role, [])
            backend.set_ticks(fam, [])
        backend.request_redraw()

    def _emit_lines_tiered(
        self,
        backend: Backend,
        family: DiagramFamily,
        majors: list,
        minors: list,
    ) -> None:
        """Render majors + minors as one tiered collection.

        Majors styled with :attr:`major_style`, minors with
        :attr:`minor_style`. Minors come first in the emitted list so
        majors render on top — matters only at exact crossings, which
        with our picker is every minor's two neighbours.

        When the user has overridden the grid style (``tiered_default``
        is False), everything is drawn at :attr:`line_style`.
        """
        segs = minors + majors
        endpoints = [((s.f0, s.v0), (s.f1, s.v1)) for s in segs]
        if self.tiered_default:
            per_line = [dict(self.minor_style) for _ in minors] + [
                dict(self.major_style) for _ in majors
            ]
            backend.set_lines(family, endpoints, per_line)
        else:
            backend.set_lines(family, endpoints, None)

    def _pixel_angles(self, backend: Backend, segs: list) -> np.ndarray:
        if not segs:
            return np.zeros(0)
        a = np.array([[s.f0, s.v0] for s in segs], dtype=float)
        b = np.array([[s.f1, s.v1] for s in segs], dtype=float)
        pa = backend.data_to_pixel(a)
        pb = backend.data_to_pixel(b)
        dx = pb[:, 0] - pa[:, 0]
        dy = pb[:, 1] - pa[:, 1]
        ang = np.degrees(np.arctan2(dy, dx))
        ang = np.where(ang > 90, ang - 180, ang)
        ang = np.where(ang < -90, ang + 180, ang)
        return ang

    def _emit_midpoint_labels(
        self, backend: Backend, family: DiagramFamily, segs: list,
    ) -> None:
        if not segs:
            backend.set_labels(family, "midpoint", [])
            return

        angles = self._pixel_angles(backend, segs)

        f_mid = np.array([math.sqrt(s.f0 * s.f1) for s in segs])
        v_mid = np.array([math.sqrt(s.v0 * s.v1) for s in segs])
        mid_pix = backend.data_to_pixel(np.column_stack([f_mid, v_mid]))

        keep = self._density_filter(segs, mid_pix, MIN_PIX_MIDPOINT)

        unit_suffix = self._unit_suffix(family)
        suffix = f" {unit_suffix}" if unit_suffix else ""

        items: list[LabelItem] = []
        for i in keep:
            items.append(
                LabelItem(
                    text=_diag.format_value(segs[i].value) + suffix,
                    anchor=(float(f_mid[i]), float(v_mid[i])),
                    rotation_deg=float(angles[i]),
                    offset_pt=(0.0, 0.0),
                    halign="center",
                    valign="center",
                    stroke=False,
                    style_key="label",
                )
            )
        backend.set_labels(family, "midpoint", items)

    def _density_filter(self, segs: list, pix: np.ndarray, min_pix: float) -> list[int]:
        """Walk segs in order; keep if pixel-gap to last kept >= min_pix.
        Major (m==1) always kept so decade anchors never drop."""
        keep: list[int] = []
        last = None
        for i, s in enumerate(segs):
            is_major = _mantissa_int(s.value) == 1
            xy = pix[i]
            if last is None or is_major:
                keep.append(i)
                last = xy
                continue
            gap = math.hypot(float(xy[0] - last[0]), float(xy[1] - last[1]))
            if gap >= min_pix:
                keep.append(i)
                last = xy
        return keep

    def _emit_edge(
        self,
        backend: Backend,
        family: DiagramFamily,
        segs: list,
        side: str,
        xlim,
        ylim,
        g: float,
    ) -> None:
        """Edge-placed rotated labels + overflow fallback.

        Top edge carries constant-d lines; right edge carries constant-a.
        Each line's crossing with the designated edge is the anchor; lines
        that miss it are routed to ``fallback`` (on the line, inside the
        plot) so every label-eligible isoline is labeled exactly once.
        """
        xmin, xmax = xlim
        ymin, ymax = ylim
        two_pi = 2.0 * math.pi

        # Compute crossings and classify (designated-edge vs miss → fallback)
        on_edge: list[tuple] = []  # (seg, fe, ve)
        missed: list[tuple] = []  # (seg, fe, ve, fallback_edge)
        for s in segs:
            if s.value <= 0:
                continue
            if family is DiagramFamily.DISPLACEMENT:
                f_top = ymax / (two_pi * s.value)
                if xmin <= f_top <= xmax:
                    on_edge.append((s, f_top, ymax))
                    continue
                v_right = two_pi * s.value * xmax
                if ymin <= v_right <= ymax:
                    missed.append((s, xmax, v_right, "right"))
            else:  # acceleration on 'right'
                a = s.value * g
                v_right = a / (two_pi * xmax)
                if ymin <= v_right <= ymax:
                    on_edge.append((s, xmax, v_right))
                    continue
                f_bot = a / (two_pi * ymin)
                if xmin <= f_bot <= xmax:
                    missed.append((s, f_bot, ymin, "bottom"))

        unit_suffix = self._unit_suffix(family)
        suffix = f" {unit_suffix}" if unit_suffix else ""

        # Edge labels (density filtered along the spine)
        if on_edge:
            on_segs = [c[0] for c in on_edge]
            angles = self._pixel_angles(backend, on_segs)
            if side == "top":
                pts = np.array([[c[1], ymax] for c in on_edge])
            else:
                pts = np.array([[xmax, c[2]] for c in on_edge])
            pix = backend.data_to_pixel(pts)
            # Reduce to scalar position along the spine for density walk
            if side == "top":
                pix1d = pix[:, 0]
            else:
                pix1d = pix[:, 1]

            # Replicate _density_filter's pixel-gap walk with 1D positions
            keep_e: list[int] = []
            last = None
            for i, s in enumerate(on_segs):
                is_major = _mantissa_int(s.value) == 1
                p = float(pix1d[i])
                if last is None or is_major or abs(p - last) >= MIN_PIX_EDGE:
                    keep_e.append(i)
                    last = p

            edge_items: list[LabelItem] = []
            ticks: list[TickItem] = []
            ux, uy = (0.0, 1.0) if side == "top" else (1.0, 0.0)
            outer_offset_pt = 7.0
            inner_pad_pt = 2.0

            for i in keep_e:
                seg, fe, ve = on_edge[i]
                ang = float(angles[i])
                ar = math.radians(ang)
                lx, ly = math.cos(ar), math.sin(ar)

                overflow_inside = False
                if side == "top":
                    frac = _log_frac(fe, xmin, xmax)
                    if frac > 1.0 - OVERFLOW_LOG_FRAC:
                        overflow_inside = True
                else:
                    frac = _log_frac(ve, ymin, ymax)
                    if frac < OVERFLOW_LOG_FRAC:
                        overflow_inside = True

                text = _diag.format_value(seg.value) + suffix
                if overflow_inside:
                    if lx * (-ux) + ly * (-uy) >= 0:
                        along_x, along_y = lx, ly
                    else:
                        along_x, along_y = -lx, -ly
                    try:
                        half_w = backend.measure_text_width_pt(text, "label") / 2
                    except Exception:
                        half_w = 15.0
                    d = half_w + inner_pad_pt
                    edge_items.append(
                        LabelItem(
                            text=text,
                            anchor=(float(fe), float(ve)),
                            rotation_deg=ang,
                            offset_pt=(along_x * d, along_y * d),
                            halign="center",
                            valign="center_baseline",
                            stroke=True,
                            style_key="label",
                        )
                    )
                else:
                    if lx * ux + ly * uy >= 0:
                        ha = "left"
                    else:
                        ha = "right"
                    edge_items.append(
                        LabelItem(
                            text=text,
                            anchor=(float(fe), float(ve)),
                            rotation_deg=ang,
                            offset_pt=(ux * outer_offset_pt, uy * outer_offset_pt),
                            halign=ha,
                            valign="center_baseline",
                            stroke=False,
                            style_key="label",
                        )
                    )
                    if side == "top":
                        ticks.append(TickItem(edge="top", position=float(fe)))
                    else:
                        ticks.append(TickItem(edge="right", position=float(ve)))

            backend.set_labels(family, "edge", edge_items)
            backend.set_ticks(family, ticks)
        else:
            backend.set_labels(family, "edge", [])
            backend.set_ticks(family, [])

        # Fallback labels — missed-edge segments placed on the line inside
        if missed:
            fb_items: list[LabelItem] = []
            miss_segs = [c[0] for c in missed]
            angles_fb = self._pixel_angles(backend, miss_segs)
            inner_pad_pt = 2.0
            edge_normals = {
                "right": (1.0, 0.0), "left": (-1.0, 0.0),
                "top": (0.0, 1.0),   "bottom": (0.0, -1.0),
            }
            for i, (seg, fe, ve, fb_edge) in enumerate(missed):
                ang = float(angles_fb[i])
                ar = math.radians(ang)
                lx, ly = math.cos(ar), math.sin(ar)
                ux, uy = edge_normals[fb_edge]
                if lx * (-ux) + ly * (-uy) >= 0:
                    along_x, along_y = lx, ly
                else:
                    along_x, along_y = -lx, -ly
                text = _diag.format_value(seg.value) + suffix
                try:
                    half_w = backend.measure_text_width_pt(text, "label") / 2
                except Exception:
                    half_w = 15.0
                d = half_w + inner_pad_pt
                fb_items.append(
                    LabelItem(
                        text=text,
                        anchor=(float(fe), float(ve)),
                        rotation_deg=ang,
                        offset_pt=(along_x * d, along_y * d),
                        halign="center",
                        valign="center_baseline",
                        stroke=True,
                        style_key="label",
                    )
                )
            backend.set_labels(family, "fallback", fb_items)
        else:
            backend.set_labels(family, "fallback", [])

    def _emit_axis_title(
        self, backend: Backend, family: DiagramFamily, segs: list,
    ) -> None:
        """The big centered 'Displacement (in)' / 'Acceleration (g)'
        callout. Anchored at the viewport's geometric (log-space) center
        rather than any individual segment — segments enter/leave on pan,
        and a segment-anchored title would teleport around. The viewport
        center is the only stable reference while the user is dragging.

        The title is snapped onto the nearest displacement/acceleration
        line so it still sits visually on a grid diagonal (the tripartite
        convention), but the *position along that line* tracks the
        viewport rather than the line's segment index.
        """
        if len(segs) < 2:
            backend.set_labels(family, "axis_title", [])
            return

        xlim = backend.get_xlim()
        ylim = backend.get_ylim()
        # Viewport centroid in log-space == geometric mean in linear space.
        fc = math.sqrt(xlim[0] * xlim[1])
        vc = math.sqrt(ylim[0] * ylim[1])

        # Snap (fc, vc) onto the nearest line in the family. For disp
        # (slope +1), the constant is d = v / (2π f); for accel (-1),
        # it's a = 2π f v. Pick the in-view segment closest in value.
        two_pi = 2.0 * math.pi
        if family is DiagramFamily.DISPLACEMENT:
            target_val = vc / (two_pi * fc)
        else:
            target_val = two_pi * fc * vc / self._g_value()
        nearest = min(segs, key=lambda s: abs(math.log10(s.value) - math.log10(target_val)))

        # Midpoint along THAT specific line, within the current viewport.
        fm = math.sqrt(nearest.f0 * nearest.f1)
        vm = math.sqrt(nearest.v0 * nearest.v1)
        angles = self._pixel_angles(backend, [nearest])
        ang = float(angles[0]) if len(angles) else 0.0
        unit = self._unit_suffix(family)
        if family is DiagramFamily.DISPLACEMENT:
            text = f"Displacement ({unit})" if unit else "Displacement"
        else:
            text = f"Acceleration ({unit})" if unit else "Acceleration"
        item = LabelItem(
            text=text,
            anchor=(fm, vm),
            rotation_deg=ang,
            offset_pt=(0.0, 0.0),
            halign="center",
            valign="center",
            stroke=False,
            style_key="axis_title",
        )
        backend.set_labels(family, "axis_title", [item])

    def _viewport_valid(self, xlim, ylim) -> bool:
        if not (all(np.isfinite(xlim)) and all(np.isfinite(ylim))):
            return False
        if min(xlim) <= 0 or min(ylim) <= 0:
            return False
        if xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
            return False
        return True


def _log_frac(v: float, lo: float, hi: float) -> float:
    if v <= 0 or lo <= 0 or hi <= 0:
        return 0.5
    return (math.log10(v) - math.log10(lo)) / max(
        math.log10(hi) - math.log10(lo), 1e-30
    )


def _clip_values(
    values: list[float],
    xlim,
    ylim,
    family: str,
    g: float,
) -> list:
    """Turn a list of constant-value picks into clipped segments.

    ``family`` is ``'disp'`` or ``'accel'``. ``g`` is the acceleration
    scale factor (1.0 for SI; 386.089 for imperial-in-g). Drops values
    whose clipped segment doesn't intersect the viewport.
    """
    out = []
    if family == "disp":
        for v in values:
            seg = _diag.displacement_segment(v, xlim, ylim)
            if seg is not None:
                out.append(seg)
    else:
        for v in values:
            seg = _diag.acceleration_segment(v, xlim, ylim, g_value=g)
            if seg is not None:
                out.append(seg)
    return out


def _decade_span(lo: float, hi: float) -> float:
    """Viewport span in decades. Returns 0 for degenerate ranges so the
    caller can treat sub-positive inputs as "not enough info" and fall
    back to the densest ladder."""
    if lo <= 0 or hi <= 0 or hi <= lo:
        return 0.0
    return math.log10(hi) - math.log10(lo)
