"""User-anchored constant-value isolines for TripartiteAxes.

Lets callers drop a permanent line at a specific displacement,
acceleration, or velocity value onto an existing tripartite plot, each
one auto-recomputing its clipped segment + edge tick on every zoom /
pan. Independent from the dynamic gridline picker — these lines are
deliberate annotations from the caller, not automatic grid decoration.

Tick placement rules — ticks sit on the **opposite spine** from the
family's grid ticks so they read visually as "user-anchored", not
"grid":

+----------+--------------+-------------------+
| family   | grid ticks   | user-isoline tick |
+==========+==============+===================+
| disp     | top spine    | bottom spine      |
| accel    | right spine  | left spine        |
| velocity | (yaxis left) | right spine       |
+----------+--------------+-------------------+

When geometry means the isoline doesn't cross its opposite spine
(common near the corners of the viewport), the tick falls back to the
spine it *does* intersect, but on the opposite side from where the
grid's own fallback labels live — same "opposite" principle, graceful
degradation.

Artists are plain matplotlib ``Line2D`` / ``Text`` instances with no
wrapping, so callers can ``set_color``, ``set_linewidth``, or swap the
full style dict directly on the returned artist handle.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.transforms import blended_transform_factory


# Acceptable family spellings — everything routes to one of three canonical
# internal names so the rebuild code stays simple.
_FAMILY_ALIASES = {
    "disp": "disp", "displacement": "disp", "d": "disp",
    "accel": "accel", "acceleration": "accel", "a": "accel",
    "vel": "vel", "velocity": "vel", "v": "vel", "pv": "vel",
    "pseudo_velocity": "vel", "pseudo-velocity": "vel",
}

_TWO_PI = 2.0 * math.pi


# --- defaults borrowed to match triplot's native tick look -----------------
_DEFAULT_LINE_STYLE = dict(
    color="#c0392b", linewidth=1.3, linestyle="-", alpha=0.9,
)
_DEFAULT_TICK_STYLE = dict(
    color="#c0392b", linewidth=1.0,
)
# Ticks extend 0.6% of axes extent — same length as grid ticks so they read
# as native to the plot even though they sit on the opposite spine.
_TICK_LEN_FRAC = 0.012  # slightly longer than grid ticks (0.006) for visibility
_LABEL_OFFSET_PT = 4.0


@dataclass
class UserIsoline:
    """Spec + artist handles for one user-anchored isoline.

    ``family`` is the canonical name (``'disp'``, ``'accel'``, ``'vel'``).
    ``value`` is in the family's label units (inches for disp imperial,
    g's for accel when ``accel_in_g``, in/s for velocity imperial).
    ``label`` is optional annotation text drawn near the tick.

    Artists are matplotlib objects; tweak them directly via
    ``spec.line.set_color(...)`` etc.
    """
    family: str
    value: float
    line: Line2D
    tick: Line2D
    label: Optional[Text] = None
    label_text: str = ""
    line_style: dict = field(default_factory=dict)
    tick_style: dict = field(default_factory=dict)

    def remove(self) -> None:
        """Detach all artists from the axes. After calling this the
        spec is dead — don't pass it back to the updater."""
        for art in (self.line, self.tick, self.label):
            if art is None:
                continue
            try:
                art.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass


def _canonical_family(name: str) -> str:
    key = name.strip().lower()
    if key not in _FAMILY_ALIASES:
        raise ValueError(
            f"unknown isoline family {name!r}; "
            f"expected one of: disp, accel, vel"
        )
    return _FAMILY_ALIASES[key]


def _compute_segment(family: str, value: float, xlim, ylim, g: float):
    """Return ``((f0, v0), (f1, v1))`` clipped to the viewport, or ``None``
    if the isoline doesn't intersect the viewport at all. Pure math; no
    matplotlib state.
    """
    if value <= 0:
        return None
    xmin, xmax = xlim
    ymin, ymax = ylim

    if family == "disp":
        # v = 2π f d  — slope +1 in log space
        # At x=xmin: v = 2π xmin d
        # At x=xmax: v = 2π xmax d
        v_at_xmin = _TWO_PI * xmin * value
        v_at_xmax = _TWO_PI * xmax * value
        f_at_ymin = ymin / (_TWO_PI * value)
        f_at_ymax = ymax / (_TWO_PI * value)
    elif family == "accel":
        # v = a / (2π f)  — slope -1
        a = value * g
        v_at_xmin = a / (_TWO_PI * xmin)
        v_at_xmax = a / (_TWO_PI * xmax)
        f_at_ymin = a / (_TWO_PI * ymin)
        f_at_ymax = a / (_TWO_PI * ymax)
    else:  # vel
        # v = const — horizontal line at y=value
        if value < ymin or value > ymax:
            return None
        return (xmin, value), (xmax, value)

    # Collect candidate intersections with the 4 viewport edges
    pts = []
    if ymin <= v_at_xmin <= ymax:
        pts.append((xmin, v_at_xmin))
    if ymin <= v_at_xmax <= ymax:
        pts.append((xmax, v_at_xmax))
    if xmin <= f_at_ymin <= xmax:
        pts.append((f_at_ymin, ymin))
    if xmin <= f_at_ymax <= xmax:
        pts.append((f_at_ymax, ymax))
    # De-duplicate corner hits
    uniq = []
    for p in pts:
        if not any(
            math.isclose(p[0], q[0]) and math.isclose(p[1], q[1])
            for q in uniq
        ):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    uniq.sort()
    return uniq[0], uniq[-1]


def _opposite_edge_crossing(family: str, value: float, xlim, ylim, g: float):
    """Crossing of the isoline with the OPPOSITE spine (user-tick edge).

    Returns ``(edge, position)`` where ``edge`` is one of
    ``'top' | 'bottom' | 'left' | 'right'`` and ``position`` is the
    coordinate along that spine. Returns ``None`` if the opposite edge
    isn't crossed.

    Preferred opposite:
      disp   -> bottom (grid ticks live top)
      accel  -> left   (grid ticks live right)
      vel    -> right  (default y-axis ticks live left)

    If the preferred spine isn't crossed, fall back to the orthogonal
    opposite (``top`` -> ``left`` for disp, ``bottom`` -> ``right`` for
    accel) so a tick still lands somewhere visible.
    """
    xmin, xmax = xlim
    ymin, ymax = ylim

    if family == "disp":
        # Preferred: bottom. Isoline v = 2π f d crosses y=ymin at f = ymin/(2π d)
        f_bot = ymin / (_TWO_PI * value)
        if xmin <= f_bot <= xmax:
            return "bottom", f_bot
        # Fallback: left spine. x=xmin -> v = 2π xmin d
        v_left = _TWO_PI * xmin * value
        if ymin <= v_left <= ymax:
            return "left", v_left
        return None

    if family == "accel":
        a = value * g
        # Preferred: left. x=xmin -> v = a/(2π xmin)
        v_left = a / (_TWO_PI * xmin)
        if ymin <= v_left <= ymax:
            return "left", v_left
        # Fallback: bottom. y=ymin -> f = a/(2π ymin)
        f_bot = a / (_TWO_PI * ymin)
        if xmin <= f_bot <= xmax:
            return "bottom", f_bot
        return None

    # vel: horizontal line at y=value crosses right spine at (xmax, value)
    if ymin <= value <= ymax:
        return "right", value
    return None


def _make_line(ax, style: dict) -> Line2D:
    """Empty Line2D in data coords, styled + in_layout=False so layout
    engines ignore it."""
    line = Line2D([], [], **style)
    line.set_clip_on(True)
    try:
        line.set_in_layout(False)
    except AttributeError:
        pass
    ax.add_line(line)
    return line


def _make_tick(ax, edge: str, style: dict) -> Line2D:
    """Short spine-anchored tick. Uses blended transform so one axis is
    the spine (axes fraction) and the other is data space — matches
    triplot's grid ticks."""
    if edge in ("top", "bottom"):
        trans = blended_transform_factory(ax.transData, ax.transAxes)
    else:
        trans = blended_transform_factory(ax.transAxes, ax.transData)
    line = Line2D([], [], transform=trans, **style)
    line.set_clip_on(False)
    try:
        line.set_in_layout(False)
    except AttributeError:
        pass
    ax.add_line(line)
    return line


def _tick_endpoints(edge: str, position: float):
    """Return ``(xs, ys)`` lists for a tick at ``position`` on ``edge``,
    protruding INTO the plot from the spine. Length is :const:`_TICK_LEN_FRAC`
    of the axes extent."""
    L = _TICK_LEN_FRAC
    if edge == "top":
        # Protrude inward (downward, into plot)
        return [position, position], [1.0, 1.0 - L]
    if edge == "bottom":
        return [position, position], [0.0, 0.0 + L]
    if edge == "right":
        return [1.0, 1.0 - L], [position, position]
    # left
    return [0.0, 0.0 + L], [position, position]


def add(
    ax,
    family: str,
    value: float,
    *,
    label: Optional[str] = None,
    line_style: Optional[dict] = None,
    tick_style: Optional[dict] = None,
) -> UserIsoline:
    """Attach a user-anchored isoline to ``ax`` and return its spec.

    Call :meth:`UserIsoline.remove` or ``ax.remove_isoline(spec)`` to
    take it back off. The isoline's artists live on the axes and auto-
    update via ``ax.draw`` — the same zoom/pan cycle that refreshes the
    grid refreshes these too.
    """
    fam = _canonical_family(family)
    if value <= 0:
        raise ValueError(f"isoline value must be positive, got {value}")

    ls = dict(_DEFAULT_LINE_STYLE)
    if line_style:
        ls.update(line_style)
    ts = dict(_DEFAULT_TICK_STYLE)
    if tick_style:
        ts.update(tick_style)

    line = _make_line(ax, ls)
    # Edge isn't known yet — tick created with a placeholder edge, the
    # updater pass re-parents the transform when the edge changes.
    tick = _make_tick(ax, "bottom", ts)

    text: Optional[Text] = None
    if label:
        text = Text(
            0, 0, label,
            color=ts.get("color", "#222"),
            fontsize=9, ha="left", va="center",
        )
        try:
            text.set_in_layout(False)
        except AttributeError:
            pass
        ax.add_artist(text)

    spec = UserIsoline(
        family=fam,
        value=float(value),
        line=line,
        tick=tick,
        label=text,
        label_text=label or "",
        line_style=ls,
        tick_style=ts,
    )
    return spec


def update(ax, spec: UserIsoline, g: float) -> None:
    """Recompute segment + tick for ``spec`` from the current axes
    viewport. Called on every draw; cheap enough to run per-isoline per
    frame."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if xlim[0] <= 0 or ylim[0] <= 0 or xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
        _hide(spec)
        return

    seg = _compute_segment(spec.family, spec.value, xlim, ylim, g)
    if seg is None:
        _hide(spec)
        return
    (f0, v0), (f1, v1) = seg
    spec.line.set_data([f0, f1], [v0, v1])
    spec.line.set_visible(True)

    crossing = _opposite_edge_crossing(spec.family, spec.value, xlim, ylim, g)
    if crossing is None:
        spec.tick.set_visible(False)
        if spec.label is not None:
            spec.label.set_visible(False)
        return

    edge, pos = crossing
    # Reparent the tick's transform when the edge flips — e.g. disp line
    # moves from bottom (preferred) to left (fallback) during aggressive
    # pan. A stale transform leaves the tick drawn on the wrong spine.
    _retarget_tick_transform(ax, spec.tick, edge)
    xs, ys = _tick_endpoints(edge, pos)
    spec.tick.set_data(xs, ys)
    spec.tick.set_visible(True)

    if spec.label is not None:
        _position_label(ax, spec, edge, pos)


def _retarget_tick_transform(ax, tick: Line2D, edge: str) -> None:
    if edge in ("top", "bottom"):
        tick.set_transform(blended_transform_factory(ax.transData, ax.transAxes))
    else:
        tick.set_transform(blended_transform_factory(ax.transAxes, ax.transData))


def _position_label(ax, spec: UserIsoline, edge: str, pos: float) -> None:
    """Place ``spec.label`` just inside the spine, offset from the tick
    so the two don't collide."""
    # Transform the tick's inner end to data coords for anchor placement
    if edge == "bottom":
        xs, ys = [pos, pos], [0.0, _TICK_LEN_FRAC]
        spec.label.set_transform(
            blended_transform_factory(ax.transData, ax.transAxes)
        )
        spec.label.set_position((pos, _TICK_LEN_FRAC + 0.005))
        spec.label.set_ha("center")
        spec.label.set_va("bottom")
    elif edge == "top":
        spec.label.set_transform(
            blended_transform_factory(ax.transData, ax.transAxes)
        )
        spec.label.set_position((pos, 1.0 - _TICK_LEN_FRAC - 0.005))
        spec.label.set_ha("center")
        spec.label.set_va("top")
    elif edge == "left":
        spec.label.set_transform(
            blended_transform_factory(ax.transAxes, ax.transData)
        )
        spec.label.set_position((_TICK_LEN_FRAC + 0.005, pos))
        spec.label.set_ha("left")
        spec.label.set_va("center")
    else:  # right
        spec.label.set_transform(
            blended_transform_factory(ax.transAxes, ax.transData)
        )
        spec.label.set_position((1.0 - _TICK_LEN_FRAC - 0.005, pos))
        spec.label.set_ha("right")
        spec.label.set_va("center")
    spec.label.set_text(spec.label_text)
    spec.label.set_visible(True)


def _hide(spec: UserIsoline) -> None:
    spec.line.set_visible(False)
    spec.tick.set_visible(False)
    if spec.label is not None:
        spec.label.set_visible(False)


# ---------------------------------------------------------------------------
# Span isolines — finite in frequency, label sticks to visible midpoint
# ---------------------------------------------------------------------------
#
# Same family math as UserIsoline but the line is bounded by an explicit
# ``f_range = (f_start, f_end)``. The label is a matplotlib Text that sits
# at the geometric midpoint of whatever portion of the line is currently
# in view — "squish" behaviour: as the viewport crops the line, the label
# slides along the visible portion to stay centered. When the entire
# (f_range × family-value) line leaves the viewport, the label is hidden
# (so you don't get phantom text hanging off-screen).


@dataclass
class UserSpanIsoline:
    """Finite isoline with a midpoint-sticking label.

    Like :class:`UserIsoline` but:
      * limited to ``f_range = (f_start, f_end)``
      * no mirror-spine tick
      * ``label`` tracks the visible midpoint (log-geometric) and
        rotates with the on-screen line slope (in pixel space)

    ``line`` and ``label`` are plain matplotlib artists — tweak styles
    directly.
    """
    family: str
    value: float
    f_start: float
    f_end: float
    line: Line2D
    label: Text
    label_text: str
    line_style: dict = field(default_factory=dict)
    label_style: dict = field(default_factory=dict)

    def remove(self) -> None:
        for art in (self.line, self.label):
            if art is None:
                continue
            try:
                art.remove()
            except (ValueError, NotImplementedError, AttributeError):
                pass


_DEFAULT_SPAN_LABEL_STYLE = dict(
    color="#8e44ad", fontsize=9, family="serif",
)


def _nominal_span_endpoints(
    family: str, value: float, f_start: float, f_end: float, g: float,
):
    """Compute the (f, v) endpoints of the constant-family line over
    ``[f_start, f_end]``. No clipping to viewport — that's layered on
    top. Returns ``((f0, v0), (f1, v1))`` with f0 < f1."""
    if f_start > f_end:
        f_start, f_end = f_end, f_start
    if family == "disp":
        v0 = _TWO_PI * f_start * value
        v1 = _TWO_PI * f_end * value
    elif family == "accel":
        a = value * g
        v0 = a / (_TWO_PI * f_start)
        v1 = a / (_TWO_PI * f_end)
    else:  # vel — horizontal, v is constant
        v0 = value
        v1 = value
    return (f_start, v0), (f_end, v1)


def _clip_to_viewport(
    p0: tuple[float, float], p1: tuple[float, float], xlim, ylim,
):
    """Clip the segment ``p0->p1`` to the viewport. Returns clipped
    endpoints or ``None`` if fully outside. Log-space aware (works in
    linear data coords but assumes positive log-axis values)."""
    if xlim[0] <= 0 or ylim[0] <= 0:
        return None
    (x0, y0), (x1, y1) = p0, p1
    if x0 <= 0 or x1 <= 0 or y0 <= 0 or y1 <= 0:
        return None
    # Parameterize in log space so clipping is a Liang-Barsky-style
    # scalar interpolation — cheap, stable.
    lx0, ly0 = math.log10(x0), math.log10(y0)
    lx1, ly1 = math.log10(x1), math.log10(y1)
    lxmin, lxmax = math.log10(xlim[0]), math.log10(xlim[1])
    lymin, lymax = math.log10(ylim[0]), math.log10(ylim[1])

    dx = lx1 - lx0
    dy = ly1 - ly0

    t_enter = 0.0
    t_exit = 1.0

    # Clip against each of the four viewport edges in log space.
    for p, q in (
        (-dx, lx0 - lxmin),   # left   p*t <= q
        ( dx, lxmax - lx0),   # right
        (-dy, ly0 - lymin),   # bottom
        ( dy, lymax - ly0),   # top
    ):
        if abs(p) < 1e-30:
            if q < 0:
                return None  # line parallel to edge AND outside — no visible portion
            continue
        t = q / p
        if p < 0:
            if t > t_exit:
                return None
            if t > t_enter:
                t_enter = t
        else:
            if t < t_enter:
                return None
            if t < t_exit:
                t_exit = t

    if t_exit < t_enter:
        return None

    # Reconstruct visible endpoints in linear coords
    f_a = 10.0 ** (lx0 + dx * t_enter)
    v_a = 10.0 ** (ly0 + dy * t_enter)
    f_b = 10.0 ** (lx0 + dx * t_exit)
    v_b = 10.0 ** (ly0 + dy * t_exit)
    return (f_a, v_a), (f_b, v_b)


def add_span(
    ax,
    family: str,
    value: float,
    f_range: tuple[float, float],
    *,
    label: Optional[str] = None,
    line_style: Optional[dict] = None,
    label_style: Optional[dict] = None,
) -> UserSpanIsoline:
    """Attach a finite-span isoline with a midpoint-sticking label.

    Unlike :func:`add`, which draws a viewport-spanning isoline with a
    mirror-spine tick, this variant is bounded on both ends by
    ``f_range = (f_start, f_end)`` and carries an in-plot text label
    rather than an axis tick.

    Label behaviour:
      * rotates to match the line's on-screen slope (pixel-space angle,
        so aspect-ratio changes keep the rotation correct)
      * sits at the geometric midpoint of whatever *visible* portion of
        the line is in the current viewport
      * hides when the line is fully outside the viewport
    """
    fam = _canonical_family(family)
    if value <= 0:
        raise ValueError(f"isoline value must be positive, got {value}")
    if f_range[0] <= 0 or f_range[1] <= 0 or f_range[0] == f_range[1]:
        raise ValueError(
            f"f_range must be two positive, distinct frequencies; got {f_range!r}"
        )

    ls = dict(_DEFAULT_LINE_STYLE)
    if line_style:
        ls.update(line_style)
    tx_style = dict(_DEFAULT_SPAN_LABEL_STYLE)
    if label_style:
        tx_style.update(label_style)

    line = _make_line(ax, ls)
    text = Text(0, 0, label or "", **tx_style)
    text.set_ha("center")
    text.set_va("center")
    text.set_rotation_mode("anchor")
    try:
        text.set_in_layout(False)
    except AttributeError:
        pass
    ax.add_artist(text)

    spec = UserSpanIsoline(
        family=fam,
        value=float(value),
        f_start=float(f_range[0]),
        f_end=float(f_range[1]),
        line=line,
        label=text,
        label_text=label or "",
        line_style=ls,
        label_style=tx_style,
    )
    return spec


def update_span(ax, spec: UserSpanIsoline, g: float) -> None:
    """Recompute the visible portion of ``spec`` and reposition its
    label. Cheap — runs per-frame."""
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if xlim[0] <= 0 or ylim[0] <= 0 or xlim[1] <= xlim[0] or ylim[1] <= ylim[0]:
        _hide_span(spec)
        return

    p0, p1 = _nominal_span_endpoints(
        spec.family, spec.value, spec.f_start, spec.f_end, g,
    )
    clipped = _clip_to_viewport(p0, p1, xlim, ylim)
    if clipped is None:
        _hide_span(spec)
        return

    (fa, va), (fb, vb) = clipped
    spec.line.set_data([fa, fb], [va, vb])
    spec.line.set_visible(True)

    if not spec.label_text:
        spec.label.set_visible(False)
        return

    # Midpoint of the visible portion in log space (== geometric midpoint
    # in linear space). This is the "squish" behaviour — as the viewport
    # crops the line, the label slides toward the visible middle rather
    # than hanging at an invisible nominal center.
    fm = math.sqrt(fa * fb)
    vm = math.sqrt(va * vb)

    # Rotation in pixel space so the label rides with the line's
    # on-screen slope, not the data-space slope (which would be wrong
    # under aspect='auto' or asymmetric zoom).
    try:
        pa = ax.transData.transform((fa, va))
        pb = ax.transData.transform((fb, vb))
        angle = math.degrees(math.atan2(pb[1] - pa[1], pb[0] - pa[0]))
        # Keep text upright: clamp rotation to [-90, 90].
        if angle > 90:
            angle -= 180
        if angle < -90:
            angle += 180
    except Exception:
        angle = 0.0

    spec.label.set_position((fm, vm))
    spec.label.set_rotation(angle)
    spec.label.set_text(spec.label_text)
    spec.label.set_visible(True)


def _hide_span(spec: UserSpanIsoline) -> None:
    spec.line.set_visible(False)
    spec.label.set_visible(False)
