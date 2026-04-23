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
