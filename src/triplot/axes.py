"""TripartiteAxes — matplotlib Axes subclass using the pluggable tripartite core.

The Axes is a thin shell over :class:`triplot.core.TripartiteCore` plus a
:class:`triplot.backends.matplotlib_backend.MatplotlibBackend`. All the
math (nice-value picking, segment clipping, density filtering,
overflow-fallback routing) lives in the backend-agnostic core, so the
plotly backend renders an identical plot from the same state.

Backward compatibility: the historical test surface
(`ax._disp_labels`, `ax._disp_top_labels`, `ax._disp_fallback_labels`,
`ax._disp_axis_title`, `ax.diag_line_count`, `ax.diag_label_count`,
`ax.grid_diagonal`, `ax.set_diag_style`, `ax.set_label_mode`, ...)
still works — exposed via proxy properties that read the backend's
pools.
"""
from __future__ import annotations

import warnings

import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import Formatter, Locator

from . import ticks as _ticks

from . import core as _core
from . import isolines as _isolines
from .backends.base import DiagramFamily
from .backends.matplotlib_backend import MatplotlibBackend
from .units import UnitSystem, resolve as _resolve_units


def _plain_tick_formatter(x, pos):
    return f"{x:g}"


# Safe log-range bounds. Python floats max out at ~1.8e308, so anything
# past 10^290 risks :class:`OverflowError` in downstream 10**x math.
# 290 leaves comfortable margin — at that width the grid picker has
# already degraded due to float precision (ULP ~ 1 decade near 10^300).
_LOG_MAX = 290.0


def _clamp_log(v: float) -> float:
    """Clamp ``log10(v)`` into the safe float range."""
    if v > _LOG_MAX:
        return _LOG_MAX
    if v < -_LOG_MAX:
        return -_LOG_MAX
    return v


def _zoom_log_range(a: float, b: float, anchor: float, factor: float):
    """Compute new (lo, hi) after zooming by ``factor`` about ``anchor``
    in log space. Returns ``None`` when the result would cross the safe
    float boundary — the caller should treat that as "no-op this
    scroll tick" rather than silently clamping, because clamping would
    make further scroll-outs feel unresponsive.
    """
    import math
    if a <= 0 or b <= 0 or anchor <= 0:
        return None
    la, lb = math.log10(a), math.log10(b)
    lx = math.log10(anchor)
    new_la = lx + (la - lx) * factor
    new_lb = lx + (lb - lx) * factor
    # Reject if EITHER endpoint would leave the safe range — silently
    # clamping only one side would narrow the viewport asymmetrically.
    if abs(new_la) > _LOG_MAX or abs(new_lb) > _LOG_MAX:
        return None
    return 10.0 ** new_la, 10.0 ** new_lb


def _unit_from_label(label: str) -> str:
    """Extract bracketed unit from a label — re-exported for tests."""
    from .core import _unit_from_label as _f
    return _f(label)


class AdaptiveLogLocator(Locator):
    """Custom log-axis tick locator using :func:`ticks.major_minor_split`.

    Replaces matplotlib's default ``LogLocator`` pair (one for major, one
    for minor). ``LogLocator`` often emits ~60 minors on wide zooms
    ("insane amount of minor gridlines") and drops to a single minor
    between distant majors ("only one minor when spacing goes 1E2 to
    1E5"). Our split always keeps ~5-6 majors + 2-8 minors regardless
    of span.

    Instances come in pairs — one for the major-tick slot, one for the
    minor-tick slot — and share the same ``split`` function so their
    outputs are guaranteed disjoint.
    """

    def __init__(self, *, kind: str):
        if kind not in ("major", "minor"):
            raise ValueError("kind must be 'major' or 'minor'")
        self._kind = kind

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        if vmin <= 0 or vmax <= 0 or vmax <= vmin:
            return []
        majors, minors = _ticks.major_minor_split(float(vmin), float(vmax))
        return majors if self._kind == "major" else minors

    def tick_values(self, vmin, vmax):
        if vmin <= 0 or vmax <= 0 or vmax <= vmin:
            return []
        majors, minors = _ticks.major_minor_split(float(vmin), float(vmax))
        return majors if self._kind == "major" else minors


class _MajorOnlyFormatter(Formatter):
    """Format majors as ``%g``; minors are gridline-only (no text).

    matplotlib draws minor tick text by default when a minor locator
    emits values — even if those values look weird (``'3e2'`` between
    ``'100'`` and ``'1000'``). Returning ``''`` from the minor formatter
    keeps the minor grid without the clutter.
    """

    def __init__(self, *, major: bool = True):
        self._major = major

    def __call__(self, x, pos=None):
        return f"{x:g}" if self._major else ""


class TripartiteAxes(Axes):
    """Log-log frequency / pseudo-velocity axes with dynamic diagonal grids.

    All diagonal math (constant-d / constant-a line selection, clipping,
    density-filtered labels, overflow fallback) is delegated to
    :class:`triplot.core.TripartiteCore`. This class is responsible only
    for matplotlib-specific glue: the projection registration, axis
    scales / labels / formatters, draw-hook wiring, and the public
    ``grid_diagonal`` / ``set_diag_style`` / ``set_label_mode`` API.
    """

    name = "tripartite"

    def __init__(
        self,
        *args,
        units=None,
        style="seismic",
        aspect="auto",
        label_mode="edge",
        show_diag_titles=None,
        **kwargs,
    ):
        units_obj = _resolve_units(units)
        if style not in ("seismic", "shock", "dplot"):
            raise ValueError("style must be 'seismic', 'shock', or 'dplot'")
        if label_mode not in ("edge", "midpoint"):
            raise ValueError("label_mode must be 'edge' or 'midpoint'")

        self._core = _core.TripartiteCore(
            units=units_obj,
            style=style,
            label_mode=label_mode,
            show_diag_titles=show_diag_titles,
        )
        self._tri_aspect = aspect
        self._backend: MatplotlibBackend | None = None
        self._cache_key = None
        self._user_isolines: list[_isolines.UserIsoline] = []
        self._user_span_isolines: list[_isolines.UserSpanIsoline] = []

        super().__init__(*args, **kwargs)

        self.set_xscale("log")
        self.set_yscale("log")
        x0, x1 = self.get_xlim()
        if x0 <= 0 or x1 / max(x0, 1e-12) < 20:
            self.set_xlim(1, 1000)
            self.set_ylim(0.1, 100)

        self.set_xlabel(self._core.units.freq_label)
        self.set_ylabel(self._core.units.vel_label)
        # Two-tier grid styling: majors stronger, minors faint. Both
        # toggle together via grid(which='both') — disabling one tier is
        # via the matching axis.{xaxis,yaxis}.grid(False, which=...) hook.
        self.grid(True, which="major", linestyle="-", linewidth=0.5, color="0.7")
        self.grid(True, which="minor", linestyle="-", linewidth=0.3, color="0.85")

        for axis in (self.xaxis, self.yaxis):
            axis.set_major_locator(AdaptiveLogLocator(kind="major"))
            axis.set_minor_locator(AdaptiveLogLocator(kind="minor"))
            axis.set_major_formatter(_MajorOnlyFormatter(major=True))
            axis.set_minor_formatter(_MajorOnlyFormatter(major=False))

        # adjustable='datalim' keeps the axes BOX fixed and expands limits to
        # satisfy the aspect constraint. With 'box' (the mpl default) zoom
        # asymmetry shrinks the physical plot rectangle, which feels broken
        # — the figure appears to resize on every scroll.
        adjustable = "datalim" if self._tri_aspect == "equal" else "box"
        self.set_aspect(self._tri_aspect, adjustable=adjustable)

        # Reserve enough margin on the top / right for edge labels (which
        # sit OUTSIDE the spines, with annotation_clip=False and
        # in_layout=False so they don't influence any layout engine). Fixed
        # margins via subplots_adjust keep the axes rectangle stable during
        # interactive zoom — the root cause of the "graph area keeps
        # resizing" bug was letting a layout engine recompute margins from
        # zoom-varying tick label widths.
        fig = self.figure
        if fig is not None and self._label_mode_uses_edges():
            try:
                fig.subplots_adjust(left=0.10, right=0.88, top=0.88, bottom=0.10)
            except Exception:
                pass

        # Direct interactivity without needing the toolbar: middle-mouse
        # drag pans, scroll wheel zooms in/out around the cursor. The
        # diagonals re-snap automatically via the xlim/ylim callbacks
        # wired by the backend, so panning / zooming just works.
        self._pan_state: dict | None = None
        self._nav_cids: list = []

    # ---- backend lifecycle ----------------------------------------------

    def _ensure_backend(self) -> MatplotlibBackend:
        if self._backend is None:
            self._backend = MatplotlibBackend(self)
            # Rescale callback: interactive zoom / pan invalidates cache
            # so the NEXT draw rebuilds diagonals. Doing the rebuild here
            # would fight matplotlib's own draw loop; deferring is cleaner.
            self._backend.connect_rescale(self._invalidate_cache)
            self._install_nav()
        return self._backend

    def _install_nav(self) -> None:
        """Wire middle-mouse pan + scroll-wheel zoom on this axes.

        Kept idempotent so repeat backend attaches don't stack handlers.
        """
        if self._nav_cids:
            return
        canvas = self.figure.canvas
        if canvas is None:
            return
        self._nav_cids = [
            canvas.mpl_connect("button_press_event", self._on_button_press),
            canvas.mpl_connect("button_release_event", self._on_button_release),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("scroll_event", self._on_scroll),
        ]

    def _on_button_press(self, event) -> None:
        if event.inaxes is not self or event.button != 2:
            return
        # Capture PIXEL anchor (not data coords). Using event.xdata here
        # would break: set_xlim inside _on_motion changes the data-to-pixel
        # transform, so a cursor that hasn't moved would report a shifting
        # xdata frame-to-frame. Pixel deltas stay stable.
        self._pan_state = {
            "x_px": event.x,
            "y_px": event.y,
            "xlim": self.get_xlim(),
            "ylim": self.get_ylim(),
            "bbox": self.bbox.bounds,  # x0, y0, width, height in px
        }

    def _on_button_release(self, event) -> None:
        if event.button == 2:
            self._pan_state = None

    def _on_motion(self, event) -> None:
        if self._pan_state is None:
            return
        if event.x is None or event.y is None:
            return
        import math
        state = self._pan_state
        _, _, w, h = state["bbox"]
        if w <= 0 or h <= 0:
            return
        dx_px = event.x - state["x_px"]
        dy_px = event.y - state["y_px"]

        x0, x1 = state["xlim"]
        y0, y1 = state["ylim"]
        lx0, lx1 = math.log10(x0), math.log10(x1)
        ly0, ly1 = math.log10(y0), math.log10(y1)
        # Translate pixel drag -> log-space shift anchored on the STARTING
        # viewport. No cumulative drift since starting state is constant
        # across all motion events within a single press-release.
        log_shift_x = (dx_px / w) * (lx1 - lx0)
        log_shift_y = (dy_px / h) * (ly1 - ly0)
        self.set_xlim(10 ** (lx0 - log_shift_x), 10 ** (lx1 - log_shift_x))
        self.set_ylim(10 ** (ly0 - log_shift_y), 10 ** (ly1 - log_shift_y))
        self.figure.canvas.draw_idle()

    def _on_scroll(self, event) -> None:
        """Scroll wheel zooms about the cursor. Up = zoom in, down = out.

        Zoom factor per tick tuned to feel similar to plotly / Google Maps —
        small enough not to overshoot, large enough to reach any scale in
        a handful of ticks.

        Limits are clamped to ``[10^-290, 10^+290]`` so runaway zoom-out
        doesn't trip :class:`OverflowError` in ``10**x`` near the float
        max (~1.8e308). Users can still sweep 580 decades — plenty — and
        the grid picker stops producing sensible output past ~200
        decades anyway (float precision collapses).
        """
        if event.inaxes is not self or event.xdata is None or event.ydata is None:
            return
        factor = 0.8 if event.button == "up" else 1.25
        x0, x1 = self.get_xlim()
        y0, y1 = self.get_ylim()
        new_xlim = _zoom_log_range(x0, x1, event.xdata, factor)
        new_ylim = _zoom_log_range(y0, y1, event.ydata, factor)
        if new_xlim is None or new_ylim is None:
            return  # zoom-out would exceed safe float range; no-op
        self.set_xlim(*new_xlim)
        self.set_ylim(*new_ylim)
        self.figure.canvas.draw_idle()

    def _invalidate_cache(self) -> None:
        self._cache_key = None
        self.stale = True

    def _label_mode_uses_edges(self) -> bool:
        return self._core.label_mode == "edge"

    # ---- public API -----------------------------------------------------

    def set_displacement_label(self, label: str) -> None:
        u = self._core.units
        self._core.units = UnitSystem(
            u.name, u.freq_label, u.vel_label, label, u.accel_label,
            u.g_value, u.accel_in_g,
        )
        self._invalidate_cache()

    def set_acceleration_label(self, label: str) -> None:
        u = self._core.units
        self._core.units = UnitSystem(
            u.name, u.freq_label, u.vel_label, u.disp_label, label,
            u.g_value, u.accel_in_g,
        )
        self._invalidate_cache()

    def set_label_mode(self, mode: str) -> None:
        """Switch labeling style at runtime. ``'edge'`` or ``'midpoint'``."""
        self._core.set_label_mode(mode)
        self._invalidate_cache()

    def set_show_diag_titles(self, show) -> None:
        """Enable / disable the big centered 'Displacement' / 'Acceleration'
        titles. ``None`` restores the style-dependent default."""
        self._core.show_diag_titles = show
        self._invalidate_cache()

    def set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        """In edge label_mode the top margin carries rotated disp labels,
        so add extra ``pad`` when the caller didn't supply one — keeps the
        title clear of the rotated edge labels sitting along the top spine.

        NOTE: older versions of this method also auto-enabled
        ``constrained_layout``. That layout engine re-runs on every draw
        and rescales the axes rectangle whenever tick-label widths change
        (e.g. zooming between short labels like "10" and long ones like
        "0.001"). The observable effect during interactive zoom was "the
        plot area keeps resizing itself" — so the auto-enable is gone.
        Users who want constrained_layout can opt in on their figure.
        """
        if pad is None and self._core.label_mode == "edge":
            pad = 30.0
        return super().set_title(label, fontdict=fontdict, loc=loc, pad=pad, y=y, **kwargs)

    def legend(self, *args, **kwargs):
        """In edge label_mode the top-right / right margins carry labels,
        so matplotlib's ``loc='best'`` tends to collide. Default to
        ``'upper left'`` — the only corner edge labels don't touch."""
        if "loc" not in kwargs and self._core.label_mode == "edge":
            has_pos_loc = any(isinstance(a, str) for a in args[:4])
            if not has_pos_loc:
                kwargs["loc"] = "upper left"
        return super().legend(*args, **kwargs)

    def grid_diagonal(self, visible: bool = True, which: str = "major", **kwargs) -> None:
        """Toggle diagonal displacement / acceleration grid.

        ``which`` selects which tier of gridlines is shown. ``kwargs``
        override the uniform line style (color / linewidth / linestyle /
        alpha) — supplying any switches off the default tiered major /
        minor rendering."""
        if which not in ("major", "minor", "both"):
            raise ValueError("which must be 'major', 'minor', or 'both'")
        self._core.diag_visible = bool(visible)
        self._core.diag_which = which
        if kwargs:
            self._core.tiered_default = False
            self._core.line_style.update(kwargs)
            self._core.major_style.update(kwargs)
            self._core.minor_style.update(kwargs)
        self._invalidate_cache()

    def set_diag_style(
        self,
        major_linewidth=None,
        major_color=None,
        minor_linewidth=None,
        minor_color=None,
        label_fontsize=None,
        label_color=None,
    ) -> None:
        """Tune tiered diagonal grid appearance per-axes."""
        if major_linewidth is not None:
            self._core.major_style["linewidth"] = float(major_linewidth)
        if major_color is not None:
            self._core.major_style["color"] = major_color
        if minor_linewidth is not None:
            self._core.minor_style["linewidth"] = float(minor_linewidth)
        if minor_color is not None:
            self._core.minor_style["color"] = minor_color
        if label_fontsize is not None:
            self._core.label_style["fontsize"] = float(label_fontsize)
        if label_color is not None:
            self._core.label_style["color"] = label_color
        self._core.tiered_default = True
        self._invalidate_cache()

    def set_damping(self, ratio: float) -> None:
        if ratio < 0:
            raise ValueError("damping ratio must be >= 0")
        self._core.damping = float(ratio)

    def get_damping(self):
        return self._core.damping

    # ---- user-anchored isolines -----------------------------------------

    def add_isoline(
        self,
        family: str,
        value: float,
        *,
        label: str | None = None,
        line_style: dict | None = None,
        tick_style: dict | None = None,
    ) -> _isolines.UserIsoline:
        """Attach a permanent isoline for a specific constant value.

        ``family`` is one of ``'disp'`` / ``'displacement'``,
        ``'accel'`` / ``'acceleration'``, ``'vel'`` / ``'velocity'``
        (aliases documented in :mod:`triplot.isolines`). ``value`` is
        in the family's label units (inches for imperial disp, g's for
        ``accel_in_g``, in/s for imperial velocity).

        The returned :class:`~triplot.isolines.UserIsoline` carries the
        matplotlib artists — ``.line`` (the diagonal itself), ``.tick``
        (the mirror-spine tick), and ``.label`` (optional text). Tweak
        style directly, e.g.::

            iso = ax.add_isoline('disp', 0.5, label='0.5 in', line_style={'color': 'red'})
            iso.line.set_linewidth(2.0)

        The artists re-compute on every zoom/pan so the tick follows
        the crossing.
        """
        spec = _isolines.add(
            self,
            family, float(value),
            label=label,
            line_style=line_style,
            tick_style=tick_style,
        )
        self._user_isolines.append(spec)
        # Update once now so the spec is visible immediately — the
        # rebuild cycle only kicks in on draw, and a caller showing an
        # already-displayed figure expects the isoline to appear
        # without an extra redraw.
        g = self._core._g_value()
        _isolines.update(self, spec, g)
        self._invalidate_cache()
        return spec

    def remove_isoline(self, spec) -> None:
        """Remove a previously-added isoline or span-isoline."""
        if isinstance(spec, _isolines.UserSpanIsoline):
            try:
                self._user_span_isolines.remove(spec)
            except ValueError:
                return
            spec.remove()
            self._invalidate_cache()
            return
        try:
            self._user_isolines.remove(spec)
        except ValueError:
            return
        spec.remove()
        self._invalidate_cache()

    def get_isolines(self) -> list[_isolines.UserIsoline]:
        """Snapshot of attached user isolines (viewport-spanning)."""
        return list(self._user_isolines)

    def add_span_isoline(
        self,
        family: str,
        value: float,
        f_range: tuple[float, float],
        *,
        label: str | None = None,
        line_style: dict | None = None,
        label_style: dict | None = None,
    ) -> _isolines.UserSpanIsoline:
        """Attach a finite-span isoline bounded by ``f_range`` with a
        midpoint-sticking label.

        Like :meth:`add_isoline` but:
          * the line is drawn only between ``f_range[0]`` and
            ``f_range[1]`` on the frequency axis
          * no mirror-spine tick — the identifier is an in-plot text
            label instead
          * the label sits at the geometric midpoint of whichever
            portion of the line is currently visible, so as you pan and
            the line crops, the label slides to stay centered on what's
            on screen; when the line leaves the viewport entirely the
            label hides

        Same family aliases as :meth:`add_isoline`
        (``'disp'``/``'accel'``/``'vel'``). Returns a
        :class:`~triplot.isolines.UserSpanIsoline` with ``.line`` and
        ``.label`` matplotlib artists for direct styling.
        """
        spec = _isolines.add_span(
            self, family, float(value), f_range,
            label=label, line_style=line_style, label_style=label_style,
        )
        self._user_span_isolines.append(spec)
        g = self._core._g_value()
        _isolines.update_span(self, spec, g)
        self._invalidate_cache()
        return spec

    def get_span_isolines(self) -> list[_isolines.UserSpanIsoline]:
        """Snapshot of attached finite-span isolines."""
        return list(self._user_span_isolines)

    def _update_user_isolines(self) -> None:
        """Refresh each user isoline against the current viewport.
        Called from :meth:`draw` after the grid rebuild."""
        if not self._user_isolines and not self._user_span_isolines:
            return
        g = self._core._g_value()
        for spec in self._user_isolines:
            try:
                _isolines.update(self, spec, g)
            except Exception:
                # Never let a single bad isoline break the draw loop.
                continue
        for spec in self._user_span_isolines:
            try:
                _isolines.update_span(self, spec, g)
            except Exception:
                continue

    # ---- backward-compat properties (test / debug surface) --------------

    @property
    def _disp_labels(self):
        if self._backend is None:
            return []
        return self._backend._label_pools.get((DiagramFamily.DISPLACEMENT, "midpoint"), [])

    @property
    def _accel_labels(self):
        if self._backend is None:
            return []
        return self._backend._label_pools.get((DiagramFamily.ACCELERATION, "midpoint"), [])

    @property
    def _disp_top_labels(self):
        if self._backend is None:
            return []
        return self._backend._label_pools.get((DiagramFamily.DISPLACEMENT, "edge"), [])

    @property
    def _accel_right_labels(self):
        if self._backend is None:
            return []
        return self._backend._label_pools.get((DiagramFamily.ACCELERATION, "edge"), [])

    @property
    def _disp_top_ticks(self):
        if self._backend is None:
            return []
        return self._backend._tick_pools[DiagramFamily.DISPLACEMENT]

    @property
    def _accel_right_ticks(self):
        if self._backend is None:
            return []
        return self._backend._tick_pools[DiagramFamily.ACCELERATION]

    @property
    def _disp_fallback_labels(self):
        if self._backend is None:
            return []
        return self._backend._label_pools.get((DiagramFamily.DISPLACEMENT, "fallback"), [])

    @property
    def _accel_fallback_labels(self):
        if self._backend is None:
            return []
        return self._backend._label_pools.get((DiagramFamily.ACCELERATION, "fallback"), [])

    @property
    def _disp_axis_title(self):
        if self._backend is None:
            return None
        pool = self._backend._label_pools.get((DiagramFamily.DISPLACEMENT, "axis_title"), [])
        return pool[0] if pool else None

    @property
    def _accel_axis_title(self):
        if self._backend is None:
            return None
        pool = self._backend._label_pools.get((DiagramFamily.ACCELERATION, "axis_title"), [])
        return pool[0] if pool else None

    @property
    def _disp_collection(self):
        if self._backend is None:
            return None
        return self._backend._collections.get(DiagramFamily.DISPLACEMENT)

    @property
    def _accel_collection(self):
        if self._backend is None:
            return None
        return self._backend._collections.get(DiagramFamily.ACCELERATION)

    @property
    def _label_mode(self) -> str:
        return self._core.label_mode

    @property
    def _style(self) -> str:
        return self._core.style

    @property
    def _units(self) -> UnitSystem:
        return self._core.units

    @property
    def diag_line_count(self) -> int:
        if self._backend is None:
            return 0
        n = 0
        for coll in self._backend._collections.values():
            if coll is not None:
                n += len(coll.get_segments())
        return n

    @property
    def diag_label_count(self) -> int:
        if self._backend is None:
            return 0
        n = 0
        for (_, role), pool in self._backend._label_pools.items():
            if role == "axis_title":
                continue
            n += sum(1 for a in pool if a.get_visible())
        return n

    @property
    def _diag_artists(self) -> list:
        """Flat list of current diagonal artists — kept for debug use."""
        if self._backend is None:
            return []
        out = []
        for coll in self._backend._collections.values():
            if coll is not None and len(coll.get_segments()) > 0:
                out.append(coll)
        for pool in self._backend._label_pools.values():
            out.extend(pool)
        return out

    # ---- backward-compat hooks for tests --------------------------------

    def _subdivisions(self) -> tuple[float, ...]:
        """Expose the core's subdivision selection — used by tests."""
        return self._core.subdivisions()

    def _label_subdivisions(self) -> tuple[float, ...]:
        return self._core.label_subdivisions()

    def _rebuild_diagonals(self) -> None:
        """Rebuild diagonals via the core. Tests monkeypatch this to
        inject failures; keeping the name preserves that hook."""
        self._core.rebuild(self._ensure_backend())

    def _view_is_valid(self, xlim, ylim) -> bool:
        """Gate used by the core before feeding limits to the tick picker.
        Rejects non-finite, non-positive, or inverted ranges."""
        return self._core._viewport_valid(xlim, ylim)

    @property
    def _diag_visible(self) -> bool:
        return self._core.diag_visible

    @_diag_visible.setter
    def _diag_visible(self, value: bool) -> None:
        self._core.diag_visible = bool(value)
        self._invalidate_cache()

    # ---- draw / cache ---------------------------------------------------

    def _cache_signature(self):
        try:
            bbox = tuple(self.bbox.bounds)
        except Exception:
            bbox = (0.0, 0.0, 0.0, 0.0)
        c = self._core
        return (
            tuple(self.get_xlim()),
            tuple(self.get_ylim()),
            self.get_xscale(),
            self.get_yscale(),
            c.diag_visible,
            c.diag_which,
            c.tiered_default,
            tuple(sorted(c.line_style.items())),
            tuple(sorted(c.major_style.items())),
            tuple(sorted(c.minor_style.items())),
            tuple(sorted(c.label_style.items())),
            c.units.name,
            c.units.accel_in_g,
            c.units.g_value,
            c.units.disp_label,
            c.units.accel_label,
            c.label_mode,
            c.show_diag_titles,
            bbox,
        )

    def draw(self, renderer):
        self._unstale_viewLim()
        backend = self._ensure_backend()
        try:
            sig = self._cache_signature()
        except Exception:
            sig = None
        if sig is None or sig != self._cache_key:
            try:
                self._rebuild_diagonals()
                self._cache_key = sig
            except Exception as exc:  # noqa: BLE001 — never break draw()
                warnings.warn(f"triplot: diagonal rebuild failed: {exc!r}", stacklevel=2)
                self._cache_key = None
        # User isolines refresh every draw regardless of the cache
        # signature — their tick position depends on xlim/ylim, and
        # those deltas are what cache misses detect anyway. Running the
        # per-isoline update outside the cache gate keeps the grid
        # rebuild fast while still tracking isolines through zoom/pan.
        self._update_user_isolines()
        super().draw(renderer)

    def clear(self):
        # User isolines own Line2D/Text artists that matplotlib will
        # detach in clear() — drop our references so we don't carry
        # zombie specs pointing at removed artists.
        self._user_isolines = []
        self._user_span_isolines = []
        result = super().clear()
        if self._backend is not None:
            try:
                self._backend.teardown()
            except Exception:
                pass
            self._backend = None
        self._cache_key = None
        return result

    # ---- pickle ---------------------------------------------------------

    def __getstate__(self):
        state = self.__dict__.copy()
        # Drop live backend (artists aren't picklable across all backends);
        # it rebuilds lazily on next draw.
        state["_backend"] = None
        state["_cache_key"] = None
        # Legacy keys — preserved so pickled plots from before the backend
        # refactor can round-trip, and so tests can inspect them.
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
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
