"""Backend protocol for triplot.

The backend is a thin surface translating backend-agnostic draw requests
(emitted by :class:`triplot.core.TripartiteCore`) into artist or figure
operations specific to matplotlib or plotly. Only rendering state
crosses the boundary — math (clipping, nice values, label selection)
lives in the shared core.

Two families of diagonal live side by side and must be tracked
independently so styles and pools don't collide:

- ``DiagramFamily.DISPLACEMENT`` — constant-d lines (slope +1 in log-log).
- ``DiagramFamily.ACCELERATION`` — constant-a lines (slope -1 in log-log).

Every backend-specific drawing op addresses one family + one "role":

- ``'line'`` — the diagonal gridline itself.
- ``'midpoint'`` — labels inside the plot at segment midpoints.
- ``'edge'`` — rotated labels at the designated spine crossing.
- ``'edge_tick'`` — short spine ticks under the edge labels.
- ``'fallback'`` — labels placed inside the plot when a line misses the
  designated edge (overflow handling).
- ``'axis_title'`` — the big centered "Displacement" / "Acceleration"
  callout across the middle of the plot.

The core batches its work per-frame, so a backend can choose to diff
against previous state for efficiency — but doing so is optional.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Iterable


class DiagramFamily(str, Enum):
    DISPLACEMENT = "displacement"
    ACCELERATION = "acceleration"


@dataclass(frozen=True)
class LabelItem:
    """A label the core wants the backend to draw (or reposition).

    ``anchor`` — (x, y) in data coords. For edge labels this is the spine
    crossing; for midpoints it's the segment midpoint; for fallback it's
    the opposite-edge crossing.

    ``rotation_deg`` — pre-computed to match the line's on-screen slope
    (already corrected to [-90, 90]). The core computes this using the
    backend's ``data_to_pixel`` so it matches the actual aspect.

    ``offset_pt`` — (dx, dy) in points, applied on top of the anchor.
    Lets the backend support both "nudge outward from spine" (for edge
    labels) and "sit on the line" (for midpoint / fallback) without the
    core needing to know pixel math.

    ``halign`` / ``valign`` — standard 'left' / 'center' / 'right' and
    'top' / 'center' / 'bottom' / 'center_baseline'.

    ``stroke`` — True means render with a white glyph halo effect
    (readable over grid without erasing a bbox rectangle). False means
    plain bbox halo.
    """
    text: str
    anchor: tuple[float, float]
    rotation_deg: float
    offset_pt: tuple[float, float] = (0.0, 0.0)
    halign: str = "center"
    valign: str = "center"
    stroke: bool = False
    style_key: str = "label"  # backend-resolved style bucket


@dataclass(frozen=True)
class TickItem:
    """A short edge tick the backend should draw at a spine crossing."""
    edge: str  # 'top' | 'right' | 'bottom' | 'left'
    position: float  # data coord along the spine (x for top/bot, y for left/right)
    length_axes: float = 0.006  # fraction of axes extent


@dataclass
class BackendStyle:
    """Style bucket a backend resolves when rendering. Populated by the
    core from user overrides (set_diag_style, grid_diagonal kwargs)."""
    major_line: dict = field(default_factory=dict)
    minor_line: dict = field(default_factory=dict)
    label: dict = field(default_factory=dict)
    axis_title: dict = field(default_factory=dict)


class Backend(ABC):
    """Abstract rendering surface the core talks to.

    Backends are responsible for:

    1. Providing axis limits (log-log frequency / pseudo-velocity).
    2. Translating data coords to pixel coords for rotation / density math.
    3. Drawing / updating artist pools for each family + role.
    4. Wiring a rescale callback so interactive zoom / pan / resize
       triggers a core rebuild (without a full re-render of user data).
    5. Optionally measuring rendered label widths so along-line offsets
       can be computed before drawing.
    """

    # ---- viewport --------------------------------------------------------

    @abstractmethod
    def get_xlim(self) -> tuple[float, float]:
        """Current x-axis limits (data coords)."""

    @abstractmethod
    def get_ylim(self) -> tuple[float, float]:
        """Current y-axis limits (data coords)."""

    @abstractmethod
    def is_log_log(self) -> bool:
        """True if both axes are logarithmic — diagonals are meaningless
        on linear axes, so the core will short-circuit when this is
        False."""

    # ---- coordinate transforms ------------------------------------------

    @abstractmethod
    def data_to_pixel(self, points):
        """``points`` is an (N, 2) array-like of (x, y) data coords.
        Returns an (N, 2) numpy array of display-space pixel coords.
        Used by the core for angle computation and density filtering."""

    def measure_text_width_pt(self, text: str, style_key: str = "label") -> float:
        """Optional: return rendered width of ``text`` in points.

        Used for placing inside-plot fallback labels so the text box
        sits centered on the line with its full width clear of the
        spine. Backends that can't measure should return a sane default
        (the core treats a None return as ~30 pt)."""
        return 30.0

    # ---- style ----------------------------------------------------------

    @abstractmethod
    def apply_style(self, style: BackendStyle) -> None:
        """Commit a bundle of resolved style dicts so subsequent draw
        calls pick them up. Called once per rebuild."""

    # ---- drawing --------------------------------------------------------

    @abstractmethod
    def set_lines(
        self,
        family: DiagramFamily,
        segments: list[tuple[tuple[float, float], tuple[float, float]]],
        per_line_style: list[dict] | None = None,
    ) -> None:
        """Render the full list of diagonal gridlines for ``family``.

        ``segments`` is a list of ``((x0, y0), (x1, y1))`` endpoints in
        data coords. ``per_line_style`` is a per-segment list of style
        dicts for tiered major/minor rendering (or ``None`` to use a
        uniform style pulled from :meth:`apply_style`).
        """

    @abstractmethod
    def set_labels(
        self,
        family: DiagramFamily,
        role: str,
        items: list[LabelItem],
    ) -> None:
        """Render the full list of labels for ``(family, role)``.

        Role is one of ``'midpoint'``, ``'edge'``, ``'fallback'``,
        ``'axis_title'``. The backend should grow / shrink its pool so
        ``len(items)`` is the visible count — no stale artists.
        """

    @abstractmethod
    def set_ticks(
        self,
        family: DiagramFamily,
        items: list[TickItem],
    ) -> None:
        """Render the full list of edge ticks for ``family``."""

    # ---- interactivity --------------------------------------------------

    @abstractmethod
    def connect_rescale(self, callback: Callable[[], None]) -> None:
        """Install ``callback`` to fire whenever the viewport changes —
        zoom, pan, resize, aspect-ratio toggle, etc. The callback is
        invoked with zero args; it's up to the backend to debounce /
        batch if needed."""

    def request_redraw(self) -> None:
        """Nudge the backend to flush pending updates. Optional — some
        backends redraw eagerly on every ``set_*`` call."""
        pass

    # ---- introspection (tests / debug) ----------------------------------

    def describe_artists(self) -> dict[str, Any]:
        """Return a dict summary of current artist counts per
        ``(family, role)``. Used by tests and by the matplotlib TripartiteAxes
        properties (``diag_line_count`` etc)."""
        return {}
