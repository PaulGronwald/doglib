"""Interactive demo of user-anchored isolines.

Two isoline flavours coexist on this plot:

    (A) ax.add_isoline(family, value, label=...)

        A **full-span** line — always reaches both viewport edges
        regardless of zoom level. The clipped segment recomputes on
        every draw so the line never "stops" mid-plot; it's effectively
        an axline-equivalent for constant-d / constant-a / constant-v
        geometry on LOG axes, which matplotlib's own ``ax.axline`` can't
        handle correctly (axline interpolates in linear data space).

        A mirror-spine tick marks the value on the opposite spine from
        the grid's own tick family:

            disp   grid: top     -> user: bottom  (fallback: left)
            accel  grid: right   -> user: left    (fallback: top)
            vel    grid: (left)  -> user: left

        The tick is drawn as a short ROTATED TANGENT at the crossing —
        visually a little extension of the line itself, not a straight
        perpendicular notch. Rotation tracks the on-screen slope so it
        stays coherent at any aspect / zoom.

        Labels use a light white glyph halo (patheffects stroke) for
        readability over the grid, and inherit font family / size from
        matplotlib's rcParams by default — set a project-wide font
        once and every triplot element picks it up. Override
        per-isoline via ``label_style={...}``.

    (B) ax.add_span_isoline(family, value, f_range=(lo, hi), label=...)

        A **finite** line bounded on both ends by ``f_range``, with a
        text label that sticks to the geometric midpoint of whatever
        portion is currently visible. Pan so only part of the line is
        in view and the label slides along to stay centered on the
        visible middle — "squish" behaviour. Pan so the line leaves the
        viewport entirely and the label hides (so you don't see phantom
        text over empty grid).

Both return matplotlib artists directly — ``.line``, ``.tick``,
``.label`` — so you can restyle in place without any triplot-specific
API::

    iso.line.set_linestyle('--')
    iso.line.set_color('#2c3e50')
    iso.label.set_fontsize(11)

Controls (powered by the TripartiteAxes default bindings):

  scroll wheel     zoom at cursor
  middle-drag      pan in log space
  toolbar          zoom-to-rect / pan / home still work
  Q                close window

Run::

    python examples/isolines_interactive.py
"""
from __future__ import annotations

import matplotlib

# Prefer a native interactive backend so the window pops up rather than
# rendering to PNG. If none is available the call falls through and you
# get whatever backend pyplot picked.
for _b in ("TkAgg", "Qt5Agg", "QtAgg"):
    try:
        matplotlib.use(_b, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import numpy as np

import triplot  # noqa: F401 — registers projection


def srs(f, fn=30.0, zeta=0.05):
    """Simple-ish SRS-ish curve for demo purposes."""
    r = f / fn
    mag = 1.0 / np.sqrt((1 - r ** 2) ** 2 + (2 * zeta * r) ** 2)
    return mag * (fn * 0.5)


def main():
    f = np.logspace(0, 3, 400)
    pv = srs(f, fn=30.0, zeta=0.05)

    fig, ax = triplot.plot(f, pv, figsize=(11, 9))
    ax.set_title("triplot — user isolines (scroll=zoom, middle-drag=pan)")

    # ------------------------------------------------------------------
    # (A) full-span isolines — viewport-spanning with mirror-spine ticks
    # ------------------------------------------------------------------
    # Design criterion spec from the API: these are the permanent
    # "constraint lines" a caller overlays on top of a plot — allowable
    # stress, code-limit acceleration, headroom displacement, etc. They
    # always fill the visible viewport because the segment is recomputed
    # from the current xlim/ylim on every draw.
    d_iso = ax.add_isoline(
        "disp", 0.5,
        label="d = 0.5 in",
        line_style={"color": "#c0392b", "linewidth": 1.4, "linestyle": "-"},
        tick_style={"color": "#c0392b", "linewidth": 1.2},
    )
    a_iso = ax.add_isoline(
        "accel", 2.0,
        label="a = 2 g",
        line_style={"color": "#2980b9", "linewidth": 1.4, "linestyle": "-"},
        tick_style={"color": "#2980b9", "linewidth": 1.2},
    )
    v_iso = ax.add_isoline(
        "vel", 10.0,
        label="v = 10 in/s",
        line_style={"color": "#27ae60", "linewidth": 1.4, "linestyle": "-"},
        tick_style={"color": "#27ae60", "linewidth": 1.2},
    )

    # Direct artist restyle — no triplot API needed. This is why the
    # isoline API hands back the matplotlib Line2D/Text objects rather
    # than wrapping them.
    d_iso.line.set_linestyle((0, (6, 3)))   # loose dashes

    # ------------------------------------------------------------------
    # (B) finite-span isolines — label rides the visible midpoint
    # ------------------------------------------------------------------
    # Use case: marking a regulatory or test range that only applies
    # over a specific frequency band. Pan the viewport left/right and
    # watch the label slide along the visible part of the line.
    ax.add_span_isoline(
        "accel", 5.0,
        f_range=(5.0, 200.0),
        label="high-g zone (5-200 Hz)",
        line_style={"color": "#8e44ad", "linewidth": 1.8},
        label_style={"color": "#8e44ad", "fontsize": 9, "fontweight": "bold"},
    )
    ax.add_span_isoline(
        "disp", 0.05,
        f_range=(20.0, 500.0),
        label="disp limit",
        line_style={"color": "#d35400", "linewidth": 1.6},
        label_style={"color": "#d35400", "fontsize": 9},
    )
    ax.add_span_isoline(
        "vel", 30.0,
        f_range=(10.0, 300.0),
        label="v = 30 in/s band",
        line_style={"color": "#16a085", "linewidth": 1.6},
        label_style={"color": "#16a085", "fontsize": 9, "fontstyle": "italic"},
    )

    print(f"backend: {matplotlib.get_backend()}")
    print("Scroll to zoom, middle-mouse drag to pan.")
    print("Try panning so the finite (B) lines half-leave the viewport")
    print("— their labels slide to stay on the visible midpoint.")
    plt.show()


if __name__ == "__main__":
    main()
