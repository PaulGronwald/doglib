"""Interactive tripartite viewer.

Controls:
  scroll wheel      zoom at cursor
  left-click drag   pan
  toolbar buttons   zoom-to-rectangle / home / etc. still work

Close the window to exit.
"""
from __future__ import annotations

import math
import matplotlib

# Prefer an interactive backend on Windows.
for _b in ("TkAgg", "Qt5Agg", "QtAgg"):
    try:
        matplotlib.use(_b, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import numpy as np

import triplot  # noqa: F401 — registers projection


def srs_curve(f, fn=3.0, zeta=0.05):
    r = f / fn
    mag = 1.0 / np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)
    return mag * (fn * 0.5) * 0.2


def main():
    fig, ax = plt.subplots(
        figsize=(9, 9),
        subplot_kw={"projection": "tripartite", "style": "dplot"},
    )
    ax.set_xlim(0.1, 1000)
    ax.set_ylim(0.001, 10)

    f = np.logspace(-1, 3, 400)
    for zeta, color, label in [
        (0.001, "red", "Undamped"),
        (0.05, "green", "5.00% damping"),
        (0.10, "blue", "10.00% damping"),
    ]:
        ax.plot(f, srs_curve(f, zeta=zeta), color=color, linewidth=1.3, label=label)

    ax.set_title("Shock Spectra  --  scroll = zoom, drag = pan")
    ax.set_xlabel("Frequency, Hz")
    ax.set_ylabel("PseudoVelocity, in/sec")
    ax.legend(loc="lower center", framealpha=0.95)

    # --- scroll-wheel zoom centered on cursor ---------------------------------
    def on_scroll(event):
        if event.inaxes is not ax:
            return
        scale = 0.8 if event.button == "up" else 1.25
        x, y = event.xdata, event.ydata
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        lx, ly = math.log10(x), math.log10(y)
        lx0, lx1 = math.log10(x0), math.log10(x1)
        ly0, ly1 = math.log10(y0), math.log10(y1)
        ax.set_xlim(10 ** (lx - (lx - lx0) * scale), 10 ** (lx + (lx1 - lx) * scale))
        ax.set_ylim(10 ** (ly - (ly - ly0) * scale), 10 ** (ly + (ly1 - ly) * scale))
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("scroll_event", on_scroll)

    # --- left-drag pan (in log space) -----------------------------------------
    state = {"active": False, "x": None, "y": None}

    def on_press(event):
        if event.inaxes is not ax or event.button != 1:
            return
        # skip if toolbar is in zoom/pan mode
        tb = fig.canvas.manager.toolbar
        if tb is not None and getattr(tb, "mode", "") != "":
            return
        state["active"] = True
        state["x"] = event.xdata
        state["y"] = event.ydata

    def on_motion(event):
        if not state["active"] or event.inaxes is not ax:
            return
        if event.xdata is None or event.ydata is None:
            return
        # shift limits by (log_prev - log_now) so the grabbed data-point stays
        # under the cursor.
        dlx = math.log10(state["x"]) - math.log10(event.xdata)
        dly = math.log10(state["y"]) - math.log10(event.ydata)
        x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
        ax.set_xlim(10 ** (math.log10(x0) + dlx), 10 ** (math.log10(x1) + dlx))
        ax.set_ylim(10 ** (math.log10(y0) + dly), 10 ** (math.log10(y1) + dly))
        fig.canvas.draw_idle()

    def on_release(event):
        state["active"] = False

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("button_release_event", on_release)

    print(f"backend: {matplotlib.get_backend()}")
    print("Scroll to zoom, left-drag to pan. Close window to exit.")
    plt.show()


if __name__ == "__main__":
    main()
