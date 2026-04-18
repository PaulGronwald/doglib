"""Interactive 6-panel gallery viewer - same content as examples/gallery.py
but opens in a TkAgg window instead of saving to PNG."""
from __future__ import annotations

import matplotlib
for _b in ("TkAgg", "Qt5Agg", "QtAgg"):
    try:
        matplotlib.use(_b, force=True)
        break
    except Exception:
        continue

import matplotlib.pyplot as plt
import numpy as np

import triplot  # noqa: F401


def srs(f, fn, zeta):
    r = f / fn
    mag = 1.0 / np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)
    return mag * (fn * 0.5)


fig = plt.figure(figsize=(18, 11))

ax1 = fig.add_subplot(2, 3, 1, projection="tripartite", style="seismic")
ax1.set_xlim(1, 1000); ax1.set_ylim(0.1, 100)
f = np.logspace(0, 3, 400)
ax1.plot(f, srs(f, 30, 0.05) * 0.4, color="#c0392b", lw=1.6)
ax1.set_title("(1) Seismic - single SRS")

ax2 = fig.add_subplot(2, 3, 2, projection="tripartite", style="dplot")
ax2.set_xlim(0.1, 1000); ax2.set_ylim(0.001, 10)
f = np.logspace(-1, 3, 400)
for z, c, lbl in [(0.001, "red", "Undamped"), (0.05, "green", "5%"), (0.10, "blue", "10%")]:
    ax2.plot(f, srs(f, 3, z) * 0.2, color=c, lw=1.2, label=lbl)
ax2.set_title("(2) DPlot - damping overlay")
ax2.legend(loc="lower center", fontsize=8)

ax3 = fig.add_subplot(2, 3, 3, projection="tripartite", style="dplot", units="SI")
ax3.set_xlim(0.1, 100); ax3.set_ylim(0.01, 10)
f = np.logspace(-1, 2, 400)
ax3.plot(f, srs(f, 2, 0.05) * 0.5, color="#2980b9", lw=1.6)
ax3.set_title("(3) SI units - civil range")

ax4 = fig.add_subplot(2, 3, 4, projection="tripartite", style="dplot")
ax4.set_xlim(1e-3, 1e7); ax4.set_ylim(1e-4, 1e5)
ax4.set_title("(4) Extreme range - 10 decades")

ax5 = fig.add_subplot(2, 3, 5, projection="tripartite", style="seismic")
ax5.set_xlim(20, 80); ax5.set_ylim(5, 20)
f = np.linspace(20, 80, 200)
ax5.plot(f, srs(f, 30, 0.03) * 0.4, color="#8e44ad", lw=1.6)
ax5.set_title("(5) Narrow zoom - single decade")

ax6 = fig.add_subplot(2, 3, 6, projection="tripartite", style="shock")
ax6.set_xlim(1, 1000); ax6.set_ylim(0.1, 100)
f = np.logspace(0, 3, 400)
ax6.plot(f, srs(f, 50, 0.05) * 0.4, color="#16a085", lw=1.6)
ax6.set_title("(6) Shock - minimalist")

fig.suptitle("triplot gallery - 6 use cases  (scroll = zoom, drag = pan)", fontsize=14)
fig.tight_layout()

# ---- interactive zoom / pan across ALL panels -------------------------------
import math

def on_scroll(event):
    ax = event.inaxes
    if ax is None or event.xdata is None:
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

state = {"ax": None, "x": None, "y": None, "bg": None}

def on_press(event):
    if event.inaxes is None or event.button != 1 or event.xdata is None:
        return
    tb = fig.canvas.manager.toolbar
    if tb is not None and getattr(tb, "mode", "") != "":
        return
    ax = event.inaxes
    state["ax"] = ax
    state["x"] = event.xdata
    state["y"] = event.ydata
    fig.canvas.draw()
    state["bg"] = fig.canvas.copy_from_bbox(fig.bbox)

def on_motion(event):
    ax = state["ax"]
    if ax is None or event.inaxes is not ax or event.xdata is None:
        return
    dlx = math.log10(state["x"]) - math.log10(event.xdata)
    dly = math.log10(state["y"]) - math.log10(event.ydata)
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    ax.set_xlim(10 ** (math.log10(x0) + dlx), 10 ** (math.log10(x1) + dlx))
    ax.set_ylim(10 ** (math.log10(y0) + dly), 10 ** (math.log10(y1) + dly))
    if state["bg"] is not None:
        fig.canvas.restore_region(state["bg"])
        ax.draw(fig.canvas.get_renderer())
        fig.canvas.blit(ax.bbox)
    else:
        fig.canvas.draw_idle()

def on_release(event):
    state["ax"] = None
    state["bg"] = None
    fig.canvas.draw_idle()

fig.canvas.mpl_connect("scroll_event", on_scroll)
fig.canvas.mpl_connect("button_press_event", on_press)
fig.canvas.mpl_connect("motion_notify_event", on_motion)
fig.canvas.mpl_connect("button_release_event", on_release)

print(f"backend: {matplotlib.get_backend()}")
print("Scroll on any panel to zoom, drag to pan. Close window to exit.")
plt.show()
