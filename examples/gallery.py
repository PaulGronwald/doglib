"""6-panel gallery showing different use cases of the tripartite axes:
  (1) single SRS curve, default seismic style
  (2) three damping ratios overlaid, DPlot style
  (3) SI units (earthquake / civil engineering range)
  (4) extreme wide range - dynamic grid over 10 decades
  (5) narrow zoom - single decade
  (6) shock (sparse) style - minimalist grid

Demonstrates that the axes is a general-purpose drop-in projection.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import triplot  # noqa: F401


def srs(f, fn, zeta):
    r = f / fn
    mag = 1.0 / np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)
    return mag * (fn * 0.5)


fig = plt.figure(figsize=(18, 12))

# (1) single SRS, seismic style -----------------------------------------------
ax1 = fig.add_subplot(2, 3, 1, projection="tripartite", style="seismic")
ax1.set_xlim(1, 1000); ax1.set_ylim(0.1, 100)
f = np.logspace(0, 3, 400)
ax1.plot(f, srs(f, 30, 0.05) * 0.4, color="#c0392b", lw=1.6)
ax1.set_title("(1) Seismic style - single SRS")

# (2) three damping curves, DPlot style ---------------------------------------
ax2 = fig.add_subplot(2, 3, 2, projection="tripartite", style="dplot")
ax2.set_xlim(0.1, 1000); ax2.set_ylim(0.001, 10)
f = np.logspace(-1, 3, 400)
for z, c, lbl in [(0.001, "red", "Undamped"), (0.05, "green", "5%"), (0.10, "blue", "10%")]:
    ax2.plot(f, srs(f, 3, z) * 0.2, color=c, lw=1.2, label=lbl)
ax2.set_title("(2) DPlot style - damping overlay")
ax2.legend(loc="lower center", fontsize=8)

# (3) SI units (civil engineering) --------------------------------------------
ax3 = fig.add_subplot(2, 3, 3, projection="tripartite", style="dplot", units="SI")
ax3.set_xlim(0.1, 100); ax3.set_ylim(0.01, 10)
f = np.logspace(-1, 2, 400)
ax3.plot(f, srs(f, 2, 0.05) * 0.5, color="#2980b9", lw=1.6)
ax3.set_title("(3) SI units - civil/earthquake range")

# (4) extreme wide range (10 decades X, 9 decades Y) --------------------------
ax4 = fig.add_subplot(2, 3, 4, projection="tripartite", style="dplot")
ax4.set_xlim(1e-3, 1e7); ax4.set_ylim(1e-4, 1e5)
ax4.set_title("(4) Extreme range - 10 decades X, 9 decades Y")

# (5) narrow zoom (one decade each) -------------------------------------------
ax5 = fig.add_subplot(2, 3, 5, projection="tripartite", style="seismic")
ax5.set_xlim(20, 80); ax5.set_ylim(5, 20)
f = np.linspace(20, 80, 200)
ax5.plot(f, srs(f, 30, 0.03) * 0.4, color="#8e44ad", lw=1.6)
ax5.set_title("(5) Narrow zoom - grid stays correct")

# (6) shock / sparse style ----------------------------------------------------
ax6 = fig.add_subplot(2, 3, 6, projection="tripartite", style="shock")
ax6.set_xlim(1, 1000); ax6.set_ylim(0.1, 100)
f = np.logspace(0, 3, 400)
ax6.plot(f, srs(f, 50, 0.05) * 0.4, color="#16a085", lw=1.6)
ax6.set_title("(6) Shock style - minimalist (1 line/decade)")

fig.suptitle("triplot gallery - 6 use cases of the tripartite projection", fontsize=14, y=1.00)
fig.tight_layout()
fig.savefig(r"gallery.png", dpi=110, bbox_inches="tight")
print("wrote gallery.png")
