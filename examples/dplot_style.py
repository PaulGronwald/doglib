"""Reproduce the DPlot commercial tripartite layout:
   https://www.dplot.com/tripartite/shock-spectra-on-tripartite-grid.png

   - dense 1..9 per-decade diagonals
   - labels at m=1 and m=5 per decade, with unit suffix ("5 in.", "1 g")
   - no in-plot axis titles
   - wide range: 0.1-1000 Hz, 0.001-10 in/s
   - three damping curves plotted over the grid
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import triplot  # noqa: F401


def srs(f, fn, zeta):
    """Toy SRS amplitude for a SDOF with natural freq fn, damping zeta."""
    r = f / fn
    mag = 1.0 / np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)
    # fold into pseudo-velocity-like shape peaking around fn
    return mag * (fn * 0.5)


fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"projection": "tripartite", "style": "dplot"})
ax.set_xlim(0.1, 1000)
ax.set_ylim(0.001, 10)

f = np.logspace(-1, 3, 400)
fn = 3.0  # peak near 3 Hz like the reference
for zeta, color, label in [(0.001, "red", "Undamped"),
                            (0.05, "green", "5.00% damping"),
                            (0.10, "blue", "10.00% damping")]:
    v = srs(f, fn, zeta) * 0.2
    ax.plot(f, v, color=color, linewidth=1.3, label=label)

ax.set_title("Shock Spectra")
ax.set_xlabel("Frequency, Hz")
ax.set_ylabel("PseudoVelocity, in/sec")
ax.legend(loc="lower center", framealpha=0.95)

fig.savefig(r"C:\Users\pmarq\source\repos\triplot\examples\dplot_style.png", dpi=140, bbox_inches="tight")
print("wrote dplot_style.png")
