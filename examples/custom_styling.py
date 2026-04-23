"""Full reusability demo for the `tripartite` projection.

Shows that the axes is a generic drop-in for *any* log-log frequency /
pseudo-velocity data, and every visual knob is user-tunable:

  * arbitrary user data (here: two analytic curves + one noisy measurement)
  * custom x/y limits
  * grid style preset  (`style="seismic" | "dplot" | "shock"`)
  * tiered line weights via `ax.set_diag_style(...)`
  * label font tuning
  * optional uniform-grid override via `ax.grid_diagonal(..., color=..., linewidth=...)`
  * standard matplotlib APIs still work (title, legend, annotate, axvline, ...)

Produces `examples/custom_styling.png`.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import triplot  # noqa: F401  -- registers the 'tripartite' projection


rng = np.random.default_rng(7)


def arbitrary_curve(f, peak_hz, zeta, scale):
    """Any user function returning (f, v) pairs on a log-log grid."""
    r = f / peak_hz
    mag = 1.0 / np.sqrt((1 - r**2) ** 2 + (2 * zeta * r) ** 2)
    return mag * (peak_hz * 0.5) * scale


fig = plt.figure(figsize=(16, 11))

# ---------------- (A) default tiered grid --------------------------------------
axA = fig.add_subplot(2, 2, 1, projection="tripartite", style="dplot")
axA.set_xlim(0.1, 1000)
axA.set_ylim(0.001, 10)
f = np.logspace(-1, 3, 400)
axA.plot(f, arbitrary_curve(f, 3.0, 0.05, 0.2), color="#c0392b", lw=1.4, label="analytic")
axA.set_title("(A) default tiered grid")
axA.legend(loc="lower center", fontsize=8)

# ---------------- (B) custom tier weights + bolder labels ----------------------
axB = fig.add_subplot(2, 2, 2, projection="tripartite", style="dplot")
axB.set_xlim(0.1, 1000)
axB.set_ylim(0.001, 10)
axB.set_diag_style(
    major_linewidth=1.4, major_color="#1a1a1a",
    minor_linewidth=0.25, minor_color="#bfbfbf",
    label_fontsize=9, label_color="#1a1a1a",
)
# three overlaid user curves
for peak, zeta, color, label in [
    (1.5, 0.02, "#8e44ad", "mode 1"),
    (5.0, 0.05, "#27ae60", "mode 2"),
    (25.0, 0.10, "#2980b9", "mode 3"),
]:
    axB.plot(f, arbitrary_curve(f, peak, zeta, 0.15), color=color, lw=1.3, label=label)
axB.set_title("(B) bold majors, faint minors, bigger labels")
axB.legend(loc="lower center", fontsize=8)

# ---------------- (C) uniform grid override (single weight, one color) ---------
axC = fig.add_subplot(2, 2, 3, projection="tripartite", style="seismic", units="SI")
axC.set_xlim(0.1, 100)
axC.set_ylim(0.01, 10)
# The kwargs to grid_diagonal() disable tiered mode and paint every gridline
# with the same weight/color -- useful for quick overlays or print.
axC.grid_diagonal(True, color="#d35400", linewidth=0.6, alpha=0.6)

# noisy user "measurement"
f2 = np.logspace(-1, 2, 120)
noise = 10 ** (rng.normal(0, 0.08, f2.size))
axC.plot(f2, arbitrary_curve(f2, 2.0, 0.05, 0.5) * noise,
         marker="o", ms=3, lw=0, color="#2c3e50", label="measurement")
axC.plot(f2, arbitrary_curve(f2, 2.0, 0.05, 0.5), color="#c0392b", lw=1.2, label="fit")
axC.set_title("(C) uniform grid override + SI units")
axC.legend(loc="lower center", fontsize=8)

# ---------------- (D) minimalist shock style + annotations ---------------------
axD = fig.add_subplot(2, 2, 4, projection="tripartite", style="shock")
axD.set_xlim(1, 1000)
axD.set_ylim(0.1, 100)
axD.set_diag_style(major_linewidth=1.0, major_color="#333")
peak = 50.0
axD.plot(f, arbitrary_curve(f, peak, 0.03, 0.4), color="#16a085", lw=1.6)
# standard matplotlib tools keep working -----------------------------------
axD.axvline(peak, color="k", ls=":", lw=0.8)
axD.annotate(f"peak f = {peak:g} Hz",
             xy=(peak, arbitrary_curve(np.array([peak]), peak, 0.03, 0.4)[0]),
             xytext=(peak * 2, 30),
             arrowprops=dict(arrowstyle="->", lw=0.8),
             fontsize=9)
axD.set_title("(D) shock style + annotate / axvline still work")

fig.suptitle("triplot -- reusable tripartite projection with user-tunable styling",
             fontsize=14)
fig.tight_layout()
import os
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_styling.png")
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"wrote {out}")
