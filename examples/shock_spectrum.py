"""Render a synthetic shock response spectrum on a tripartite grid.

Run:  python examples/shock_spectrum.py
Output: shock_spectrum.png next to this script.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import triplot  # noqa: F401 — registers projection


def srs_curve(f, f_corner=50.0, peak_v=20.0):
    """Toy SRS-ish shape: velocity-flat through mid freqs, rolling off at ends."""
    return peak_v / np.sqrt(1.0 + (f / (3.0 * f_corner)) ** 2 + (f_corner / f) ** 2)


def main():
    f = np.logspace(0, 3, 400)
    v = srs_curve(f)

    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": "tripartite"})
    ax.set_xlim(1, 1000)
    ax.set_ylim(0.1, 100)
    ax.plot(f, v, linewidth=2.0, color="#c0392b", label="5% damping")
    ax.grid_diagonal(True)
    ax.set_damping(0.05)
    ax.set_title("Shock Response Spectrum — tripartite grid")
    ax.legend(loc="lower right")

    out = Path(__file__).with_name("shock_spectrum.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
