"""Overlay known reference points on the tripartite grid. If math is correct,
each marker sits exactly on its labeled diagonal.

References (imperial, g = 386.089 in/s^2):
  (100 Hz, 1 g)   -> v = g / (2*pi*f) = 0.6144 in/s      -- on the a=1g line
  (10  Hz, 1 in)  -> v = 2*pi*f*d    = 62.832 in/s       -- on the d=1in line
  (1   Hz, 1 in)  -> v = 6.283 in/s                       -- on the d=1in line
  (100 Hz, 10 g)  -> v = 6.144 in/s                       -- on the a=10g line
  (10  Hz, 0.1in) -> v = 6.283 in/s                       -- on the d=0.1in line
"""
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import triplot  # noqa: F401

TWO_PI = 2 * math.pi
G = 386.089

refs = [
    ("1 g @ 100Hz",   100.0, G / (TWO_PI * 100.0),   "accel=1g"),
    ("10 g @ 100Hz",  100.0, 10.0 * G / (TWO_PI * 100.0), "accel=10g"),
    ("1 in @ 10Hz",   10.0,  TWO_PI * 10.0 * 1.0,    "disp=1in"),
    ("1 in @ 1Hz",    1.0,   TWO_PI * 1.0 * 1.0,     "disp=1in"),
    ("0.1 in @ 10Hz", 10.0,  TWO_PI * 10.0 * 0.1,    "disp=0.1in"),
    ("0.01 g @ 10Hz", 10.0,  0.01 * G / (TWO_PI * 10.0), "accel=0.01g"),
]

fig, ax = plt.subplots(figsize=(9, 9), subplot_kw={"projection": "tripartite"})
ax.set_xlim(1, 1000)
ax.set_ylim(0.1, 100)
for name, f, v, expect in refs:
    ax.plot([f], [v], "o", color="red", markersize=8, zorder=10)
    ax.annotate(f"{name}\n({expect})", (f, v), fontsize=8,
                xytext=(6, 6), textcoords="offset points",
                color="red", zorder=11)
    print(f"{name:20s} f={f:>6.2f}  v={v:>10.4f}  -> expect on {expect}")

fig.canvas.draw()
fig.savefig(r"C:\Users\pmarq\source\repos\triplot\scripts\verify_lines.png", dpi=180)
print("wrote verify_lines.png")
