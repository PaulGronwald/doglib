# triplot

Matplotlib projection for **tripartite (four-coordinate) log-log plots** used in
shock response spectra, earthquake engineering, and vibration analysis.

Frequency on X, pseudo-velocity on Y, constant-displacement lines at +45°,
constant-acceleration lines at −45°. Grid regenerates dynamically on zoom/pan —
no precomputed fixed range.

## Install

```bash
pip install -e .
```

## Use

```python
import matplotlib.pyplot as plt
import numpy as np
import triplot  # registers the projection

fig, ax = plt.subplots(subplot_kw={"projection": "tripartite"})
f = np.logspace(0, 3, 200)
v = 10 / (1 + (f / 50) ** 2) ** 0.5
ax.plot(f, v)
ax.grid_diagonal(True)
plt.show()
```
