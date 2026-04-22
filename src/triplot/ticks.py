"""Nice-value tick picker for logarithmic diagonal axes.

Guarantees at least ``min_count`` visible tick values for any positive
[lo, hi] range — including aggressively zoomed or aspect-squished
viewports. The legacy ``_nice_decades`` picker silently returned an empty
list when no {1, 2, 5}x10^k value landed inside the requested range (e.g.
a narrow zoom to [1.1, 1.4] on the displacement axis would drop ALL
gridlines). That is the "rescaling stops at a specific level" bug — once
the viewport gets narrower than the coarsest subdivision, nothing renders.

Strategy:

1. Try increasingly dense subdivision ladders (1,2,5) → (1,2,3,5,7) →
   (1..9) → (1, 1.5, 2, 2.5, ..., 9.5) → finer still.
2. Accept the first ladder that produces >= ``min_count`` values inside
   [lo, hi].
3. If nothing clears the bar, fall back to linear nice-stepping between
   lo and hi directly (span is sub-decade, so log-nice loses meaning —
   treat it like a small linear range).
4. Always extend one ladder step beyond [lo, hi] to provide "overflow"
   ticks: edge-placed labels on lines whose crossing falls just outside
   the viewport still render a useful anchor during pan/zoom.

All math is pure; no rendering dependency.
"""
from __future__ import annotations

import math

# Progressive ladders, coarse -> fine. Each tuple is the set of mantissas
# (1 <= m < 10) emitted per decade.
_LADDERS: tuple[tuple[float, ...], ...] = (
    (1.0, 2.0, 5.0),
    (1.0, 2.0, 3.0, 5.0, 7.0),
    (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0),
    (
        1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ),
    (
        1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0,
        6.0, 7.0, 8.0, 9.0,
    ),
    (
        1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9,
        2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
    ),
)


def _enum_ladder(
    lo: float,
    hi: float,
    ladder: tuple[float, ...],
    decade_step: int = 1,
) -> list[float]:
    """All ``m * 10^k`` values in [lo, hi] for ``m`` in the ladder, ``k``
    stepped by ``decade_step``.

    ``decade_step > 1`` emits ticks every Nth decade only. Anchored at
    k=0 so the step is stable under pan — e.g. ``decade_step=5`` always
    emits 10^0, 10^5, 10^10, ... rather than whichever decade happens
    to fall at the start of the viewport.
    """
    if lo <= 0 or hi <= 0 or hi < lo or decade_step < 1:
        return []
    kmin = math.floor(math.log10(lo))
    kmax = math.ceil(math.log10(hi))
    out: list[float] = []
    # Start iteration at the nearest-below multiple of decade_step so the
    # tick grid stays aligned to absolute decade indices, not to the
    # viewport boundary.
    k_start = (kmin - 1) - ((kmin - 1) % decade_step)
    k_end = kmax + 2
    for k in range(k_start, k_end, decade_step):
        base = 10.0 ** k
        for m in ladder:
            v = m * base
            if lo <= v <= hi:
                out.append(v)
    return out


def _linear_nice_step(span: float) -> float:
    """Return a nice linear step size roughly span/5. Steps in {1,2,2.5,5}*10^k."""
    if span <= 0:
        return 1.0
    raw = span / 5.0
    exp = math.floor(math.log10(raw))
    frac = raw / (10.0 ** exp)
    for nice in (1.0, 2.0, 2.5, 5.0, 10.0):
        if frac <= nice:
            return nice * (10.0 ** exp)
    return 10.0 * (10.0 ** exp)


def _linear_fallback(lo: float, hi: float, min_count: int) -> list[float]:
    """Nice linear steps inside [lo, hi] — used when the span is sub-decade
    and log-nice ladders have all failed. Keeps refining the step until
    at least ``min_count`` values fit."""
    span = hi - lo
    step = _linear_nice_step(span)
    for _ in range(8):  # bounded refinement
        start = math.ceil(lo / step) * step
        vals: list[float] = []
        v = start
        while v <= hi + step * 1e-9:
            if lo <= v <= hi and v > 0:
                vals.append(v)
            v += step
        if len(vals) >= min_count:
            return vals
        step *= 0.5
    return vals  # whatever we have


def nice_values(
    lo: float,
    hi: float,
    min_count: int = 2,
    preferred: tuple[float, ...] | None = None,
    decade_step: int = 1,
) -> list[float]:
    """Pick "nice" tick values covering [lo, hi] with >= ``min_count`` entries.

    ``preferred`` is an optional starting ladder — if it already yields
    enough values, we use it verbatim (preserves caller style). Otherwise
    we progress through ``_LADDERS`` until the threshold is met.

    ``decade_step > 1`` is the "multiple decades between ticks" mode —
    used by the core at extreme zoom-outs (span > ~20 decades) where
    even one tick per decade becomes visual noise. The progressive
    ladder fallback still applies within the sampled decades.
    """
    if lo <= 0 or hi <= 0 or hi <= lo:
        return []

    if preferred is not None:
        vals = _enum_ladder(lo, hi, tuple(preferred), decade_step=decade_step)
        if len(vals) >= min_count:
            return vals

    for ladder in _LADDERS:
        vals = _enum_ladder(lo, hi, ladder, decade_step=decade_step)
        if len(vals) >= min_count:
            return vals

    # Sub-decade, ultra-narrow: ladders all saturated. Linear nice stepping.
    # decade_step is meaningless here — the span is sub-decade by definition.
    return _linear_fallback(lo, hi, min_count)


def overflow_pad(lo: float, hi: float) -> tuple[float, float]:
    """Pad the value range by a fraction of a decade on each side so the
    next tick just outside the viewport is also emitted — used by the
    edge-label / overflow-tick renderer to keep labels anchored while the
    user pans / resizes through the tick boundary."""
    if lo <= 0 or hi <= 0 or hi <= lo:
        return lo, hi
    span = math.log10(hi) - math.log10(lo)
    pad = max(span * 0.15, 0.08)  # ~15% of span, floor 0.08 decade
    return 10.0 ** (math.log10(lo) - pad), 10.0 ** (math.log10(hi) + pad)
