"""Circular statistics for resonant arguments.

Why this is its own module
--------------------------
A resonant argument lives on a circle, and the standard way of summarising one
-- "how far does it swing?" -- is a circular statistic, not a linear one. Doing
it linearly produces a specific, silent failure that this project already hit.

The broken version, which shipped in the Pluto rung's precondition::

    wrapped = np.where(phi > 180.0, phi - 360.0, phi)
    amplitude = wrapped.max() - wrapped.min()

For an argument librating about **0** that is fine. For one librating about
**180** -- which is exactly where a 3:2 mean-motion resonance sits -- the branch
cut runs straight through the middle of the libration. Samples at 175 and 185
degrees map to +175 and -175, and the reported span is 350 degrees: a
comfortably resonant orbit reported as circulating.

That is what happened. Pluto's argument, librating at 141.8 degrees peak to
peak about a centre of 178.5, was reported as spanning 360 degrees, and the
conclusion drawn from it -- that tabulated J2000 elements could not represent
the resonance -- was wrong. The algorithm below is due to Codex, which
replaced the broken wrap while implementing the rung; this module exists to put
it somewhere importable without a Skyfield dependency, and to test it against
the failure it fixes.

The general lesson is the same one the validation ladder exists for: a check
that has never been run against a case with a known answer is not evidence. The
precondition had never been run against a system known to be in resonance.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

__all__ = ["minimum_circular_span_degrees", "naive_linear_span_degrees"]


def minimum_circular_span_degrees(angles: Sequence[float]) -> tuple[float, float]:
    """Shortest arc containing every sample, and that arc's centre.

    Sort the angles onto [0, 360), find the largest empty gap between
    consecutive samples (treating the wrap as one more gap), and the occupied
    arc is the complement of that gap. This is orientation-free: it gives the
    same answer wherever the libration happens to be centred, which is the whole
    point.

    Returns
    -------
    ``(span_degrees, centre_degrees)``. A span near 360 means the argument
    circulates; a span comfortably below it means libration, and half the span
    is the libration amplitude about the centre.
    """

    values = np.asarray(angles, dtype=float)
    if values.ndim != 1:
        raise ValueError("angles must be one-dimensional")
    if values.size == 0:
        raise ValueError("at least one angular sample is required")
    if not np.all(np.isfinite(values)):
        raise ValueError("angular samples must be finite")

    values = np.sort(np.mod(values, 360.0))
    if values.size == 1:
        return 0.0, float(values[0])

    gaps = np.diff(np.concatenate((values, values[:1] + 360.0)))
    largest = int(np.argmax(gaps))
    span = 360.0 - float(gaps[largest])
    arc_start = float(values[(largest + 1) % values.size])
    centre = (arc_start + 0.5 * span) % 360.0
    return span, centre


def naive_linear_span_degrees(angles: Sequence[float]) -> float:
    """The broken statistic, kept so the regression tests can demonstrate it.

    Do not use this for anything. It exists to be compared against
    :func:`minimum_circular_span_degrees` in the tests, so the failure mode
    stays visible after the fix.
    """

    values = np.asarray(angles, dtype=float) % 360.0
    wrapped = np.where(values > 180.0, values - 360.0, values)
    return float(wrapped.max() - wrapped.min())
