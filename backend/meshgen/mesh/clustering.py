"""1-D node-distribution functions for structured grids.

These map a normalised parameter ``s in [0, 1]`` to clustered node positions,
which is how a structured mesh concentrates points where the flow needs them
(the wall boundary layer, the leading edge, the shock).
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "uniform",
    "geometric",
    "tanh_one_sided",
    "tanh_two_sided",
    "distribute_wall_normal",
]


def uniform(n: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def geometric(n: int, ratio: float) -> np.ndarray:
    """Geometric distribution with successive-spacing ``ratio`` (>1 grows)."""
    if abs(ratio - 1.0) < 1e-9:
        return uniform(n)
    k = np.arange(n)
    d = (ratio ** k - 1.0) / (ratio ** (n - 1) - 1.0)
    return d


def _stretch_beta(n: int, first: float) -> float:
    """Find the tanh stretching parameter giving a target first spacing.

    ``first`` is the desired normalised first-cell size (delta/L).  Solved by
    bisection on the one-sided tanh spacing.
    """
    target = first
    lo, hi = 1e-6, 20.0

    def first_spacing(beta: float) -> float:
        s = tanh_one_sided(n, beta)
        return s[1] - s[0]

    flo = first_spacing(lo) - target
    fhi = first_spacing(hi) - target
    # first_spacing decreases as beta grows.
    if flo < 0:  # even the mildest stretch is finer than requested
        return lo
    if fhi > 0:  # cannot reach that fine a first cell
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = first_spacing(mid) - target
        if abs(fm) < 1e-12:
            return mid
        if fm > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def tanh_one_sided(n: int, beta: float) -> np.ndarray:
    """One-sided hyperbolic-tangent clustering toward ``s = 0`` (the wall).

    Vinokur-style stretching: larger ``beta`` clusters more tightly at 0.
    """
    if beta < 1e-6:
        return uniform(n)
    xi = np.linspace(0.0, 1.0, n)
    s = 1.0 + np.tanh(beta * (xi - 1.0)) / np.tanh(beta)
    s[0] = 0.0
    s[-1] = 1.0
    return s


def tanh_two_sided(n: int, beta: float) -> np.ndarray:
    """Two-sided clustering toward both ends (leading/trailing edges)."""
    if beta < 1e-6:
        return uniform(n)
    xi = np.linspace(0.0, 1.0, n)
    s = 0.5 * (1.0 + np.tanh(beta * (xi - 0.5)) / np.tanh(0.5 * beta))
    s[0] = 0.0
    s[-1] = 1.0
    return s


def distribute_wall_normal(
    n: int,
    total_height: float,
    first_cell: float,
) -> np.ndarray:
    """Wall-normal node positions from 0 to ``total_height``.

    Hits the requested ``first_cell`` height at the wall and smoothly stretches
    to the outer boundary -- the standard boundary-layer distribution driven by
    a ``y+`` target.

    Returns absolute distances (not normalised).
    """
    if n < 2:
        raise ValueError("need at least 2 wall-normal nodes")
    first_norm = first_cell / total_height
    # Clamp: first cell cannot exceed a uniform division.
    first_norm = min(first_norm, 1.0 / (n - 1))
    beta = _stretch_beta(n, first_norm)
    s = tanh_one_sided(n, beta)
    return s * total_height
