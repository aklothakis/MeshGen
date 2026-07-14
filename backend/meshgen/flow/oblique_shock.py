"""Compressible-flow relations for oblique shocks.

These closed-form relations are the backbone of the wedge-derived (caret)
waverider and of flow-aware domain sizing.  All angles are in radians unless a
name ends in ``_deg``.  Perfect-gas assumption with constant ``gamma``.

References
----------
Anderson, *Modern Compressible Flow*, 3rd ed., Ch. 4 (oblique shocks).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ShockState",
    "mach_angle",
    "theta_from_beta",
    "beta_from_theta",
    "max_deflection",
    "oblique_shock",
]


def mach_angle(mach: float) -> float:
    """Mach angle mu = asin(1/M)."""
    if mach <= 1.0:
        raise ValueError(f"Mach angle undefined for subsonic/sonic M={mach}")
    return math.asin(1.0 / mach)


def theta_from_beta(mach: float, beta: float, gamma: float = 1.4) -> float:
    """Flow deflection ``theta`` produced by a shock of wave angle ``beta``.

    This is the explicit direction of the theta-beta-Mach relation.
    Returns the deflection angle in radians (>= 0 for beta in (mu, pi/2)).
    """
    if mach <= 1.0:
        raise ValueError("theta-beta-M requires supersonic freestream")
    m2s2 = (mach * math.sin(beta)) ** 2
    num = m2s2 - 1.0
    den = mach ** 2 * (gamma + math.cos(2.0 * beta)) + 2.0
    tan_theta = 2.0 / math.tan(beta) * num / den
    return math.atan(tan_theta)


def max_deflection(mach: float, gamma: float = 1.4) -> tuple[float, float]:
    """Return ``(theta_max, beta_at_theta_max)`` for a given Mach number.

    Deflections above ``theta_max`` have no attached-shock (weak/strong)
    solution -- the shock detaches.  Found by a bounded 1-D maximisation of
    ``theta_from_beta`` over admissible wave angles.
    """
    mu = mach_angle(mach)
    lo, hi = mu + 1e-6, math.pi / 2 - 1e-6
    # Golden-section search for the maximum of theta(beta).
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - gr * (hi - lo)
    d = lo + gr * (hi - lo)
    fc = theta_from_beta(mach, c, gamma)
    fd = theta_from_beta(mach, d, gamma)
    for _ in range(200):
        if fc < fd:
            lo, c, fc = c, d, fd
            d = lo + gr * (hi - lo)
            fd = theta_from_beta(mach, d, gamma)
        else:
            hi, d, fd = d, c, fc
            c = hi - gr * (hi - lo)
            fc = theta_from_beta(mach, c, gamma)
        if hi - lo < 1e-10:
            break
    beta_star = 0.5 * (lo + hi)
    return theta_from_beta(mach, beta_star, gamma), beta_star


def beta_from_theta(
    mach: float,
    theta: float,
    gamma: float = 1.4,
    weak: bool = True,
) -> float:
    """Invert the theta-beta-Mach relation for the wave angle ``beta``.

    Parameters
    ----------
    weak:
        Select the weak-shock root (the physically relevant branch for an
        attached leading-edge shock) when ``True``; otherwise the strong root.

    Raises
    ------
    ValueError
        If ``theta`` exceeds the maximum attached-shock deflection.
    """
    if theta <= 0.0:
        return mach_angle(mach)
    theta_max, beta_star = max_deflection(mach, gamma)
    if theta > theta_max + 1e-9:
        raise ValueError(
            f"Deflection {math.degrees(theta):.3f} deg exceeds theta_max="
            f"{math.degrees(theta_max):.3f} deg for M={mach}: shock detaches."
        )
    mu = mach_angle(mach)
    if weak:
        lo, hi = mu + 1e-9, beta_star
    else:
        lo, hi = beta_star, math.pi / 2 - 1e-9

    # Bisection on f(beta) = theta_from_beta(beta) - theta, monotone on each branch.
    def f(b: float) -> float:
        return theta_from_beta(mach, b, gamma) - theta

    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0:
        # Numerical edge at theta ~ theta_max: return the peak.
        return beta_star
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if abs(fm) < 1e-12:
            return mid
        if flo * fm <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ShockState:
    """Post-shock state and jump ratios across an oblique shock."""

    mach1: float          # upstream Mach number
    beta: float           # wave angle (rad)
    theta: float          # flow deflection (rad)
    mach2: float          # downstream Mach number
    p2_p1: float          # static pressure ratio
    rho2_rho1: float      # density ratio
    t2_t1: float          # temperature ratio
    p02_p01: float        # total-pressure ratio (< 1, entropy rise)

    @property
    def beta_deg(self) -> float:
        return math.degrees(self.beta)

    @property
    def theta_deg(self) -> float:
        return math.degrees(self.theta)


def oblique_shock(mach: float, beta: float, gamma: float = 1.4) -> ShockState:
    """Full oblique-shock solution given upstream Mach and wave angle."""
    theta = theta_from_beta(mach, beta, gamma)
    mn1 = mach * math.sin(beta)
    mn1_2 = mn1 * mn1
    # Normal-shock jump on the shock-normal Mach component.
    mn2_2 = (1.0 + 0.5 * (gamma - 1.0) * mn1_2) / (gamma * mn1_2 - 0.5 * (gamma - 1.0))
    mn2 = math.sqrt(mn2_2)
    mach2 = mn2 / math.sin(beta - theta)
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (mn1_2 - 1.0)
    rho2_rho1 = (gamma + 1.0) * mn1_2 / (2.0 + (gamma - 1.0) * mn1_2)
    t2_t1 = p2_p1 / rho2_rho1
    # Total-pressure ratio across the shock.
    p02_p01 = (
        (rho2_rho1) ** (gamma / (gamma - 1.0))
        * (p2_p1) ** (-1.0 / (gamma - 1.0))
    )
    return ShockState(
        mach1=mach,
        beta=beta,
        theta=theta,
        mach2=mach2,
        p2_p1=p2_p1,
        rho2_rho1=rho2_rho1,
        t2_t1=t2_t1,
        p02_p01=p02_p01,
    )
