"""Taylor-Maccoll conical-flow solver.

Supersonic flow over a right circular cone at zero incidence is governed by the
Taylor-Maccoll ordinary differential equation.  Given the freestream Mach
number and the conical *shock* angle, we integrate inward from the shock to the
cone surface, yielding the cone half-angle and the full flow field between the
shock and the body.  This field is the basis of the osculating-cone waverider:
streamlines traced through it define the compression (lower) surface.

Velocities are non-dimensionalised by the maximum theoretical speed
``V_max = sqrt(2 h0)`` so that ``V' = V / V_max`` lies in ``[0, 1]``.

References
----------
Anderson, *Modern Compressible Flow*, 3rd ed., Ch. 10 (conical flow).
Taylor & Maccoll (1933).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from .oblique_shock import oblique_shock

__all__ = ["ConicalField", "solve_taylor_maccoll", "mach_from_vprime"]


def vprime_from_mach(mach: float, gamma: float) -> float:
    """Non-dimensional total speed V' = V/V_max for a given local Mach."""
    m2 = mach * mach
    return math.sqrt(m2 / (m2 + 2.0 / (gamma - 1.0)))


def mach_from_vprime(vprime: float, gamma: float) -> float:
    """Inverse of :func:`vprime_from_mach`."""
    v2 = vprime * vprime
    v2 = min(v2, 1.0 - 1e-15)
    return math.sqrt(2.0 / (gamma - 1.0) * v2 / (1.0 - v2))


@dataclass(frozen=True)
class ConicalField:
    """Result of a Taylor-Maccoll integration for one conical shock."""

    mach_inf: float
    beta: float             # conical shock angle (rad)
    theta_c: float          # cone half-angle (rad)
    gamma: float
    theta: np.ndarray       # polar angle samples from shock (beta) -> cone (theta_c)
    v_r: np.ndarray         # non-dim radial velocity  V'_r(theta)
    v_theta: np.ndarray     # non-dim polar velocity   V'_theta(theta)
    mach_c: float           # surface (cone) Mach number

    @property
    def beta_deg(self) -> float:
        return math.degrees(self.beta)

    @property
    def theta_c_deg(self) -> float:
        return math.degrees(self.theta_c)

    def velocity_at(self, theta: float) -> tuple[float, float]:
        """Interpolate ``(V'_r, V'_theta)`` at polar angle ``theta`` (rad)."""
        # theta array is monotonically decreasing (shock -> cone); flip for interp.
        th = self.theta[::-1]
        vr = self.v_r[::-1]
        vt = self.v_theta[::-1]
        return (
            float(np.interp(theta, th, vr)),
            float(np.interp(theta, th, vt)),
        )


def _tm_rhs(theta: float, y: np.ndarray, gamma: float) -> list[float]:
    """Right-hand side of the Taylor-Maccoll ODE.

    State ``y = [V_r, V_theta]`` with ``V_theta = dV_r/dtheta``.
    """
    v_r, v_theta = y
    a2 = 0.5 * (gamma - 1.0) * (1.0 - v_r * v_r - v_theta * v_theta)  # (a/Vmax)^2
    denom = a2 - v_theta * v_theta
    # dV_r/dtheta = V_theta (definition of the polar component for conical flow)
    dvr = v_theta
    # dV_theta/dtheta from the Taylor-Maccoll equation, solved for the 2nd deriv.
    num = v_r * v_theta * v_theta - a2 * (2.0 * v_r + v_theta / math.tan(theta))
    dvtheta = num / denom
    return [dvr, dvtheta]


def solve_taylor_maccoll(
    mach_inf: float,
    beta: float,
    gamma: float = 1.4,
    n_samples: int = 400,
) -> ConicalField:
    """Integrate the Taylor-Maccoll equation from a conical shock to the cone.

    Parameters
    ----------
    mach_inf:
        Freestream Mach number (> 1).
    beta:
        Conical *shock* half-angle in radians.
    gamma:
        Ratio of specific heats.
    n_samples:
        Number of dense output samples across the shock layer.

    Returns
    -------
    ConicalField
        The cone half-angle and sampled flow field.
    """
    if mach_inf <= 1.0:
        raise ValueError("Taylor-Maccoll requires supersonic freestream")

    # State immediately behind the (locally oblique) conical shock.
    shock = oblique_shock(mach_inf, beta, gamma)
    delta = shock.theta                      # streamline deflection at the shock
    v2 = vprime_from_mach(shock.mach2, gamma)
    # Decompose the post-shock velocity into spherical (r, theta) components
    # about the cone axis at polar angle theta = beta.
    v_r0 = v2 * math.cos(beta - delta)
    v_theta0 = -v2 * math.sin(beta - delta)  # points toward the axis

    # Event: V_theta -> 0 marks the cone surface (flow becomes purely radial).
    # scipy forwards ``args`` to event callbacks too, so accept and ignore them.
    def surface_event(theta: float, y: np.ndarray, *_args: object) -> float:
        return y[1]

    surface_event.terminal = True
    surface_event.direction = 1.0  # V_theta rises from negative toward zero

    sol = solve_ivp(
        _tm_rhs,
        t_span=(beta, 1e-4),          # integrate toward the axis (decreasing theta)
        y0=[v_r0, v_theta0],
        args=(gamma,),
        events=surface_event,
        dense_output=True,
        rtol=1e-9,
        atol=1e-11,
        max_step=math.radians(0.25),
    )

    if not sol.t_events[0].size:
        raise RuntimeError(
            f"No cone surface found for M={mach_inf}, beta={math.degrees(beta):.2f} deg. "
            "The shock angle may be below the Mach angle (no conical shock)."
        )

    theta_c = float(sol.t_events[0][0])
    theta = np.linspace(beta, theta_c, n_samples)
    ys = sol.sol(theta)
    v_r = ys[0]
    v_theta = ys[1]
    v_surface = float(v_r[-1])
    mach_c = mach_from_vprime(v_surface, gamma)

    return ConicalField(
        mach_inf=mach_inf,
        beta=beta,
        theta_c=theta_c,
        gamma=gamma,
        theta=theta,
        v_r=v_r,
        v_theta=v_theta,
        mach_c=mach_c,
    )
