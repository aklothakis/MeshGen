"""Cone-derived (osculating-cone) hypersonic waverider generator.

A waverider is defined so that its leading edge lies exactly on a known shock
surface, "riding" its own shock with no spillage.  Here the shock is the
conical shock produced by the freestream Mach number and the design shock
angle; the flow field between shock and cone comes from the validated
Taylor-Maccoll solver.

Construction (coordinates: +x streamwise/freestream, +y spanwise, +z up; cone
apex at the origin, axis along +x):

1. The design shock is a cone of half-angle ``beta`` about the x-axis.  At the
   base plane ``x = L`` it traces a circle of radius ``Rs = L tan(beta)``.
2. The **upper (freestream) surface** is a horizontal plane ``z = -d`` where
   ``d = height_ratio * Rs``.  Because it is aligned with the freestream it
   carries no shock.  Its intersection with the shock cone is the **leading
   edge**.
3. From every leading-edge point a streamline is traced *through the conical
   flow field* down to the base plane -- the swept surface is the **lower
   (compression) surface** that generates the ride-along shock.
4. The **base** closes the volume between the two surfaces at ``x = L``.

The two surfaces come out as structured ``(n_span, n_stream, 3)`` arrays that
share the leading-edge row, which is exactly the topology the multiblock mesher
consumes.

References
----------
Nonweiler (1959); Rasmussen, *Hypersonic Flow* (cone-derived waveriders);
Sobieczky et al. (osculating-cone method).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid

from ..flow.taylor_maccoll import ConicalField, solve_taylor_maccoll

__all__ = ["WaveriderParams", "WaveriderGeometry", "generate_waverider"]


@dataclass
class WaveriderParams:
    """Design inputs for a cone-derived waverider."""

    mach: float = 6.0
    shock_angle_deg: float = 14.0
    length: float = 2.0            # m, nose-to-base
    height_ratio: float = 0.35     # d/Rs, flat-top offset below the cone axis (0..1)
    width_trim: float = 0.98       # fraction of the tip span to keep (<1 rounds tips)
    gamma: float = 1.4
    n_span: int = 61               # spanwise nodes across the full span
    n_stream: int = 41             # streamwise nodes from leading edge to base

    def validate(self) -> None:
        if self.mach <= 1.0:
            raise ValueError("mach must be supersonic (> 1)")
        if not (0.0 < self.height_ratio < 1.0):
            raise ValueError("height_ratio must be in (0, 1)")
        if self.length <= 0:
            raise ValueError("length must be positive")
        if self.n_span < 5 or self.n_stream < 3:
            raise ValueError("need n_span >= 5 and n_stream >= 3")


@dataclass
class WaveriderGeometry:
    """Generated waverider surfaces and derived design data."""

    params: WaveriderParams
    field: ConicalField
    upper: np.ndarray              # (n_span, n_stream, 3) freestream surface
    lower: np.ndarray              # (n_span, n_stream, 3) compression surface
    leading_edge: np.ndarray       # (n_span, 3)
    theta_c_deg: float
    shock_radius_base: float       # Rs at x = L
    volume: float
    planform_area: float

    @property
    def span(self) -> float:
        return float(self.leading_edge[:, 1].max() - self.leading_edge[:, 1].min())

    def base_curves(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(upper_base, lower_base)`` node rows at ``x = L``."""
        return self.upper[:, -1, :], self.lower[:, -1, :]

    def as_dict(self) -> dict:
        return {
            "mach": self.params.mach,
            "shock_angle_deg": self.params.shock_angle_deg,
            "cone_angle_deg": self.theta_c_deg,
            "surface_mach": self.field.mach_c,
            "length": self.params.length,
            "span": self.span,
            "shock_radius_base": self.shock_radius_base,
            "volume": self.volume,
            "planform_area": self.planform_area,
            "volumetric_efficiency": (
                self.volume ** (2.0 / 3.0) / self.planform_area
                if self.planform_area > 0 else 0.0
            ),
        }


def _streamline_scale(field: ConicalField) -> tuple[np.ndarray, np.ndarray]:
    """Tabulate ``g(theta) = r(theta)/r_shock`` along a conical streamline.

    Because the conical field is axisymmetric, every streamline shares the same
    normalised radial growth ``g``; only the shock-crossing radius differs per
    azimuth.  Integrates ``d(ln r)/dtheta = V_r / V_theta``.
    """
    theta = field.theta                    # decreasing: beta -> theta_c
    integrand = field.v_r / field.v_theta  # V_theta < 0 in the shock layer
    ln_r = cumulative_trapezoid(integrand, theta, initial=0.0)
    # Near the cone surface V_theta -> 0, so ln_r diverges; physical streamlines
    # reach the base plane long before then. Clip to keep g finite for interp.
    ln_r = np.clip(ln_r, -30.0, 30.0)
    g = np.exp(ln_r)                       # g(beta) = 1
    return theta, g


def generate_waverider(params: WaveriderParams) -> WaveriderGeometry:
    """Build a cone-derived waverider from Mach number and shock angle."""
    params.validate()
    beta = math.radians(params.shock_angle_deg)
    L = params.length

    field = solve_taylor_maccoll(params.mach, beta, params.gamma)
    theta_tab, g_tab = _streamline_scale(field)
    # Ensure ascending theta for np.interp.
    order = np.argsort(theta_tab)
    theta_asc = theta_tab[order]
    g_asc = g_tab[order]

    Rs = L * math.tan(beta)                       # shock-circle radius at base
    d = params.height_ratio * Rs                  # flat-top offset below axis
    y_tip = math.sqrt(max(Rs * Rs - d * d, 0.0)) * params.width_trim

    n_span, n_stream = params.n_span, params.n_stream
    y_le = np.linspace(-y_tip, y_tip, n_span)

    upper = np.zeros((n_span, n_stream, 3))
    lower = np.zeros((n_span, n_stream, 3))
    leading_edge = np.zeros((n_span, 3))

    def g_of_theta(th: float) -> float:
        return float(np.interp(th, theta_asc, g_asc))

    for i, y in enumerate(y_le):
        rho_le = math.hypot(y, d)                 # radial distance from axis at LE
        x_le = rho_le / math.tan(beta)            # LE lies on the shock cone
        r_le = math.hypot(x_le, rho_le)           # spherical radius at the shock
        psi = math.atan2(-d, y)                   # azimuth of this streamline
        leading_edge[i] = (x_le, y, -d)

        # ---- upper (freestream) surface: straight streamwise line to base ----
        xs_u = np.linspace(x_le, L, n_stream)
        upper[i, :, 0] = xs_u
        upper[i, :, 1] = y
        upper[i, :, 2] = -d

        # ---- lower (compression) surface: streamline through conical field ---
        # Find theta at the base (x = r_le g(theta) cos(theta) = L) by scanning.
        thetas = np.linspace(beta, field.theta_c, 240)
        gvals = np.interp(thetas, theta_asc, g_asc)
        x_of_theta = r_le * gvals * np.cos(thetas)
        # x increases as theta decreases; clamp target to achievable range.
        x_target = np.linspace(x_le, L, n_stream)
        x_target[-1] = min(L, x_of_theta.max())
        # Interpolate theta(x): x_of_theta is monotonic increasing as theta drops.
        th_sorted_idx = np.argsort(x_of_theta)
        x_sorted = x_of_theta[th_sorted_idx]
        th_sorted = thetas[th_sorted_idx]
        theta_stream = np.interp(x_target, x_sorted, th_sorted)

        r_stream = r_le * np.interp(theta_stream, theta_asc, g_asc)
        x_stream = r_stream * np.cos(theta_stream)
        rho_stream = r_stream * np.sin(theta_stream)
        lower[i, :, 0] = x_stream
        lower[i, :, 1] = rho_stream * math.cos(psi)
        lower[i, :, 2] = rho_stream * math.sin(psi)

    # Enforce a shared leading-edge row exactly (j = 0).
    lower[:, 0, :] = leading_edge
    upper[:, 0, :] = leading_edge

    volume, planform = _volume_and_planform(upper, lower)

    return WaveriderGeometry(
        params=params,
        field=field,
        upper=upper,
        lower=lower,
        leading_edge=leading_edge,
        theta_c_deg=field.theta_c_deg,
        shock_radius_base=Rs,
        volume=volume,
        planform_area=planform,
    )


def _volume_and_planform(upper: np.ndarray, lower: np.ndarray) -> tuple[float, float]:
    """Approximate enclosed volume and planform area from the surface grids."""
    # Planform: project lower surface onto x-y and sum quad areas.
    planform = 0.0
    volume = 0.0
    ni, nj, _ = lower.shape
    for i in range(ni - 1):
        for j in range(nj - 1):
            # Quad corners on lower surface.
            p = [lower[i, j], lower[i + 1, j], lower[i + 1, j + 1], lower[i, j + 1]]
            # Planform area via shoelace on (x, y).
            a = 0.0
            for k in range(4):
                x1, y1 = p[k][0], p[k][1]
                x2, y2 = p[(k + 1) % 4][0], p[(k + 1) % 4][1]
                a += x1 * y2 - x2 * y1
            planform += abs(a) * 0.5
            # Local thickness between upper and lower (z gap), times planform cell.
            th = 0.25 * sum(
                (upper[ii, jj, 2] - lower[ii, jj, 2])
                for ii, jj in [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
            )
            volume += abs(a) * 0.5 * abs(th)
    return volume, planform
