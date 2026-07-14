"""Freestream flow conditions and mesh-relevant derived quantities.

Turns a hypersonic operating point (Mach + altitude, or Mach + p/T) into the
quantities a structured mesh needs to be *flow-aware*:

* Reynolds number per unit length and over the body,
* the first wall-cell height that hits a target ``y+`` (boundary-layer
  resolution),
* an estimate of the boundary-layer thickness for wall-normal growth budgeting.

A minimal US Standard Atmosphere 1976 model is included so the user can specify
an altitude instead of raw thermodynamic state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = ["FlowConditions", "standard_atmosphere", "sutherland_viscosity"]

R_AIR = 287.058          # J/(kg K), specific gas constant for air
G0 = 9.80665             # m/s^2


def sutherland_viscosity(temperature: float) -> float:
    """Dynamic viscosity of air via Sutherland's law (SI, Pa s)."""
    mu_ref = 1.716e-5
    t_ref = 273.15
    s = 110.4
    return mu_ref * (temperature / t_ref) ** 1.5 * (t_ref + s) / (temperature + s)


def standard_atmosphere(altitude_m: float) -> tuple[float, float, float]:
    """US Standard Atmosphere 1976 up to 86 km: ``(pressure, temperature, density)``.

    Piecewise-linear temperature layers integrated for pressure; adequate for
    setting freestream conditions in an external-aerodynamics mesher.
    """
    # (base geopotential height m, base temp K, lapse rate K/m, base pressure Pa)
    layers = [
        (0.0, 288.15, -0.0065, 101325.0),
        (11000.0, 216.65, 0.0, 22632.06),
        (20000.0, 216.65, 0.001, 5474.889),
        (32000.0, 228.65, 0.0028, 868.0187),
        (47000.0, 270.65, 0.0, 110.9063),
        (51000.0, 270.65, -0.0028, 66.93887),
        (71000.0, 214.65, -0.002, 3.956420),
    ]
    h = max(0.0, min(altitude_m, 86000.0))
    hb, tb, lr, pb = layers[0]
    for i in range(len(layers)):
        h_base = layers[i][0]
        h_top = layers[i + 1][0] if i + 1 < len(layers) else 86000.0
        if h <= h_top or i == len(layers) - 1:
            hb, tb, lr, pb = layers[i]
            break
    temperature = tb + lr * (h - hb)
    if abs(lr) < 1e-12:
        pressure = pb * math.exp(-G0 * (h - hb) / (R_AIR * tb))
    else:
        pressure = pb * (temperature / tb) ** (-G0 / (lr * R_AIR))
    density = pressure / (R_AIR * temperature)
    return pressure, temperature, density


@dataclass(frozen=True)
class FlowConditions:
    """Freestream operating point plus derived aero/meshing quantities."""

    mach: float
    temperature: float       # K
    pressure: float          # Pa
    gamma: float = 1.4
    reference_length: float = 1.0   # m, body length used for Re

    # ---- constructors -------------------------------------------------
    @classmethod
    def from_altitude(
        cls,
        mach: float,
        altitude_m: float,
        reference_length: float = 1.0,
        gamma: float = 1.4,
    ) -> "FlowConditions":
        p, t, _ = standard_atmosphere(altitude_m)
        return cls(mach=mach, temperature=t, pressure=p,
                   gamma=gamma, reference_length=reference_length)

    # ---- thermodynamic state -----------------------------------------
    @property
    def density(self) -> float:
        return self.pressure / (R_AIR * self.temperature)

    @property
    def sound_speed(self) -> float:
        return math.sqrt(self.gamma * R_AIR * self.temperature)

    @property
    def velocity(self) -> float:
        return self.mach * self.sound_speed

    @property
    def viscosity(self) -> float:
        return sutherland_viscosity(self.temperature)

    @property
    def reynolds_per_length(self) -> float:
        return self.density * self.velocity / self.viscosity

    @property
    def reynolds(self) -> float:
        return self.reynolds_per_length * self.reference_length

    # ---- boundary-layer / mesh quantities ----------------------------
    def skin_friction(self, x: float | None = None) -> float:
        """Local turbulent flat-plate skin-friction coefficient (Schlichting).

        Uses ``Cf = 0.0592 Re_x^{-1/5}`` at station ``x`` (defaults to the
        reference length).  A crude but standard first estimate for sizing the
        wall cell; the hypersonic value is conservative (real Cf is lower).
        """
        x = self.reference_length if x is None else x
        re_x = max(self.reynolds_per_length * x, 1.0)
        return 0.0592 * re_x ** (-0.2)

    def first_cell_height(self, y_plus: float = 1.0, x: float | None = None) -> float:
        """Wall-normal height of the first cell for a target ``y+``.

        y1 = y+ * mu / (rho * u_tau), with u_tau = sqrt(tau_w / rho) and
        tau_w = Cf * 0.5 * rho * U^2.
        """
        cf = self.skin_friction(x)
        tau_w = cf * 0.5 * self.density * self.velocity ** 2
        u_tau = math.sqrt(tau_w / self.density)
        return y_plus * self.viscosity / (self.density * u_tau)

    def boundary_layer_thickness(self, x: float | None = None) -> float:
        """Turbulent boundary-layer thickness estimate ``delta = 0.37 x Re_x^{-1/5}``."""
        x = self.reference_length if x is None else x
        re_x = max(self.reynolds_per_length * x, 1.0)
        return 0.37 * x * re_x ** (-0.2)

    def summary(self) -> dict[str, float]:
        return {
            "mach": self.mach,
            "temperature_K": self.temperature,
            "pressure_Pa": self.pressure,
            "density_kg_m3": self.density,
            "sound_speed_m_s": self.sound_speed,
            "velocity_m_s": self.velocity,
            "viscosity_Pa_s": self.viscosity,
            "reynolds_per_m": self.reynolds_per_length,
            "reynolds_L": self.reynolds,
            "first_cell_height_yplus1_m": self.first_cell_height(1.0),
            "bl_thickness_m": self.boundary_layer_thickness(),
        }
