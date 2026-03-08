from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

_C_UM_PER_FS = 0.299792458

NDArrayF = npt.NDArray[np.float64]

__all__ = [
    "TreacyAnalyticMetrics",
    "compute_treacy_analytic_metrics",
    "phase_from_treacy_dispersion",
]


@dataclass(frozen=True, slots=True)
class TreacyAnalyticMetrics:
    """Closed-form Treacy grating-pair dispersion metrics."""

    gdd_fs2: float
    tod_fs3: float
    line_density_lpmm: float
    period_um: float
    wavelength_nm: float
    wavelength_um: float
    incidence_angle_deg: float
    incidence_angle_rad: float
    littrow_angle_deg: float
    diffraction_angle_deg: float
    omega0_rad_per_fs: float
    n_passes: int
    separation_um: float
    diffraction_order: int


def phase_from_treacy_dispersion(
    delta_omega_rad_per_fs: npt.ArrayLike,
    *,
    gdd_fs2: float,
    tod_fs3: float,
) -> NDArrayF:
    """Evaluate the Treacy phase polynomial on an offset-frequency grid."""

    delta_omega = np.asarray(delta_omega_rad_per_fs, dtype=np.float64)
    phase = 0.5 * float(gdd_fs2) * delta_omega**2
    if float(tod_fs3) != 0.0:
        phase = phase + (1.0 / 6.0) * float(tod_fs3) * delta_omega**3
    return np.asarray(phase, dtype=np.float64)


def compute_treacy_analytic_metrics(
    *,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    wavelength_nm: float = 1030.0,
    diffraction_order: int = -1,
    n_passes: int = 2,
    include_tod: bool = True,
) -> TreacyAnalyticMetrics:
    """Return Treacy grating-pair GDD/TOD using the plane-wave analytic model."""

    line_density = float(line_density_lpmm)
    incidence_angle = float(incidence_angle_deg)
    separation = float(separation_um)
    wavelength = float(wavelength_nm)
    order = int(diffraction_order)
    passes = int(n_passes)

    if line_density <= 0.0:
        raise ValueError("line_density_lpmm must be > 0.")
    if separation < 0.0:
        raise ValueError("separation_um must be >= 0.")
    if wavelength <= 0.0:
        raise ValueError("wavelength_nm must be > 0.")
    if passes < 1:
        raise ValueError("n_passes must be >= 1.")

    wavelength_um = wavelength * 1e-3
    period_um = 1000.0 / line_density
    theta_i_rad = math.radians(incidence_angle)

    littrow_arg = wavelength_um / (2.0 * period_um)
    theta_l_rad = _safe_asin(littrow_arg, context="littrow angle")

    diffraction_geometry = -float(order) * (wavelength_um / period_um)
    diff_arg = diffraction_geometry - math.sin(theta_i_rad)
    theta_d_rad = _safe_asin(diff_arg, context="diffraction angle")

    gdd_bracket = 1.0 - diff_arg**2
    if gdd_bracket <= 0.0:
        raise ValueError(
            f"Invalid Treacy geometry: GDD radical argument must be > 0. Got {gdd_bracket:.12g}."
        )
    gdd_fs2 = -(
        (passes * float(order) ** 2 * separation * wavelength_um**3)
        / (2.0 * math.pi * _C_UM_PER_FS**2 * period_um**2)
    ) * gdd_bracket ** (-1.5)

    tod_den = gdd_bracket
    if tod_den <= 0.0:
        raise ValueError(
            f"Invalid Treacy geometry: TOD denominator must be > 0. Got {tod_den:.12g}."
        )
    tod_num = 1.0 + diffraction_geometry * math.sin(theta_i_rad) - math.sin(theta_i_rad) ** 2
    tod_fs3 = (
        -((3.0 * wavelength_um) / (2.0 * math.pi * _C_UM_PER_FS)) * (tod_num / tod_den) * gdd_fs2
    )
    if not include_tod:
        tod_fs3 = 0.0

    omega0_rad_per_fs = 2.0 * math.pi * _C_UM_PER_FS / wavelength_um

    return TreacyAnalyticMetrics(
        gdd_fs2=float(gdd_fs2),
        tod_fs3=float(tod_fs3),
        line_density_lpmm=line_density,
        period_um=float(period_um),
        wavelength_nm=wavelength,
        wavelength_um=float(wavelength_um),
        incidence_angle_deg=incidence_angle,
        incidence_angle_rad=float(theta_i_rad),
        littrow_angle_deg=float(math.degrees(theta_l_rad)),
        diffraction_angle_deg=float(math.degrees(theta_d_rad)),
        omega0_rad_per_fs=float(omega0_rad_per_fs),
        n_passes=passes,
        separation_um=separation,
        diffraction_order=order,
    )


def _safe_asin(arg: float, *, context: str) -> float:
    if arg < -1.0 or arg > 1.0:
        raise ValueError(
            f"Invalid asin domain for {context}: arg={arg:.12g} (must be within [-1, 1])."
        )
    return math.asin(arg)
