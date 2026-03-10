from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abcdef_sim.optics.base import ArrayLike, NDArrayF, Optic

_C_UM_PER_FS = 0.299792458


@dataclass(slots=True)
class Grating(Optic):
    """Single planar grating primitive in Martinez ABCDEF form."""

    line_density_lpmm: float = 1200.0
    incidence_angle_deg: float = 35.0
    diffraction_order: int = -1
    immersion_refractive_index: float = 1.0

    def matrix(self, omega: ArrayLike, *, omega0: float | None = None) -> NDArrayF:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        if omega_arr.size == 0:
            return np.zeros((0, 3, 3), dtype=np.float64)

        omega_ref = float(np.mean(omega_arr) if omega0 is None else omega0)
        if omega_ref <= 0.0:
            raise ValueError("Grating requires omega0 > 0.")

        d_um = 1000.0 / float(self.line_density_lpmm)
        theta_i_rad = np.deg2rad(float(self.incidence_angle_deg))
        m = float(self.diffraction_order)
        n_medium = float(self.immersion_refractive_index)

        theta_d_rad = _diffraction_angle_rad(
            omega_arr,
            period_um=d_um,
            incidence_angle_rad=theta_i_rad,
            diffraction_order=m,
        )
        cos_i = np.cos(theta_i_rad)
        cos_d = np.cos(theta_d_rad)
        if np.any(np.abs(cos_d) <= 1e-12) or abs(cos_i) <= 1e-12:
            raise ValueError("Grating geometry is singular because cos(theta) is near zero.")

        a = cos_d / cos_i
        d = cos_i / cos_d
        f = _grating_f_exact_local(
            omega_arr,
            omega_ref=omega_ref,
            period_um=d_um,
            incidence_angle_rad=theta_i_rad,
            diffraction_order=m,
            immersion_refractive_index=n_medium,
        )

        matrices = np.zeros((omega_arr.size, 3, 3), dtype=np.float64)
        matrices[:, 0, 0] = a
        matrices[:, 1, 1] = d
        matrices[:, 1, 2] = f
        matrices[:, 2, 2] = 1.0
        return matrices

    def n(self, omega: ArrayLike, *, omega0: float | None = None) -> NDArrayF:
        del omega0
        omega_arr = np.asarray(omega, dtype=np.float64)
        return np.full_like(omega_arr, self.immersion_refractive_index, dtype=np.float64)

    def cache_params(self) -> tuple:
        return (
            float(self.line_density_lpmm),
            float(self.incidence_angle_deg),
            int(self.diffraction_order),
            float(self.immersion_refractive_index),
        )

    def l2_cache_safe(self) -> bool:
        return False


def _diffraction_angle_rad(
    omega: NDArrayF,
    *,
    period_um: float,
    incidence_angle_rad: float,
    diffraction_order: float,
) -> NDArrayF:
    lambda_um = (2.0 * np.pi * _C_UM_PER_FS) / np.asarray(omega, dtype=np.float64)
    diff_arg = -diffraction_order * (lambda_um / period_um) - np.sin(incidence_angle_rad)
    if np.any((diff_arg < -1.0) | (diff_arg > 1.0)):
        raise ValueError("Grating geometry produces an invalid diffraction angle.")
    return np.arcsin(diff_arg)


def _grating_f_exact_local(
    omega: NDArrayF,
    *,
    omega_ref: float,
    period_um: float,
    incidence_angle_rad: float,
    diffraction_order: float,
    immersion_refractive_index: float,
) -> NDArrayF:
    omega_arr = np.asarray(omega, dtype=np.float64)
    theta_ref = _diffraction_angle_rad(
        np.asarray([omega_ref], dtype=np.float64),
        period_um=period_um,
        incidence_angle_rad=incidence_angle_rad,
        diffraction_order=diffraction_order,
    )[0]
    theta = _diffraction_angle_rad(
        omega_arr,
        period_um=period_um,
        incidence_angle_rad=incidence_angle_rad,
        diffraction_order=diffraction_order,
    )
    return (immersion_refractive_index**2) * (theta_ref - theta)
