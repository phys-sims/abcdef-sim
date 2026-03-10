"""Geometry-backed plane-wave action helpers for Martinez-style phase accumulation.

The Martinez paper's ``phi0`` term is the plane-wave propagation phase accumulated
element by element. For a free-space segment in a dispersive geometry, the
physically relevant quantity is the travel time to the next surface, not just the
configured axial transport parameter used in the ABCDEF ``B`` term.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from abcdef_sim.physics.abcdef.phase_terms import SPEED_OF_LIGHT_UM_PER_FS

NDArrayF = npt.NDArray[np.float64]

__all__ = [
    "free_space_path_to_planar_surface_um",
    "group_delay_from_path_length_fs",
    "phase_from_group_delay_rad",
]


def free_space_path_to_planar_surface_um(
    *,
    axial_length_um: float,
    x_prime: object,
    refractive_index: object,
    surface_incidence_angle_rad: float,
) -> NDArrayF:
    """Return the geometric path length to the next planar surface.

    Assumptions:
      - The local ``z`` axis is aligned with the center-frequency chief ray for the
        segment, matching the Martinez construction.
      - ``axial_length_um`` is the center-ray travel distance from the current
        surface to the intersection point on the next planar surface.
      - ``surface_incidence_angle_rad`` is the signed center-ray incidence angle on
        the next surface measured in the current local frame.

    Geometry:
      - The off-frequency ray follows ``x(z) = (x' / n) * z`` in the local frame.
      - Intersecting that ray with a plane through ``(x=0, z=L)`` whose normal is
        rotated by ``gamma`` from ``+z`` gives the true path length used to compute
        travel time.
    """

    if axial_length_um < 0.0:
        raise ValueError("axial_length_um must be >= 0.")

    x_prime_arr = np.asarray(x_prime, dtype=np.float64).reshape(-1)
    n_arr = np.asarray(refractive_index, dtype=np.float64).reshape(-1)
    if x_prime_arr.shape != n_arr.shape:
        raise ValueError("x_prime and refractive_index must share one shape")
    if np.any(n_arr <= 0.0):
        raise ValueError("refractive_index must be strictly positive")

    slope = x_prime_arr / n_arr
    gamma = float(surface_incidence_angle_rad)
    denominator = np.cos(gamma) + slope * np.sin(gamma)
    if np.any(denominator <= 0.0):
        raise ValueError(
            "planar-surface intersection is invalid because the ray misses or grazes the plane"
        )
    return float(axial_length_um) * np.sqrt(1.0 + slope**2) / denominator


def group_delay_from_path_length_fs(path_length_um: object, refractive_index: object) -> NDArrayF:
    """Return the travel time ``tau(omega) = n(omega) * s(omega) / c``."""

    path_arr = np.asarray(path_length_um, dtype=np.float64).reshape(-1)
    n_arr = np.asarray(refractive_index, dtype=np.float64).reshape(-1)
    if path_arr.shape != n_arr.shape:
        raise ValueError("path_length_um and refractive_index must share one shape")
    return (n_arr * path_arr) / SPEED_OF_LIGHT_UM_PER_FS


def phase_from_group_delay_rad(
    omega: object,
    group_delay_fs: object,
    *,
    omega0_rad_per_fs: float,
) -> NDArrayF:
    """Integrate group delay to phase and center it so ``phi(omega0) = 0``.

    The physical relation is ``dphi / domega = tau(omega)``. A cumulative
    trapezoidal integral is sufficient because only derivatives above affine order
    matter for the dispersion fit.
    """

    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    tau_arr = np.asarray(group_delay_fs, dtype=np.float64).reshape(-1)
    if omega_arr.shape != tau_arr.shape:
        raise ValueError("omega and group_delay_fs must share one shape")
    if omega_arr.size == 0:
        raise ValueError("omega must contain at least one sample")
    if omega_arr.size == 1:
        return np.zeros_like(omega_arr)

    order = np.argsort(omega_arr)
    omega_sorted = omega_arr[order]
    tau_sorted = tau_arr[order]
    delta_omega = np.diff(omega_sorted)
    trapezoids = 0.5 * (tau_sorted[1:] + tau_sorted[:-1]) * delta_omega
    phase_sorted = np.concatenate(
        [np.zeros(1, dtype=np.float64), np.cumsum(trapezoids, dtype=np.float64)]
    )

    center_index_sorted = int(np.argmin(np.abs(omega_sorted - float(omega0_rad_per_fs))))
    phase_sorted = phase_sorted - phase_sorted[center_index_sorted]

    phase = np.empty_like(phase_sorted)
    phase[order] = phase_sorted
    return phase
