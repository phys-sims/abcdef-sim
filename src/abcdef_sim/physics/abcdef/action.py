"""Geometry-backed plane-wave action helpers for Martinez phase accumulation.

Martinez Eq. (24) defines ``phi0`` as the plane-wave propagation phase added
element by element. For a free-space segment bounded by explicit surfaces, the
relevant quantity is the travel time to the next surface rather than only the
paraxial transport parameter used in ``B``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from abcdef_sim.physics.abcdef.phase_terms import SPEED_OF_LIGHT_UM_PER_FS
from abcdef_sim.physics.geometry.frames import LocalFrame1D
from abcdef_sim.physics.geometry.intersection import (
    intersect_ray_with_plane_um,
    point_from_local_coordinates_um,
    ray_direction_from_x_prime,
)
from abcdef_sim.physics.geometry.surfaces import SurfacePlane1D

NDArrayF = npt.NDArray[np.float64]

__all__ = [
    "center_ray_path_to_surface_um",
    "free_space_path_to_surface_intersection_um",
    "path_length_to_surface_intersection_um",
    "group_delay_from_path_length_fs",
    "phase_from_group_delay_rad",
]


def center_ray_path_to_surface_um(
    *,
    frame: LocalFrame1D,
    plane: SurfacePlane1D,
) -> float:
    return float(
        intersect_ray_with_plane_um(
            origin_x_um=float(frame.origin_x_um),
            origin_z_um=float(frame.origin_z_um),
            direction_x=frame.z_hat[0],
            direction_z=frame.z_hat[1],
            plane=plane,
        )
    )


def free_space_path_to_surface_intersection_um(
    *,
    x_um: object,
    x_prime: object,
    refractive_index: object,
    frame: LocalFrame1D,
    plane: SurfacePlane1D,
) -> NDArrayF:
    """Return the geometric path length from the segment entrance to ``plane``."""

    x_arr = np.asarray(x_um, dtype=np.float64).reshape(-1)
    x_prime_arr = np.asarray(x_prime, dtype=np.float64).reshape(-1)
    n_arr = np.asarray(refractive_index, dtype=np.float64).reshape(-1)
    if x_arr.shape != x_prime_arr.shape or x_arr.shape != n_arr.shape:
        raise ValueError("x_um, x_prime, and refractive_index must share one shape")
    if np.any(n_arr <= 0.0):
        raise ValueError("refractive_index must be strictly positive")

    paths = np.empty_like(x_arr)
    for idx, (x_i, x_prime_i, n_i) in enumerate(zip(x_arr, x_prime_arr, n_arr, strict=True)):
        origin_x, origin_z = point_from_local_coordinates_um(frame, x_um=float(x_i), z_um=0.0)
        direction_x, direction_z = ray_direction_from_x_prime(
            x_prime=float(x_prime_i),
            refractive_index=float(n_i),
            axis_angle_rad=float(frame.axis_angle_rad),
        )
        paths[idx] = intersect_ray_with_plane_um(
            origin_x_um=origin_x,
            origin_z_um=origin_z,
            direction_x=direction_x,
            direction_z=direction_z,
            plane=plane,
        )
    return paths


def path_length_to_surface_intersection_um(
    *,
    x_um: object,
    x_prime: object,
    refractive_index: object,
    frame: LocalFrame1D,
    plane: SurfacePlane1D,
) -> NDArrayF:
    return free_space_path_to_surface_intersection_um(
        x_um=x_um,
        x_prime=x_prime,
        refractive_index=refractive_index,
        frame=frame,
        plane=plane,
    )


def group_delay_from_path_length_fs(path_length_um: object, refractive_index: object) -> NDArrayF:
    """Return travel time ``tau(omega) = n(omega) s(omega) / c``."""

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
    """Integrate ``dphi / domega = tau(omega)`` and center the phase at ``omega0``."""

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
    phase_sorted = np.concatenate(
        [
            np.zeros(1, dtype=np.float64),
            np.cumsum(0.5 * (tau_sorted[1:] + tau_sorted[:-1]) * delta_omega, dtype=np.float64),
        ]
    )
    center_idx = int(np.argmin(np.abs(omega_sorted - float(omega0_rad_per_fs))))
    phase_sorted = phase_sorted - phase_sorted[center_idx]

    phase = np.empty_like(phase_sorted)
    phase[order] = phase_sorted
    return phase
