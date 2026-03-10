from __future__ import annotations

import math

from abcdef_sim.physics.geometry.frames import LocalFrame1D
from abcdef_sim.physics.geometry.surfaces import SurfacePlane1D


def point_from_local_coordinates_um(
    frame: LocalFrame1D,
    *,
    x_um: float,
    z_um: float,
) -> tuple[float, float]:
    x_hat_x, x_hat_z = frame.x_hat
    z_hat_x, z_hat_z = frame.z_hat
    return (
        float(frame.origin_x_um) + (float(x_um) * x_hat_x) + (float(z_um) * z_hat_x),
        float(frame.origin_z_um) + (float(x_um) * x_hat_z) + (float(z_um) * z_hat_z),
    )


def ray_direction_from_local_slope(
    frame: LocalFrame1D,
    *,
    slope_dx_dz: float,
) -> tuple[float, float]:
    x_hat_x, x_hat_z = frame.x_hat
    z_hat_x, z_hat_z = frame.z_hat
    dir_x = z_hat_x + (float(slope_dx_dz) * x_hat_x)
    dir_z = z_hat_z + (float(slope_dx_dz) * x_hat_z)
    norm = math.hypot(dir_x, dir_z)
    if norm <= 0.0:
        raise ValueError("ray direction norm must be positive")
    return (dir_x / norm, dir_z / norm)


def ray_direction_from_x_prime(
    *,
    x_prime: float,
    refractive_index: float,
    axis_angle_rad: float,
) -> tuple[float, float]:
    frame = LocalFrame1D(origin_x_um=0.0, origin_z_um=0.0, axis_angle_rad=float(axis_angle_rad))
    return ray_direction_from_local_slope(
        frame,
        slope_dx_dz=float(x_prime) / float(refractive_index),
    )


def intersect_ray_with_plane_um(
    *,
    origin_x_um: float,
    origin_z_um: float,
    direction_x: float,
    direction_z: float,
    plane: SurfacePlane1D,
) -> float:
    normal_x, normal_z = plane.normal_unit
    numerator = normal_x * (float(plane.point_x_um) - float(origin_x_um)) + normal_z * (
        float(plane.point_z_um) - float(origin_z_um)
    )
    denominator = normal_x * float(direction_x) + normal_z * float(direction_z)
    if denominator <= 0.0:
        raise ValueError("ray misses or grazes the target plane")
    return numerator / denominator
