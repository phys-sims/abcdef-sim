from __future__ import annotations

import math
from dataclasses import dataclass

from abcdef_sim.physics.geometry.frames import LocalFrame1D


@dataclass(frozen=True, slots=True)
class ChiefRayGeometryState:
    point_x_um: float
    point_z_um: float
    axis_angle_rad: float

    def frame(self) -> LocalFrame1D:
        return LocalFrame1D(
            origin_x_um=float(self.point_x_um),
            origin_z_um=float(self.point_z_um),
            axis_angle_rad=float(self.axis_angle_rad),
        )

    def reflected_about_normal(self, *, normal_angle_rad: float) -> ChiefRayGeometryState:
        axis = float(self.axis_angle_rad)
        normal = float(normal_angle_rad)
        reflected_axis = (2.0 * normal) - axis + math.pi
        return ChiefRayGeometryState(
            point_x_um=float(self.point_x_um),
            point_z_um=float(self.point_z_um),
            axis_angle_rad=math.atan2(math.sin(reflected_axis), math.cos(reflected_axis)),
        )
