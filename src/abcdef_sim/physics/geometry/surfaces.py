from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SurfacePlane1D:
    point_x_um: float
    point_z_um: float
    normal_angle_rad: float

    @property
    def normal_unit(self) -> tuple[float, float]:
        angle = float(self.normal_angle_rad)
        return (math.sin(angle), math.cos(angle))
