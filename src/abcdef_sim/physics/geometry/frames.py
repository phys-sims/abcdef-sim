from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LocalFrame1D:
    origin_x_um: float
    origin_z_um: float
    axis_angle_rad: float

    @property
    def x_hat(self) -> tuple[float, float]:
        angle = float(self.axis_angle_rad)
        return (math.cos(angle), -math.sin(angle))

    @property
    def z_hat(self) -> tuple[float, float]:
        angle = float(self.axis_angle_rad)
        return (math.sin(angle), math.cos(angle))
