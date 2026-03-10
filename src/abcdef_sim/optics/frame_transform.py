from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abcdef_sim.optics.base import ArrayLike, NDArrayF, Optic
from abcdef_sim.utils.optics_builder import get_abcdef_matrices


@dataclass(slots=True)
class FrameTransform(Optic):
    """Phase-free affine ray-coordinate transform in homogeneous ABCDEF form."""

    x_offset_um: float = 0.0
    x_prime_offset: float = 0.0
    x_prime_scale: float = 1.0

    def matrix(self, omega: ArrayLike, *, omega0: float | None = None) -> NDArrayF:
        del omega0
        omega_arr = np.asarray(omega, dtype=np.float64)
        return get_abcdef_matrices(
            a=1.0,
            b=0.0,
            c=0.0,
            d=float(self.x_prime_scale),
            omega=omega_arr,
            e_of_omega=lambda w: np.full_like(w, -float(self.x_offset_um), dtype=np.float64),
            f_of_omega=lambda w: np.full_like(w, -float(self.x_prime_offset), dtype=np.float64),
        )

    def cache_params(self) -> tuple:
        return (
            float(self.x_offset_um),
            float(self.x_prime_offset),
            float(self.x_prime_scale),
        )

    def phase_model(self) -> str:
        return "none"
