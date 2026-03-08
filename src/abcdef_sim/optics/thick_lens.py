from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abcdef_sim.optics.base import ArrayLike, NDArrayF, Optic
from abcdef_sim.physics.abcd.lenses import SellmeierMaterial, refractive_index_sellmeier
from abcdef_sim.physics.abcd.matrices import thick_lens

_C_UM_PER_FS = 0.299792458


def _theta_to_xprime_matrix(
    matrix_theta: NDArrayF,
    *,
    n_in: float,
    n_out: float,
) -> NDArrayF:
    out = np.zeros((3, 3), dtype=np.float64)
    out[0, 0] = matrix_theta[0, 0]
    out[0, 1] = matrix_theta[0, 1] / n_in
    out[1, 0] = n_out * matrix_theta[1, 0]
    out[1, 1] = (n_out / n_in) * matrix_theta[1, 1]
    out[2, 2] = 1.0
    return out


@dataclass(slots=True)
class ThickLens(Optic):
    """Thick lens runtime optic built on the validated ABCD lens layer."""

    R1: float | None = None
    R2: float | None = None
    n_in: float = 1.0
    n_out: float = 1.0
    refractive_index_model: float | SellmeierMaterial = 1.5

    def matrix(self, omega: ArrayLike, *, omega0: float | None = None) -> NDArrayF:
        del omega0
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        n_lens = self.n(omega_arr)

        matrices = np.zeros((omega_arr.size, 3, 3), dtype=np.float64)
        for idx, n_value in enumerate(n_lens):
            theta_matrix = thick_lens(
                n_lens=float(n_value),
                R1=self.R1,
                R2=self.R2,
                thickness=self.length,
                n_in=self.n_in,
                n_out=self.n_out,
            )
            matrices[idx] = _theta_to_xprime_matrix(
                theta_matrix,
                n_in=float(self.n_in),
                n_out=float(self.n_out),
            )
        return matrices

    def n(self, omega: ArrayLike, *, omega0: float | None = None) -> NDArrayF:
        del omega0
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        if isinstance(self.refractive_index_model, SellmeierMaterial):
            wavelength_um = (2.0 * np.pi * _C_UM_PER_FS) / omega_arr
            refractive_index = refractive_index_sellmeier(
                self.refractive_index_model,
                wavelength_um,
            )
            return np.asarray(refractive_index, dtype=np.float64).reshape(-1)
        return np.full_like(omega_arr, float(self.refractive_index_model), dtype=np.float64)

    def cache_params(self) -> tuple:
        if isinstance(self.refractive_index_model, SellmeierMaterial):
            refractive_index_key: object = (
                self.refractive_index_model.name,
                self.refractive_index_model.b_terms,
                self.refractive_index_model.c_terms,
            )
        else:
            refractive_index_key = float(self.refractive_index_model)

        return (
            self.R1,
            self.R2,
            float(self.n_in),
            float(self.n_out),
            refractive_index_key,
        )
