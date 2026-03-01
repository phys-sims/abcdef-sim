from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abcdef_sim.optics.base import ArrayLike, NDArrayF, Optic, RefractiveIndexFn
from abcdef_sim.utils.optics_builder import get_abcdef_matrices


@dataclass(slots=True, init=False)
class FreeSpace(Optic):
    """Free-space propagation optic."""

    def __init__(
        self,
        *,
        length: float = 0.0,
        name: str = "FreeSpace",
        instance_name: str = "inst0",
        _n_fn: RefractiveIndexFn | None = None,
    ) -> None:
        self.name = name
        self.instance_name = instance_name
        self._length = float(length)
        self._n_fn = _n_fn

    def matrix(self, omega: ArrayLike) -> NDArrayF:
        omega_arr = np.asarray(omega, dtype=np.float64)
        return get_abcdef_matrices(a=1.0, b=self.length, c=0.0, d=1.0, omega=omega_arr)
