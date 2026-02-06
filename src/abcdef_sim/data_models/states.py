from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from phys_pipeline.v1.types import State, hash_ndarray, hash_small

NDArrayF = npt.NDArray[np.float64]


@dataclass(slots=True)
class RayState(State):
    """
    State carrying:
      - rays:   (N,3,1) column vectors (x, x', 1)^T
      - system: (N,3,3) cumulative system matrices for each w
    """

    rays: NDArrayF       # (N,3,1)
    system: NDArrayF     # (N,3,3)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rays = np.asarray(self.rays, dtype=float)
        self.system = np.asarray(self.system, dtype=float)

        if self.rays.ndim == 2 and self.rays.shape[1] == 3:
            # allow (N,3) -> (N,3,1)
            self.rays = self.rays[..., None]

        if self.rays.ndim != 3 or self.rays.shape[-2:] != (3, 1):
            raise ValueError(
                f"rays must have shape (N,3,1) or (N,3); got {self.rays.shape}"
            )

        if self.system.shape != self.rays.shape[:-1] + (3,):
            # rays: (N,3,1) -> (N,3)
            # system: should be (N,3,3)
            if self.system.ndim != 3 or self.system.shape[1:] != (3, 3):
                raise ValueError(
                    f"system must have shape (N,3,3); got {self.system.shape}"
                )

    # --- State API ---

    def deepcopy(self) -> RayState:
        return RayState(
            rays=self.rays.copy(),
            system=self.system.copy(),
            meta=copy.deepcopy(self.meta),
        )

    def hashable_repr(self) -> bytes:
        h = hashlib.sha256()
        h.update(hash_ndarray(self.rays))
        h.update(hash_ndarray(self.system))
        return h.digest()
