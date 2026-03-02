from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike
NDArrayF = npt.NDArray[np.float64]

RefractiveIndexFn = Callable[[NDArrayF], NDArrayF]


@dataclass(slots=True)
class Optic(ABC):
    """
    Runtime base class for a single optical element in the ABCDEF formalism.

    - matrix(ω): 3x3 ray-transfer matrix in homogeneous coords.
    - n(ω): refractive index (default vacuum n=1).
    - length: scalar geometric length (default 0).
    - name: label for debugging.
    - instance_name: distinguishes multiple instances of same optic.

    Caching support:
      - cache_key(): stable hashable identity for cache buckets
      - cache_params(): override in subclasses to include parameters affecting matrix/n
    """

    name: str = "optic"
    instance_name: str = "inst0"
    _length: float = 0.0
    _n_fn: RefractiveIndexFn | None = field(default=None, repr=False)

    @abstractmethod
    def matrix(self, omega: ArrayLike) -> NDArrayF:
        """Return the 3x3 transfer matrix for angular frequency omega.

        - omega can be scalar or array-like
        - Return shape: (..., 3, 3)
        """
        raise NotImplementedError

    def n(self, omega: ArrayLike) -> NDArrayF:
        """Refractive index vs angular frequency (defaults to n=1)."""
        omega_arr = np.asarray(omega, dtype=np.float64)

        if self._n_fn is None:
            return np.ones_like(omega_arr, dtype=np.float64)

        return self._n_fn(omega_arr)

    @property
    def length(self) -> float:
        """Geometric length [internal units]."""
        return self._length

    def cache_params(self) -> tuple:
        """Override in subclasses for parameters that change matrix/n behavior."""
        return ()

    def cache_key(self) -> Hashable:
        """Stable hashable identity for cache buckets."""
        return (
            self.__class__.__name__,
            self.name,
            self.instance_name,
            self.length,
            self.cache_params(),
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, "
            f"inst={self.instance_name!r}, L={self._length})"
        )
