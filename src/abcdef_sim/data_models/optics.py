from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Hashable
from abcdef_sim.utils.optics_builder import get_abcdef_matrices

import numpy as np
import numpy.typing as npt

ArrayLike = npt.ArrayLike
NDArrayF = npt.NDArray[np.float64]

RefractiveIndexFn = Callable[[NDArrayF], NDArrayF]


@dataclass(slots=True)
class Optic(ABC):
    """
    Base class for a single optical element in the ABCDEF formalism.

    - matrix(ω): 3x3 ray-transfer matrix in homogeneous coords.
    - n(ω): refractive index (default vacuum n=1).
    - length: scalar geometric length (default 0).
    - name: label for debugging.
    - instance_name: distinguishes multiple instances of same optic

    Caching Support:
        - cache_key(): stable hashable identity for cache buckets
        - cache_params(): override in subclasses to include parameters that affect matrix/n

    Subclasses typically use `matrix_from_abcd` with:
        a, b, c, d: scalars
        e_fn, f_fn: optional callables E(ω), F(ω) (default 0)

    """

    name: str = "optic"
    instance_name: str = "inst0"   # IMPORTANT: set this uniquely per instance in your system
    _length: float = 0.0
    _n_fn: RefractiveIndexFn | None = field(default=None, repr=False)

    # ---------- required API ----------

    @abstractmethod
    def matrix(self, omega: ArrayLike) -> NDArrayF:
        """
        Return the 3x3 transfer matrix for angular frequency omega.

        - omega can be scalar or array-like
        - Return shape: (..., 3, 3)
        """
        raise NotImplementedError

    def n(self, omega: ArrayLike) -> NDArrayF:
        """
        Refractive index vs angular frequency.

        Default: n = 1 (vacuum). You can:
        - override in subclass, or
        - inject a function via _n_fn in the constructor.
        """
        omega_arr = np.asarray(omega, dtype=np.float64)

        if self._n_fn is None:
            return np.ones_like(omega_arr, dtype=np.float64)

        return self._n_fn(omega_arr)

    @property
    def length(self) -> float:
        """
        Geometric length [internal units: e.g. µm].

        Default: 0 for thin/meta elements.
        """
        return self._length

    
    
    # --- caching hooks ---

    def cache_params(self) -> tuple:
        """
        Override this in subclasses to include parameters that affect matrix/n.
        Example for grating: groove_density, incidence_angle, etc.
        """
        return ()
    
    def cache_key(self) -> Hashable:
        """
        Cache bucket identity. MUST change if the optic's behavior changes.
        """
        return (
            self.__class__.__name__,
            self.name,
            self.instance_name,
            self.length,
            self.cache_params(),
        )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, inst={self.instance_name!r}, L={self._length})"
    




@dataclass(slots=True)
class FreeSpace(Optic):
     
     def matrix(self, omega: ArrayLike) -> NDArrayF:
         omega_arr = np.asarray(omega, dtype=float)
         return get_abcdef_matrices(a=1.0, 
                                    b=self.length,
                                    c=0.0,
                                    d=1.0,
                                    omega=omega_arr)



