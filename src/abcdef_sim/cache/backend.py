from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
import numpy.typing as npt

from abcdef_sim.optics.base import Optic
from abcdef_sim.utils.grids import LinspaceGrid

NDArrayF = npt.NDArray[np.float64]


class CacheBackend(Protocol):
    """
    Cache backend interface.

    Implementations can be:
      - Null (no caching)
      - In-memory dict
      - Disk (sqlite/lmdb)
      - Redis (with batching)
    """

    def get_or_compute(
        self,
        optic: Optic,
        omega: NDArrayF,
        grid: LinspaceGrid | None,
        *,
        matrix_fn: Callable[[Optic, NDArrayF], NDArrayF],
        n_fn: Callable[[Optic, NDArrayF], NDArrayF],
        use_l1: bool = True,
        use_l2: bool = True,
    ) -> tuple[NDArrayF, NDArrayF]: ...


class NullCacheBackend:
    """
    No caching. Always computes.
    """

    def get_or_compute(
        self,
        optic: Optic,
        omega: NDArrayF,
        grid: LinspaceGrid | None,
        *,
        matrix_fn: Callable[[Optic, NDArrayF], NDArrayF],
        n_fn: Callable[[Optic, NDArrayF], NDArrayF],
        use_l1: bool = True,
        use_l2: bool = True,
    ) -> tuple[NDArrayF, NDArrayF]:
        w = np.asarray(omega, dtype=np.float64).reshape(-1)
        mats = np.asarray(matrix_fn(optic, w), dtype=np.float64)
        ns = np.asarray(n_fn(optic, w), dtype=np.float64).reshape(-1)
        return mats, ns
