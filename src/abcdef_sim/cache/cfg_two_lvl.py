from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Hashable, Optional, Any

import numpy as np
import numpy.typing as npt

from abcdef_sim.data_models.optics import Optic
from abcdef_sim.utils.grids import LinspaceGrid

NDArrayF = npt.NDArray[np.float64]


def _qint(x: float, step: float) -> int:
    return int(np.round(np.float64(x) / np.float64(step)))


@dataclass(frozen=True)
class OmegaKeyer:
    """
    Quantize omega to integer bins for stable keys across "slightly different grids".
    """
    omega_step: float

    def key(self, w: float) -> int:
        return _qint(float(w), self.omega_step)


@dataclass(frozen=True)
class GridKeyer:
    """
    L1 key: quantized (w0, span, N).
    """
    grid_step: float

    def key(self, grid: LinspaceGrid) -> tuple[int, int, int]:
        return (_qint(grid.w0, self.grid_step), _qint(grid.span, self.grid_step), int(grid.N))


@dataclass(frozen=True)
class CachedEntry:
    abcdef: NDArrayF  # (3,3)
    n: float


@dataclass(frozen=True)
class GridEntry:
    mats: NDArrayF  # (N,3,3)
    ns: NDArrayF    # (N,)


@dataclass
class TwoLevelMemoryCacheBackend:
    """
    L1: (optic_key, grid_key) -> full arrays
    L2: optic_key -> omega_key -> per-omega entries
    """
    omega_keyer: OmegaKeyer
    grid_keyer: GridKeyer

    l1: dict[tuple[Hashable, tuple[int, int, int]], GridEntry] = field(default_factory=dict)
    l2: dict[Hashable, dict[int, CachedEntry]] = field(default_factory=dict)

    # Stats
    l1_hits: int = 0
    l1_misses: int = 0
    l2_hits: int = 0
    l2_misses: int = 0

    def get_or_compute(
        self,
        optic: Optic,
        omega: NDArrayF,
        grid: Optional[LinspaceGrid],
        *,
        matrix_fn: Callable[[Optic, NDArrayF], NDArrayF],
        n_fn: Callable[[Optic, NDArrayF], NDArrayF],
    ) -> tuple[NDArrayF, NDArrayF]:

        w = np.asarray(omega, dtype=np.float64).reshape(-1)
        N = w.size
        ok = optic.cache_key()

        # -------- L1 fast path --------
        if grid is not None:
            gk = self.grid_keyer.key(grid)
            l1k = (ok, gk)
            hit = self.l1.get(l1k)
            if hit is not None:
                self.l1_hits += 1
                return hit.mats, hit.ns
            self.l1_misses += 1

        mats = np.empty((N, 3, 3), dtype=np.float64)
        ns = np.empty((N,), dtype=np.float64)

        bucket = self.l2.setdefault(ok, {})

        miss_idx: list[int] = []
        miss_w: list[float] = []

        # -------- L2 per-omega hits/misses --------
        for i in range(N):
            wk = self.omega_keyer.key(float(w[i]))
            entry = bucket.get(wk)
            if entry is None:
                self.l2_misses += 1
                miss_idx.append(i)
                miss_w.append(float(w[i]))
            else:
                self.l2_hits += 1
                mats[i] = entry.abcdef
                ns[i] = entry.n

        # -------- compute misses vectorized --------
        if miss_idx:
            w_miss = np.asarray(miss_w, dtype=np.float64)
            mats_miss = np.asarray(matrix_fn(optic, w_miss), dtype=np.float64)
            ns_miss = np.asarray(n_fn(optic, w_miss), dtype=np.float64).reshape(-1)

            if mats_miss.shape != (w_miss.size, 3, 3):
                raise ValueError(f"matrix_fn returned {mats_miss.shape}, expected {(w_miss.size,3,3)}")
            if ns_miss.shape != (w_miss.size,):
                raise ValueError(f"n_fn returned {ns_miss.shape}, expected {(w_miss.size,)}")

            # scatter back (alignment-safe) + populate L2
            for j, i in enumerate(miss_idx):
                mats[i] = mats_miss[j]
                ns[i] = ns_miss[j]
                wk = self.omega_keyer.key(float(w_miss[j]))
                bucket[wk] = CachedEntry(abcdef=mats_miss[j].copy(), n=float(ns_miss[j]))

        # -------- populate L1 for this grid --------
        if grid is not None:
            gk = self.grid_keyer.key(grid)
            self.l1[(ok, gk)] = GridEntry(mats=mats, ns=ns)

        return mats, ns
