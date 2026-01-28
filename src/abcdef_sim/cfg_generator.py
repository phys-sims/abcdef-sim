from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Callable

import numpy as np
import numpy.typing as npt

from abcdef_sim.cache.backend import CacheBackend
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.optics import Optic
from abcdef_sim.utils.grids import LinspaceGrid, infer_linspace_grid

NDArrayF = npt.NDArray[np.float64]


@dataclass
class OpticStageCfgGenerator:
    cache: CacheBackend
    expensive_types: tuple[type, ...] = field(default_factory=tuple)
    expensive_predicate: Optional[Callable[[Optic], bool]] = None

    def is_expensive(self, optic: Optic) -> bool:
        if self.expensive_predicate is not None:
            return bool(self.expensive_predicate(optic))
        return isinstance(optic, self.expensive_types)

    def build(
        self,
        optic: Optic,
        omega: npt.ArrayLike,
        *,
        tags: Optional[dict[str, Any]] = None,
        grid: Optional[LinspaceGrid] = None,
        infer_grid: bool = True,
        freeze_arrays: bool = False,
        use_cache_for_nonexpensive: bool = False,
    ) -> OpticStageCfg:

        w = np.asarray(omega, dtype=np.float64).reshape(-1)

        if grid is None and infer_grid:
            grid = infer_linspace_grid(w)

        should_cache = self.is_expensive(optic) or use_cache_for_nonexpensive

        if should_cache:
            mats, ns = self.cache.get_or_compute(
                optic,
                w,
                grid,
                matrix_fn=lambda o, ww: o.matrix(ww),
                n_fn=lambda o, ww: o.n(ww),
            )
        else:
            mats = np.asarray(optic.matrix(w), dtype=np.float64)
            ns = np.asarray(optic.n(w), dtype=np.float64).reshape(-1)

        # shape checks (fail fast)
        if mats.shape != (w.size, 3, 3):
            raise ValueError(f"{optic} matrix returned {mats.shape}, expected {(w.size,3,3)}")
        if ns.shape != (w.size,):
            raise ValueError(f"{optic} n returned {ns.shape}, expected {(w.size,)}")

        if freeze_arrays:
            w.setflags(write=False)
            mats.setflags(write=False)
            ns.setflags(write=False)

        return OpticStageCfg(
            name="optic stage cfg",
            tags={} if tags is None else tags,
            optic_name=optic.name,
            instance_name=optic.instance_name,
            length=float(optic.length),
            omega=w,
            abcdef=mats,
            refractive_index=ns,
        )
