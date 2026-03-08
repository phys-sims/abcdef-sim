from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from abcdef_sim._phys_pipeline import PolicyBag
from abcdef_sim.cache.backend import CacheBackend
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.optics.base import Optic
from abcdef_sim.utils.grids import LinspaceGrid, infer_linspace_grid

NDArrayF = npt.NDArray[np.float64]


@dataclass
class OpticStageCfgGenerator:
    """
    Creates OpticStageCfg for a given optic and omega grid.

    CacheBackend is optional: you can pass NullCacheBackend.
    """

    cache: CacheBackend
    expensive_types: tuple[type, ...] = field(default_factory=tuple)
    expensive_predicate: Callable[[Optic], bool] | None = None

    def is_expensive(self, optic: Optic) -> bool:
        if self.expensive_predicate is not None:
            return bool(self.expensive_predicate(optic))
        return isinstance(optic, self.expensive_types)

    def build(
        self,
        optic: Optic,
        omega: npt.ArrayLike,
        *,
        delta_omega: npt.ArrayLike | None = None,
        omega0_rad_per_fs: float | None = None,
        tags: dict[str, Any] | None = None,
        policy: PolicyBag | None = None,
        infer_grid: bool = True,
        freeze_arrays: bool = False,
    ) -> OpticStageCfg:
        w = np.asarray(omega, dtype=np.float64).reshape(-1)
        omega0 = float(np.mean(w) if omega0_rad_per_fs is None else omega0_rad_per_fs)
        if delta_omega is None:
            delta_w = w - omega0
        else:
            delta_w = np.asarray(delta_omega, dtype=np.float64).reshape(-1)
            if delta_w.shape != w.shape:
                raise ValueError(
                    f"delta_omega shape must match omega shape {w.shape}; got {delta_w.shape}"
                )

        grid: LinspaceGrid | None = None
        if infer_grid:
            grid = infer_linspace_grid(w)

        # Policy knobs (run-wide)
        use_cache = True if policy is None else bool(policy.get("cfg.use_cache", True))
        use_l1 = True if policy is None else bool(policy.get("cfg.cache_l1", True))
        use_l2 = True if policy is None else bool(policy.get("cfg.cache_l2", True))
        if not optic.l2_cache_safe():
            use_l1 = False
            use_l2 = False

        should_cache = use_cache and self.is_expensive(optic)

        if should_cache:
            mats, ns = self.cache.get_or_compute(
                optic,
                w,
                grid,
                matrix_fn=lambda o, ww: o.matrix(ww, omega0=omega0),
                n_fn=lambda o, ww: o.n(ww, omega0=omega0),
                use_l1=use_l1,
                use_l2=use_l2,
            )
        else:
            mats = np.asarray(optic.matrix(w, omega0=omega0), dtype=np.float64)
            ns = np.asarray(optic.n(w, omega0=omega0), dtype=np.float64).reshape(-1)

        if mats.shape != (w.size, 3, 3):
            raise ValueError(f"{optic} matrix returned {mats.shape}, expected {(w.size, 3, 3)}")
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
            delta_omega_rad_per_fs=delta_w,
            omega0_rad_per_fs=omega0,
            abcdef=mats,
            refractive_index=ns,
        )
