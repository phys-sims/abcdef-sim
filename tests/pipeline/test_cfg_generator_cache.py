from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from abcdef_sim.cache.cfg_two_lvl import GridKeyer, OmegaKeyer, TwoLevelMemoryCache
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.optics.base import ArrayLike, NDArrayF, Optic


@dataclass(slots=True)
class Omega0SensitiveOptic(Optic):
    calls: int = 0

    def matrix(self, omega: ArrayLike, *, omega0: float | None = None) -> NDArrayF:
        omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
        omega_ref = float(np.mean(omega_arr) if omega0 is None else omega0)
        matrices = np.zeros((omega_arr.size, 3, 3), dtype=np.float64)
        matrices[:, 0, 0] = 1.0
        matrices[:, 1, 1] = 1.0
        matrices[:, 2, 2] = 1.0
        matrices[:, 1, 2] = omega_arr - omega_ref
        self.calls += 1
        return matrices

    def l2_cache_safe(self) -> bool:
        return False


def test_cfg_generator_disables_l1_for_omega0_sensitive_optics() -> None:
    cache = TwoLevelMemoryCache(
        omega_keyer=OmegaKeyer(omega_step=1e-6),
        grid_keyer=GridKeyer(grid_step=1e-6),
    )
    cfg_gen = OpticStageCfgGenerator(cache=cache, expensive_types=(Omega0SensitiveOptic,))
    optic = Omega0SensitiveOptic()
    omega = np.array([1.0, 2.0, 3.0], dtype=float)

    first = cfg_gen.build(optic=optic, omega=omega, omega0_rad_per_fs=2.0)
    second = cfg_gen.build(optic=optic, omega=omega, omega0_rad_per_fs=2.5)

    np.testing.assert_allclose(first.abcdef[:, 1, 2], np.array([-1.0, 0.0, 1.0]))
    np.testing.assert_allclose(second.abcdef[:, 1, 2], np.array([-1.5, -0.5, 0.5]))
    assert optic.calls == 2
