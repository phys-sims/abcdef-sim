from __future__ import annotations

import numpy as np

from abcdef_sim.data_models.optics import FreeSpace as LegacyFreeSpace
from abcdef_sim.optics.freespace import FreeSpace


def test_freespace_matrix_shape_and_cache_key() -> None:
    omega_grid = np.linspace(1.0, 5.0, 8, dtype=np.float64)
    optic = FreeSpace(length=2.5, instance_name="fs-1")

    mat = optic.matrix(omega_grid)

    assert mat.shape == (omega_grid.size, 3, 3)

    key_one = optic.cache_key()
    key_two = optic.cache_key()
    assert key_one == key_two
    hash(key_one)


def test_legacy_import_path_still_works() -> None:
    legacy = LegacyFreeSpace(length=1.0, instance_name="legacy")

    assert legacy.matrix(np.array([1.0], dtype=np.float64)).shape == (1, 3, 3)
