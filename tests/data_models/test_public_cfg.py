from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim import AbcdefCfg, FreeSpaceCfg
from abcdef_sim.data_models.specs import LaserSpec


def test_abcdef_cfg_rejects_duplicate_instance_names() -> None:
    with pytest.raises(ValueError, match="unique instance_name"):
        AbcdefCfg(
            optics=(
                FreeSpaceCfg(instance_name="fs", length=10.0),
                FreeSpaceCfg(instance_name="fs", length=20.0),
            )
        )


def test_abcdef_cfg_requires_at_least_one_optic() -> None:
    with pytest.raises(ValueError, match="at least one optic"):
        AbcdefCfg(optics=())


def test_laser_spec_reconstructs_absolute_omega_from_explicit_delta_grid() -> None:
    spec = LaserSpec(
        w0=2.5,
        span=0.3,
        N=4,
        delta_omega_rad_per_fs=(-0.3, -0.1, 0.1, 0.3),
    )

    np.testing.assert_allclose(spec.delta_omega(), np.array([-0.3, -0.1, 0.1, 0.3], dtype=float))
    np.testing.assert_allclose(spec.omega(), np.array([2.2, 2.4, 2.6, 2.8], dtype=float))
