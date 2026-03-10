from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.states import RayState
from abcdef_sim.optics.frame_transform import FrameTransform
from abcdef_sim.physics.abcdef.adapters import apply_cfg

pytestmark = pytest.mark.physics


def test_frame_transform_matrix_applies_affine_coordinate_change() -> None:
    omega = np.array([1.7, 1.8], dtype=np.float64)
    transform = FrameTransform(
        instance_name="frame",
        x_offset_um=12.0,
        x_prime_offset=0.3,
        x_prime_scale=-1.0,
    )

    matrix = transform.matrix(omega)
    rays_in = np.array(
        [
            [[20.0], [0.5], [1.0]],
            [[-8.0], [-0.2], [1.0]],
        ],
        dtype=np.float64,
    )
    rays_out = matrix @ rays_in

    np.testing.assert_allclose(rays_out[:, 0, 0], np.array([8.0, -20.0], dtype=np.float64))
    np.testing.assert_allclose(rays_out[:, 1, 0], np.array([-0.8, -0.1], dtype=np.float64))


def test_apply_cfg_skips_phase_bookkeeping_for_frame_transform_stage() -> None:
    omega = np.array([1.7, 1.8], dtype=np.float64)
    transform = FrameTransform(
        instance_name="frame",
        x_offset_um=5.0,
        x_prime_offset=0.25,
        x_prime_scale=-1.0,
    )
    from abcdef_sim.cache.backend import NullCacheBackend
    from abcdef_sim.cfg_generator import OpticStageCfgGenerator

    cfg = OpticStageCfgGenerator(cache=NullCacheBackend()).build(transform, omega)
    assert cfg.phase_model == "none"

    state = RayState(
        rays=np.array(
            [
                [[7.0], [0.4], [1.0]],
                [[-3.0], [-0.1], [1.0]],
            ],
            dtype=np.float64,
        ),
        system=np.repeat(np.eye(3, dtype=np.float64)[None, ...], omega.size, axis=0),
        meta={},
    )

    state_out, contribution = apply_cfg(state=state, cfg=cfg)

    np.testing.assert_allclose(state_out.rays[:, 0, 0], np.array([2.0, -8.0], dtype=np.float64))
    np.testing.assert_allclose(state_out.rays[:, 1, 0], np.array([-0.65, -0.15], dtype=np.float64))
    np.testing.assert_allclose(contribution.phi0_rad, np.zeros_like(omega))
    np.testing.assert_allclose(contribution.phi3_rad, np.zeros_like(omega))
