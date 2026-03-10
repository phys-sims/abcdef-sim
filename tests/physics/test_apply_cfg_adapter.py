from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.states import RayState
from abcdef_sim.physics.abcdef.adapters import apply_cfg
from abcdef_sim.physics.abcdef.phase_terms import martinez_k, martinez_k_center

pytestmark = pytest.mark.physics


def test_apply_cfg_propagates_state_and_returns_martinez_phase_contribution() -> None:
    omega = np.array([3.0, 3.6], dtype=float)
    rays_in = np.array(
        [
            [[1.2], [0.15], [1.0]],
            [[-0.7], [0.25], [1.0]],
        ],
        dtype=float,
    )
    system_in = np.array(
        [
            [[1.0, 0.3, 0.1], [0.2, 0.9, -0.2], [0.0, 0.0, 1.0]],
            [[0.8, -0.4, 0.0], [0.1, 1.1, 0.3], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    matrices = np.array(
        [
            [[1.0, 2.0, 0.4], [0.1, 1.1, 0.5], [0.0, 0.0, 1.0]],
            [[0.9, -1.5, -0.6], [0.0, 1.2, -0.4], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    refractive_index = np.array([1.2, 1.4], dtype=float)

    cfg = OpticStageCfg(
        name="optic stage cfg",
        tags={"kind": "test"},
        optic_name="FreeSpace",
        instance_name="fs1",
        length=4.0,
        omega=omega,
        abcdef=matrices,
        refractive_index=refractive_index,
    )
    state = RayState(rays=rays_in, system=system_in, meta={"seed": "state"})

    state_out, contribution = apply_cfg(state=state, cfg=cfg)

    expected_rays = matrices @ rays_in
    expected_system = matrices @ system_in
    k_center = martinez_k_center(omega)
    expected_phi0 = martinez_k(omega) * cfg.length * refractive_index
    expected_phi3 = 0.5 * k_center * matrices[:, 1, 2] * expected_rays[:, 0, 0]

    np.testing.assert_allclose(state_out.rays, expected_rays)
    np.testing.assert_allclose(state_out.system, expected_system)
    assert state_out.meta == state.meta
    assert state_out.meta is not state.meta

    assert contribution.optic_name == cfg.optic_name
    assert contribution.instance_name == cfg.instance_name
    np.testing.assert_allclose(contribution.omega, omega)
    np.testing.assert_allclose(contribution.phi0_rad, expected_phi0)
    np.testing.assert_allclose(contribution.phi3_rad, expected_phi3)
