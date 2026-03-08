from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.states import PHASE_CONTRIBUTIONS_META_KEY, RayState
from abcdef_sim.pipeline.stages import AbcdefOpticStage

pytestmark = pytest.mark.integration


def test_stage_process_accumulates_phase_contributions_without_aliasing() -> None:
    omega = np.array([2.5, 2.7], dtype=float)
    state0 = RayState(
        rays=np.array(
            [
                [[0.5], [0.1], [1.0]],
                [[-0.25], [0.2], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
        meta={},
    )

    stage1 = AbcdefOpticStage(cfg=_cfg(instance_name="fs1", omega=omega, length=1.5, f_shift=0.2))
    stage2 = AbcdefOpticStage(cfg=_cfg(instance_name="fs2", omega=omega, length=2.0, f_shift=-0.3))

    state1 = stage1.process(state0).state
    contribs_after_stage1 = state1.meta[PHASE_CONTRIBUTIONS_META_KEY]

    assert isinstance(contribs_after_stage1, tuple)
    assert len(contribs_after_stage1) == 1
    assert contribs_after_stage1[0].instance_name == "fs1"
    assert not np.allclose(state1.rays, state0.rays)

    state2 = stage2.process(state1).state
    contribs_after_stage2 = state2.meta[PHASE_CONTRIBUTIONS_META_KEY]

    assert isinstance(contribs_after_stage2, tuple)
    assert len(contribs_after_stage2) == 2
    assert [item.instance_name for item in contribs_after_stage2] == ["fs1", "fs2"]
    assert [item.optic_name for item in contribs_after_stage2] == ["FreeSpace", "FreeSpace"]
    assert len(contribs_after_stage1) == 1
    assert state1.meta[PHASE_CONTRIBUTIONS_META_KEY] is contribs_after_stage1
    assert not np.allclose(state2.rays, state1.rays)


def _cfg(*, instance_name: str, omega: np.ndarray, length: float, f_shift: float) -> OpticStageCfg:
    matrices = np.array(
        [
            [[1.0, length, 0.0], [0.0, 1.0, f_shift], [0.0, 0.0, 1.0]],
            [[1.0, length / 2.0, 0.1], [0.0, 1.0, f_shift / 2.0], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    return OpticStageCfg(
        name="optic stage cfg",
        tags={"kind": "test"},
        optic_name="FreeSpace",
        instance_name=instance_name,
        length=length,
        omega=omega,
        abcdef=matrices,
        refractive_index=np.ones_like(omega),
    )
