from __future__ import annotations

import numpy as np

from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.states import PHASE_CONTRIBUTIONS_META_KEY, RayState


def test_raystate_hashable_repr_includes_phase_contributions() -> None:
    contribution_a = PhaseContribution(
        optic_name="space",
        instance_name="fs1",
        omega=np.array([2.0, 2.2], dtype=float),
        phi0_rad=np.array([0.1, 0.2], dtype=float),
        phi3_rad=np.array([0.0, 0.05], dtype=float),
    )
    contribution_b = PhaseContribution(
        optic_name="space",
        instance_name="fs2",
        omega=np.array([2.0, 2.2], dtype=float),
        phi0_rad=np.array([0.1, 0.25], dtype=float),
        phi3_rad=np.array([0.0, 0.05], dtype=float),
    )

    state_a = _state_with_contributions((contribution_a,))
    state_a_repeat = _state_with_contributions((contribution_a,))
    state_b = _state_with_contributions((contribution_b,))

    assert state_a.hashable_repr() == state_a_repeat.hashable_repr()
    assert state_a.hashable_repr() != state_b.hashable_repr()


def _state_with_contributions(
    contributions: tuple[PhaseContribution, ...],
) -> RayState:
    return RayState(
        rays=np.array(
            [
                [[0.2], [0.3], [1.0]],
                [[-0.4], [0.1], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
        meta={PHASE_CONTRIBUTIONS_META_KEY: contributions},
    )
