from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.states import RayState
from abcdef_sim.physics.abcdef import compute_pipeline_result
from abcdef_sim.physics.abcdef.phase_terms import (
    combine_phi_total_rad,
    martinez_k_center,
    phi1_rad,
    phi2_rad,
)

pytestmark = pytest.mark.physics


def test_compute_pipeline_result_combines_contributions_and_optional_terms() -> None:
    omega = np.array([3.0, 3.2, 3.4], dtype=float)
    contributions = (
        PhaseContribution(
            optic_name="grating",
            instance_name="g1",
            omega=omega,
            phi0_rad=np.array([0.3, 0.4, 0.5], dtype=float),
            phi3_rad=np.array([0.05, -0.02, 0.01], dtype=float),
        ),
        PhaseContribution(
            optic_name="lens",
            instance_name="l1",
            omega=omega,
            phi0_rad=np.array([0.2, 0.1, 0.3], dtype=float),
            phi3_rad=np.array([0.01, 0.03, -0.04], dtype=float),
        ),
    )
    initial_state = RayState(
        rays=np.array(
            [
                [[1.0], [0.2], [1.0]],
                [[-0.5], [0.4], [1.0]],
                [[0.25], [-0.6], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 3, axis=0),
    )
    final_system = np.array(
        [
            [[1.0, 0.4, 0.0], [0.2, 1.2, 0.0], [0.0, 0.0, 1.0]],
            [[0.9, -0.1, 0.0], [0.3, 1.1, 0.0], [0.0, 0.0, 1.0]],
            [[1.1, 0.3, 0.0], [-0.2, 0.8, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    final_state = RayState(
        rays=np.array(
            [
                [[0.7], [0.1], [1.0]],
                [[-0.3], [0.2], [1.0]],
                [[0.5], [-0.1], [1.0]],
            ],
            dtype=float,
        ),
        system=final_system,
    )
    q_in = np.array([2.0 + 3.0j, 1.5 + 2.5j, 3.0 + 1.0j], dtype=np.complex128)
    lhs = 1.0 / np.sqrt(final_system[:, 0, 0].astype(np.complex128) + final_system[:, 0, 1] / q_in)
    w_in = np.array([1.0, 1.3, 0.8], dtype=float)
    w_out = w_in / np.abs(lhs) ** 2
    phi4 = np.array([0.02, 0.01, 0.03], dtype=float)

    result = compute_pipeline_result(
        initial_state,
        final_state,
        contributions,
        q_in=q_in,
        w_in=w_in,
        w_out=w_out,
        phi4_rad=phi4,
    )

    expected_phi0 = contributions[0].phi0_rad + contributions[1].phi0_rad
    expected_phi3 = contributions[0].phi3_rad + contributions[1].phi3_rad
    expected_phi1 = phi1_rad(final_system, q_in, w_in, w_out)
    expected_phi2 = phi2_rad(martinez_k_center(omega), initial_state.rays, final_state.rays)
    expected_total = combine_phi_total_rad(
        expected_phi0,
        expected_phi1,
        expected_phi2,
        expected_phi3,
        phi4,
    )

    assert tuple(item.instance_name for item in result.contributions) == ("g1", "l1")
    np.testing.assert_allclose(result.omega, omega)
    np.testing.assert_allclose(result.phi1_rad, expected_phi1)
    np.testing.assert_allclose(result.phi2_rad, expected_phi2)
    np.testing.assert_allclose(result.phi4_rad, phi4)
    np.testing.assert_allclose(result.phi_total_rad, expected_total)


def test_compute_pipeline_result_skips_phi1_when_beam_inputs_are_absent() -> None:
    omega = np.array([2.0, 2.2], dtype=float)
    contribution = PhaseContribution(
        optic_name="space",
        instance_name="fs1",
        omega=omega,
        phi0_rad=np.array([0.1, 0.2], dtype=float),
        phi3_rad=np.array([0.0, 0.05], dtype=float),
    )
    initial_state = RayState(
        rays=np.array(
            [
                [[0.2], [0.3], [1.0]],
                [[-0.4], [0.1], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
    )
    final_state = RayState(
        rays=np.array(
            [
                [[0.1], [0.25], [1.0]],
                [[-0.2], [0.15], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
    )

    result = compute_pipeline_result(initial_state, final_state, (contribution,))

    expected_phi2 = phi2_rad(martinez_k_center(omega), initial_state.rays, final_state.rays)
    expected_total = combine_phi_total_rad(
        contribution.phi0_rad,
        None,
        expected_phi2,
        contribution.phi3_rad,
    )

    assert result.phi1_rad is None
    assert result.phi4_rad is None
    np.testing.assert_allclose(result.phi2_rad, expected_phi2)
    np.testing.assert_allclose(result.phi_total_rad, expected_total)


def test_compute_pipeline_result_requires_non_empty_contributions() -> None:
    state = RayState(
        rays=np.array([[[0.0], [0.0], [1.0]]], dtype=float),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 1, axis=0),
    )

    with pytest.raises(ValueError, match="at least one phase contribution"):
        compute_pipeline_result(state, state, ())


def test_compute_pipeline_result_rejects_mismatched_omega() -> None:
    contribution_a = PhaseContribution(
        optic_name="space",
        instance_name="fs1",
        omega=np.array([2.0, 2.2], dtype=float),
        phi0_rad=np.array([0.1, 0.2], dtype=float),
        phi3_rad=np.array([0.0, 0.05], dtype=float),
    )
    contribution_b = PhaseContribution(
        optic_name="lens",
        instance_name="l1",
        omega=np.array([2.0, 2.3], dtype=float),
        phi0_rad=np.array([0.2, 0.1], dtype=float),
        phi3_rad=np.array([0.01, -0.02], dtype=float),
    )
    state = RayState(
        rays=np.array(
            [
                [[0.2], [0.3], [1.0]],
                [[-0.4], [0.1], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
    )

    with pytest.raises(ValueError, match="identical omega values"):
        compute_pipeline_result(state, state, (contribution_a, contribution_b))


def test_compute_pipeline_result_requires_complete_phi1_inputs() -> None:
    omega = np.array([2.0, 2.2], dtype=float)
    contribution = PhaseContribution(
        optic_name="space",
        instance_name="fs1",
        omega=omega,
        phi0_rad=np.array([0.1, 0.2], dtype=float),
        phi3_rad=np.array([0.0, 0.05], dtype=float),
    )
    state = RayState(
        rays=np.array(
            [
                [[0.2], [0.3], [1.0]],
                [[-0.4], [0.1], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
    )

    with pytest.raises(ValueError, match="must be provided together"):
        compute_pipeline_result(
            state,
            state,
            (contribution,),
            q_in=np.array([1.0 + 1.0j, 1.5 + 0.5j]),
        )
