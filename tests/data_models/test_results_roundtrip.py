from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.results import PhaseContribution, PipelineResult
from abcdef_sim.data_models.states import RayState


def _dummy_ray_state(batch_size: int) -> RayState:
    rays = np.zeros((batch_size, 3, 1), dtype=float)
    rays[:, 2, 0] = 1.0
    system = np.repeat(np.eye(3, dtype=float)[None, ...], batch_size, axis=0)
    return RayState(rays=rays, system=system)


def test_pipeline_result_roundtrip_keeps_structure_and_arrays() -> None:
    omega = np.linspace(1.0, 2.0, 3, dtype=float)
    delta_omega = omega - np.mean(omega)
    contribution = PhaseContribution(
        optic_name="grating",
        instance_name="g1",
        backend_id="rcwa:order=1",
        omega=omega,
        delta_omega_rad_per_fs=delta_omega,
        omega0_rad_per_fs=1.5,
        phi0_rad=np.array([0.1, 0.2, 0.3], dtype=float),
        phi_geom_rad=np.array([0.15, 0.25, 0.35], dtype=float),
        phi3_transport_like_rad=np.array([0.41, 0.51, 0.61], dtype=float),
        phi3_phase_rad=np.array([0.42, 0.52, 0.62], dtype=float),
        phi3_rad=np.array([0.4, 0.5, 0.6], dtype=float),
        path_length_um=np.array([1.0, 1.1, 1.2], dtype=float),
        group_delay_fs=np.array([3.0, 3.1, 3.2], dtype=float),
        filter_amp=np.array([0.9, 0.8, 0.7], dtype=float),
        filter_phase_rad=np.array([0.0, 0.1, 0.2], dtype=float),
    )
    result = PipelineResult(
        final_state=_dummy_ray_state(batch_size=3),
        omega=omega,
        delta_omega_rad_per_fs=delta_omega,
        omega0_rad_per_fs=1.5,
        contributions=(contribution,),
        phi0_axial_total_rad=np.array([0.1, 0.2, 0.3], dtype=float),
        phi_geom_total_rad=np.array([0.15, 0.25, 0.35], dtype=float),
        phi3_transport_like_total_rad=np.array([0.41, 0.51, 0.61], dtype=float),
        phi3_phase_total_rad=np.array([0.42, 0.52, 0.62], dtype=float),
        phi3_total_rad=np.array([0.4, 0.5, 0.6], dtype=float),
        phi1_rad=np.array([1.0, 1.1, 1.2], dtype=float),
        phi2_rad=np.array([2.0, 2.1, 2.2], dtype=float),
        phi4_rad=np.array([3.0, 3.1, 3.2], dtype=float),
        phi_total_rad=np.array([6.0, 6.3, 6.6], dtype=float),
    )

    dumped = result.model_dump(mode="python")
    assert list(dumped) == [
        "final_state",
        "omega",
        "delta_omega_rad_per_fs",
        "omega0_rad_per_fs",
        "contributions",
        "phi0_axial_total_rad",
        "phi_geom_total_rad",
        "phi3_transport_like_total_rad",
        "phi3_phase_total_rad",
        "phi3_total_rad",
        "phi1_rad",
        "phi2_rad",
        "phi4_rad",
        "phi_total_rad",
    ]
    assert isinstance(dumped["final_state"], dict)
    assert list(dumped["final_state"]) == ["rays", "system", "meta"]
    assert isinstance(dumped["omega"], np.ndarray)
    assert isinstance(dumped["contributions"], tuple)
    assert dumped["contributions"][0]["instance_name"] == "g1"

    restored = PipelineResult.model_validate(dumped)

    np.testing.assert_allclose(restored.omega, omega)
    np.testing.assert_allclose(restored.delta_omega_rad_per_fs, delta_omega)
    np.testing.assert_allclose(restored.contributions[0].phi0_rad, contribution.phi0_rad)
    np.testing.assert_allclose(restored.contributions[0].phi_geom_rad, contribution.phi_geom_rad)
    np.testing.assert_allclose(
        restored.contributions[0].phi3_transport_like_rad,
        contribution.phi3_transport_like_rad,
    )
    np.testing.assert_allclose(
        restored.contributions[0].phi3_phase_rad,
        contribution.phi3_phase_rad,
    )
    np.testing.assert_allclose(restored.phi_total_rad, result.phi_total_rad)
    np.testing.assert_allclose(restored.final_state.rays, result.final_state.rays)


def test_phase_contribution_rejects_mismatched_vector_lengths() -> None:
    omega = np.linspace(1.0, 2.0, 3, dtype=float)

    with pytest.raises(ValueError, match="phi3_rad must have shape"):
        PhaseContribution(
            optic_name="grating",
            instance_name="g1",
            omega=omega,
            phi0_rad=np.array([0.1, 0.2, 0.3], dtype=float),
            phi3_rad=np.array([0.4, 0.5], dtype=float),
        )


def test_pipeline_result_rejects_contribution_omega_mismatch() -> None:
    omega = np.linspace(1.0, 2.0, 3, dtype=float)
    contribution = PhaseContribution(
        optic_name="grating",
        instance_name="g1",
        omega=np.linspace(1.0, 2.5, 3, dtype=float),
        phi0_rad=np.array([0.1, 0.2, 0.3], dtype=float),
        phi3_rad=np.array([0.4, 0.5, 0.6], dtype=float),
    )

    with pytest.raises(ValueError, match="exactly match pipeline omega values"):
        PipelineResult(
            final_state=_dummy_ray_state(batch_size=3),
            omega=omega,
            contributions=(contribution,),
            phi_total_rad=np.array([0.5, 0.7, 0.9], dtype=float),
        )


def test_pipeline_result_rejects_final_state_batch_mismatch() -> None:
    omega = np.linspace(1.0, 2.0, 3, dtype=float)
    contribution = PhaseContribution(
        optic_name="grating",
        instance_name="g1",
        omega=omega,
        phi0_rad=np.array([0.1, 0.2, 0.3], dtype=float),
        phi3_rad=np.array([0.4, 0.5, 0.6], dtype=float),
    )

    with pytest.raises(ValueError, match="final_state batch dimension must match omega"):
        PipelineResult(
            final_state=_dummy_ray_state(batch_size=2),
            omega=omega,
            contributions=(contribution,),
            phi_total_rad=np.array([0.5, 0.7, 0.9], dtype=float),
        )
