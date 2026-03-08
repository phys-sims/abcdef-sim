from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from abcdef_sim.data_models.results import PhaseContribution, PipelineResult
from abcdef_sim.data_models.states import RayState
from abcdef_sim.physics.abcdef.phase_terms import (
    combine_phi_total_rad,
    martinez_k_center,
    phi1_rad,
    phi2_rad,
)

__all__ = ["compute_pipeline_result"]
NDArrayF = npt.NDArray[np.float64]


def compute_pipeline_result(
    initial_state: RayState,
    final_state: RayState,
    contributions: Sequence[PhaseContribution],
    *,
    q_in: object | None = None,
    w_in: object | None = None,
    w_out: object | None = None,
    phi4_rad: object | None = None,
) -> PipelineResult:
    """Build a PipelineResult from pure ray-state and phase-contribution data."""

    contribution_tuple = tuple(contributions)
    omega = _validate_contributions_and_get_omega(contribution_tuple)
    k_center = martinez_k_center(omega)

    phi0_total = np.zeros_like(omega)
    phi3_total = np.zeros_like(omega)
    for contribution in contribution_tuple:
        phi0_total = phi0_total + contribution.phi0_rad
        phi3_total = phi3_total + contribution.phi3_rad

    phi1 = _compute_phi1(final_state, q_in=q_in, w_in=w_in, w_out=w_out)
    phi2 = phi2_rad(k_center, initial_state.rays, final_state.rays)
    phi4 = None if phi4_rad is None else np.asarray(phi4_rad, dtype=np.float64).reshape(-1)
    phi_total = combine_phi_total_rad(phi0_total, phi1, phi2, phi3_total, phi4)

    return PipelineResult(
        final_state=final_state,
        omega=omega,
        contributions=contribution_tuple,
        phi1_rad=phi1,
        phi2_rad=phi2,
        phi4_rad=phi4,
        phi_total_rad=phi_total,
    )


def _validate_contributions_and_get_omega(
    contributions: tuple[PhaseContribution, ...],
) -> NDArrayF:
    if not contributions:
        raise ValueError("compute_pipeline_result requires at least one phase contribution")

    omega = np.asarray(contributions[0].omega, dtype=np.float64).reshape(-1)
    for contribution in contributions[1:]:
        if contribution.omega.shape != omega.shape:
            raise ValueError(
                "all phase contributions must share one omega shape: "
                f"expected {omega.shape}, got {contribution.omega.shape}"
            )
        if not np.array_equal(contribution.omega, omega):
            raise ValueError("all phase contributions must share identical omega values")
    return omega


def _compute_phi1(
    final_state: RayState,
    *,
    q_in: object | None,
    w_in: object | None,
    w_out: object | None,
) -> NDArrayF | None:
    provided = (q_in is not None, w_in is not None, w_out is not None)
    if not any(provided):
        return None
    if not all(provided):
        raise ValueError("q_in, w_in, and w_out must be provided together to compute phi1_rad")
    return phi1_rad(final_state.system, q_in, w_in, w_out)
