"""Runtime adapter boundary between pipeline stages and pure ABCDEF kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.states import RayState
from abcdef_sim.physics.abcdef.phase_terms import martinez_k_center, phi0_rad_i, phi3_rad_i
from abcdef_sim.physics.abcdef.propagation import propagate_step

if TYPE_CHECKING:
    from abcdef_sim._phys_pipeline import PolicyBag

__all__ = ["apply_cfg"]


def apply_cfg(
    state: RayState,
    cfg: OpticStageCfg,
    policy: PolicyBag | None = None,
) -> tuple[RayState, PhaseContribution]:
    """Propagate one optic config and return its Martinez phase bookkeeping."""

    del policy  # Reserved for future numeric/policy variants at this adapter seam.

    state_out = propagate_step(state, cfg.abcdef)
    k_center = martinez_k_center(cfg.omega)
    phi0_rad = phi0_rad_i(k_center, cfg.length, cfg.refractive_index)
    phi3_rad = phi3_rad_i(k_center, cfg.abcdef[:, 1, 2], state_out.rays[:, 0, 0])

    contribution = PhaseContribution(
        optic_name=cfg.optic_name,
        instance_name=cfg.instance_name,
        omega=cfg.omega,
        delta_omega_rad_per_fs=cfg.delta_omega_rad_per_fs,
        omega0_rad_per_fs=cfg.omega0_rad_per_fs,
        phi0_rad=phi0_rad,
        phi3_rad=phi3_rad,
    )
    return state_out, contribution
