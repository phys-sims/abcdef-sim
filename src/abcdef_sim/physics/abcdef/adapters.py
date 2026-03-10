"""Runtime adapter boundary between pipeline stages and pure ABCDEF kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.states import RayState
from abcdef_sim.physics.abcdef.action import (
    free_space_path_to_planar_surface_um,
    group_delay_from_path_length_fs,
    phase_from_group_delay_rad,
)
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

    if cfg.phase_model == "none":
        state_out = RayState(
            rays=np.asarray(cfg.abcdef, dtype=np.float64)
            @ np.asarray(state.rays, dtype=np.float64),
            system=np.asarray(state.system, dtype=np.float64),
            meta=dict(state.meta),
        )
        phi0_rad = np.zeros_like(np.asarray(cfg.omega, dtype=np.float64).reshape(-1))
        phi_geom_rad = None
        phi3_rad = np.zeros_like(phi0_rad)
        path_length_um = None
        group_delay_fs = None
    else:
        state_out = propagate_step(state, cfg.abcdef)
        k_center = martinez_k_center(cfg.omega)
        phi0_rad = phi0_rad_i(cfg.omega, cfg.length, cfg.refractive_index)
        phi3_rad = phi3_rad_i(k_center, cfg.abcdef[:, 1, 2], state_out.rays[:, 0, 0])
        phi_geom_rad = None
        path_length_um = None
        group_delay_fs = None
        if cfg.phi_geom_model == "free_space_to_planar_surface":
            path_length_um = free_space_path_to_planar_surface_um(
                axial_length_um=cfg.length,
                x_prime=state.rays[:, 1, 0],
                refractive_index=cfg.refractive_index,
                surface_incidence_angle_rad=float(cfg.next_surface_incidence_angle_rad),
            )
            group_delay_fs = group_delay_from_path_length_fs(
                path_length_um,
                cfg.refractive_index,
            )
            phi_geom_rad = phase_from_group_delay_rad(
                cfg.omega,
                group_delay_fs,
                omega0_rad_per_fs=cfg.omega0_rad_per_fs,
            )

    contribution = PhaseContribution(
        optic_name=cfg.optic_name,
        instance_name=cfg.instance_name,
        omega=cfg.omega,
        delta_omega_rad_per_fs=cfg.delta_omega_rad_per_fs,
        omega0_rad_per_fs=cfg.omega0_rad_per_fs,
        phi0_rad=phi0_rad,
        phi_geom_rad=phi_geom_rad,
        phi3_rad=phi3_rad,
        path_length_um=path_length_um,
        group_delay_fs=group_delay_fs,
    )
    return state_out, contribution
