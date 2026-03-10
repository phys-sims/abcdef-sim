"""Runtime adapter boundary between pipeline stages and pure ABCDEF kernels."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.states import RayState
from abcdef_sim.physics.abcdef.action import (
    free_space_path_to_surface_intersection_um,
    group_delay_from_path_length_fs,
    phase_from_group_delay_rad,
)
from abcdef_sim.physics.abcdef.phase_terms import martinez_k_center, phi0_rad_i, phi3_rad_i
from abcdef_sim.physics.abcdef.propagation import propagate_step
from abcdef_sim.physics.geometry.frames import LocalFrame1D
from abcdef_sim.physics.geometry.surfaces import SurfacePlane1D

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
        phi3_transport_like_rad = np.zeros_like(phi0_rad)
        phi3_phase_rad = np.zeros_like(phi0_rad)
        phi3_rad = np.zeros_like(phi0_rad)
        path_length_um = None
        group_delay_fs = None
    else:
        state_out = propagate_step(state, cfg.abcdef)
        k_center = martinez_k_center(cfg.omega)
        phi0_rad = phi0_rad_i(cfg.omega, cfg.length, cfg.refractive_index)
        phi3_transport_like_rad = phi3_rad_i(
            k_center,
            cfg.abcdef[:, 1, 2],
            state_out.rays[:, 0, 0],
        )
        phi3_phase_rad = phi3_rad_i(
            k_center,
            _phase_f_array(cfg),
            state_out.rays[:, 0, 0],
        )
        phi3_rad = phi3_phase_rad
        phi_geom_rad = None
        path_length_um = None
        group_delay_fs = None
        if cfg.action_model == "surface_intersection":
            if (
                cfg.entrance_surface_point_x_um is None
                or cfg.entrance_surface_point_z_um is None
                or cfg.local_axis_angle_rad is None
                or cfg.next_surface_point_x_um is None
                or cfg.next_surface_point_z_um is None
                or cfg.next_surface_normal_angle_rad is None
            ):
                raise ValueError(
                    "surface_intersection action_model requires complete surface metadata"
                )
            frame = LocalFrame1D(
                origin_x_um=float(cfg.entrance_surface_point_x_um),
                origin_z_um=float(cfg.entrance_surface_point_z_um),
                axis_angle_rad=float(cfg.local_axis_angle_rad),
            )
            plane = SurfacePlane1D(
                point_x_um=float(cfg.next_surface_point_x_um),
                point_z_um=float(cfg.next_surface_point_z_um),
                normal_angle_rad=float(cfg.next_surface_normal_angle_rad),
            )
            path_length_um = free_space_path_to_surface_intersection_um(
                x_um=state.rays[:, 0, 0],
                x_prime=state.rays[:, 1, 0],
                refractive_index=cfg.refractive_index,
                frame=frame,
                plane=plane,
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
        phi3_transport_like_rad=phi3_transport_like_rad,
        phi3_phase_rad=phi3_phase_rad,
        phi3_rad=phi3_rad,
        path_length_um=path_length_um,
        group_delay_fs=group_delay_fs,
    )
    return state_out, contribution


def _phase_f_array(cfg: OpticStageCfg) -> np.ndarray:
    f_transport = np.asarray(cfg.abcdef[:, 1, 2], dtype=np.float64).reshape(-1)
    if cfg.phase_f_variant == "transport_exact":
        return f_transport

    if (
        cfg.delta_omega_rad_per_fs is None
        or cfg.phase_grating_period_um is None
        or cfg.phase_grating_incidence_angle_rad is None
        or cfg.phase_grating_immersion_refractive_index is None
    ):
        raise ValueError("martinez_series2 phase_f_variant requires complete grating metadata")

    delta = np.asarray(cfg.delta_omega_rad_per_fs, dtype=np.float64).reshape(-1)
    theta2_ref = _diffraction_angle_from_geometry(
        omega0=float(cfg.omega0_rad_per_fs),
        period_um=float(cfg.phase_grating_period_um),
        incidence_angle_rad=float(cfg.phase_grating_incidence_angle_rad),
        diffraction_order=float(cfg.phase_grating_diffraction_order or -1),
    )
    n_medium = float(cfg.phase_grating_immersion_refractive_index)
    f_linear = (
        -(
            (2.0 * math.pi * 0.299792458 * n_medium**2)
            / (
                float(cfg.phase_grating_period_um)
                * float(cfg.omega0_rad_per_fs) ** 2
                * math.cos(theta2_ref)
            )
        )
        * delta
    )
    if float(np.dot(f_linear, f_transport)) < 0.0:
        f_linear = -f_linear
    return f_linear * (
        1.0 - (2.0 * delta / float(cfg.omega0_rad_per_fs)) + (0.5 * math.tan(theta2_ref) * f_linear)
    )


def _diffraction_angle_from_geometry(
    *,
    omega0: float,
    period_um: float,
    incidence_angle_rad: float,
    diffraction_order: float,
) -> float:
    wavelength_um = (2.0 * math.pi * 0.299792458) / float(omega0)
    diff_arg = -float(diffraction_order) * (wavelength_um / float(period_um)) - math.sin(
        float(incidence_angle_rad)
    )
    if diff_arg < -1.0 or diff_arg > 1.0:
        raise ValueError("diffraction geometry is outside the asin domain")
    return math.asin(diff_arg)
