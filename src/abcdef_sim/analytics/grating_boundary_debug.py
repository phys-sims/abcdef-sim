from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from abcdef_sim import BeamSpec, PulseSpec, StandaloneLaserSpec, run_abcdef, treacy_compressor_preset
from abcdef_sim.cache.backend import NullCacheBackend
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.specs import AbcdefCfg, FrameTransformCfg, FreeSpaceCfg, GratingCfg, ThickLensCfg
from abcdef_sim.data_models.states import RayState
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.physics.abcdef.adapters import _phase_f_array, apply_cfg
from abcdef_sim.physics.abcdef.dispersion import fit_phase_taylor_affine_detrended
from abcdef_sim.physics.abcdef.phase_terms import combine_phi_total_rad
from abcdef_sim.physics.abcdef.phase_terms import martinez_k, martinez_k_center
from abcdef_sim.physics.abcdef.pulse import build_standalone_laser_state
from abcdef_sim.physics.abcdef.treacy import compute_treacy_analytic_metrics
from abcdef_sim.pipeline._assembler import SystemAssembler
from abcdef_sim.runner import _initial_ray_state, _laser_spec_from_state, _resolve_runtime_cfg

NDArrayF = np.ndarray
_C_UM_PER_FS = 0.299792458

__all__ = [
    "SectionIVGratingBoundaryPoint",
    "SectionIVGratingBoundarySeries",
    "TreacyAlignmentAudit",
    "TreacyActualLocalMatrixPoint",
    "TreacyActualLocalMatrixSeries",
    "TreacyFittedSurfaceMapPoint",
    "TreacyFittedSurfaceMapSeries",
    "TreacyAdjacentPairSubsystemPoint",
    "TreacyAdjacentPairSubsystemSeries",
    "TreacyEndpointDecompositionPoint",
    "TreacyGratingBoundarySeries",
    "TreacyGratingBoundaryPoint",
    "run_treacy_alignment_audit",
    "run_treacy_actual_local_matrix_audit",
    "run_treacy_actual_local_matrix_series",
    "run_treacy_fitted_surface_map_audit",
    "run_treacy_fitted_surface_map_series",
    "run_treacy_adjacent_pair_subsystem_audit",
    "run_treacy_adjacent_pair_subsystem_series",
    "run_section_iv_grating_boundary_audit",
    "run_section_iv_grating_boundary_series",
    "run_treacy_endpoint_decomposition_study",
    "run_treacy_grating_boundary_series",
    "run_treacy_grating_boundary_audit",
]


@dataclass(frozen=True, slots=True)
class SectionIVGratingBoundarySeries:
    stage_name: Literal["g1", "g2"]
    omega_rad_per_fs: NDArrayF
    input_plane_x_um: NDArrayF
    input_plane_xprime: NDArrayF
    output_plane_x_um: NDArrayF
    output_plane_xprime: NDArrayF
    paired_output_x_um: NDArrayF
    paired_input_x_um: NDArrayF
    f_exact: NDArrayF
    f_paired: NDArrayF
    local_endpoint_exact_sample_k_rad: NDArrayF
    local_endpoint_paired_sample_k_rad: NDArrayF
    phi3_exact_sample_k_rad: NDArrayF
    phi3_exact_center_k_rad: NDArrayF
    phi3_paired_sample_k_rad: NDArrayF
    phi3_paired_center_k_rad: NDArrayF
    global_endpoint_exact_sample_k_rad: NDArrayF
    global_endpoint_paired_sample_k_rad: NDArrayF
    phi3_oracle_sample_k_rad: NDArrayF
    phi3_oracle_center_k_rad: NDArrayF
    configured_incidence_angle_deg: float
    magnification: float
    expected_b_um: float


@dataclass(frozen=True, slots=True)
class SectionIVGratingBoundaryPoint:
    stage_name: Literal["g1", "g2"]
    configured_incidence_angle_deg: float
    input_plane_x_rms_um: float
    output_plane_x_rms_um: float
    paired_output_x_rms_um: float
    output_x_pair_mismatch_rms_um: float
    f_exact_rms: float
    f_paired_rms: float
    f_pair_mismatch_rms: float
    output_x_pair_relative_rms: float
    f_pair_relative_rms: float
    local_endpoint_exact_sample_k_rms_rad: float
    local_endpoint_paired_sample_k_rms_rad: float
    global_endpoint_exact_sample_k_rms_rad: float
    global_endpoint_paired_sample_k_rms_rad: float
    phi3_exact_sample_k_rms_rad: float
    phi3_paired_sample_k_rms_rad: float
    phi3_oracle_sample_k_rms_rad: float
    phi3_pair_mismatch_rms_rad: float
    phi3_oracle_vs_paired_mismatch_rms_rad: float
    phi3_oracle_vs_exact_plus_global_endpoint_rms_rad: float


@dataclass(frozen=True, slots=True)
class TreacyGratingBoundarySeries:
    instance_name: str
    stage_index: int
    is_return_pass: bool
    omega_rad_per_fs: NDArrayF
    input_plane_x_um: NDArrayF
    input_plane_xprime: NDArrayF
    output_plane_x_um: NDArrayF
    surface_output_xprime: NDArrayF
    surface_coordinate_input_frame_um: NDArrayF
    surface_coordinate_output_frame_um: NDArrayF
    surface_intersection_distance_um: NDArrayF
    surface_hit_plane_residual_um: NDArrayF
    tangential_k_in_per_um: NDArrayF
    tangential_k_out_per_um: NDArrayF
    surface_local_d: NDArrayF
    incoming_state_f_local: NDArrayF
    groove_momentum_per_um: NDArrayF
    tangential_phase_match_residual_per_um: NDArrayF
    surface_phase_match_term_rad: NDArrayF
    groove_phase_term_rad: NDArrayF
    transport_like_phi3_rad: NDArrayF
    runtime_phase_phi3_rad: NDArrayF
    incidence_angle_rad: NDArrayF
    diffraction_angle_rad: NDArrayF
    configured_incidence_angle_deg: float


@dataclass(frozen=True, slots=True)
class TreacyGratingBoundaryPoint:
    instance_name: str
    stage_index: int
    is_return_pass: bool
    configured_incidence_angle_deg: float
    input_plane_x_rms_um: float
    surface_coordinate_rms_um: float
    surface_coordinate_minus_input_x_rms_um: float
    output_plane_x_rms_um: float
    surface_coordinate_output_frame_rms_um: float
    surface_coordinate_minus_output_x_rms_um: float
    surface_intersection_distance_rms_um: float
    max_abs_surface_hit_plane_residual_um: float
    rms_surface_hit_plane_residual_um: float
    max_abs_tangential_phase_match_residual_per_um: float
    rms_tangential_phase_match_residual_per_um: float
    surface_phase_match_term_rms_rad: float
    groove_phase_term_rms_rad: float
    transport_like_phi3_rms_rad: float
    runtime_phase_phi3_rms_rad: float
    mean_incidence_angle_deg: float
    mean_diffraction_angle_deg: float
    rms_incidence_angle_mismatch_deg: float
    max_abs_incidence_angle_mismatch_deg: float

    def to_dict(self) -> dict[str, float | str | bool | int]:
        payload = asdict(self)
        payload["instance_name"] = str(self.instance_name)
        return payload


@dataclass(frozen=True, slots=True)
class TreacyEndpointDecompositionPoint:
    beam_radius_mm: float
    length_to_mirror_um: float
    variant: str
    analytic_gdd_fs2: float
    analytic_tod_fs3: float
    raw_abcdef_gdd_fs2: float
    raw_abcdef_tod_fs3: float
    raw_abcdef_gdd_rel_error: float
    raw_abcdef_tod_rel_error: float
    weighted_rms_rad: float

    def to_dict(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["variant"] = str(self.variant)
        return payload


@dataclass(frozen=True, slots=True)
class TreacyAdjacentPairSubsystemSeries:
    pair_name: str
    first_instance_name: str
    second_instance_name: str
    omega_rad_per_fs: NDArrayF
    a: NDArrayF
    b_um: NDArrayF
    c_per_um: NDArrayF
    d: NDArrayF
    e_um: NDArrayF
    f: NDArrayF
    first_surface_output_x_um: NDArrayF
    first_surface_output_xprime: NDArrayF
    second_surface_input_x_um: NDArrayF
    second_surface_input_xprime: NDArrayF
    predicted_second_surface_x_um: NDArrayF
    predicted_second_surface_xprime: NDArrayF
    x_residual_um: NDArrayF
    xprime_residual: NDArrayF
    surface_anchored_a: NDArrayF
    surface_anchored_b_um: NDArrayF
    surface_anchored_c_per_um: NDArrayF
    surface_anchored_d: NDArrayF
    surface_anchored_e_um: NDArrayF
    surface_anchored_f: NDArrayF
    anchored_predicted_second_surface_x_um: NDArrayF
    anchored_predicted_second_surface_xprime: NDArrayF
    anchored_x_residual_um: NDArrayF
    anchored_xprime_residual: NDArrayF
    canonical_kernel_sample_k_rad: NDArrayF


@dataclass(frozen=True, slots=True)
class TreacyAdjacentPairSubsystemPoint:
    pair_name: str
    first_instance_name: str
    second_instance_name: str
    b_rms_um: float
    x_residual_rms_um: float
    x_relative_residual_rms: float
    xprime_residual_rms: float
    xprime_relative_residual_rms: float
    anchored_x_residual_rms_um: float
    anchored_x_relative_residual_rms: float
    anchored_xprime_residual_rms: float
    anchored_xprime_relative_residual_rms: float
    canonical_kernel_sample_k_rms_rad: float


@dataclass(frozen=True, slots=True)
class TreacyAlignmentAudit:
    g2_return_plane_normal_offset_um: float
    g2_return_plane_angle_diff_mod_pi_rad: float
    g1_return_plane_normal_offset_um: float
    g1_return_plane_angle_diff_mod_pi_rad: float
    fold_retro_axis_error_rad: float
    fold_point_roundtrip_error_um: float


@dataclass(frozen=True, slots=True)
class TreacyActualLocalMatrixSeries:
    instance_name: str
    stage_index: int
    is_return_pass: bool
    omega_rad_per_fs: NDArrayF
    input_surface_x_um: NDArrayF
    input_surface_xprime: NDArrayF
    output_surface_x_um: NDArrayF
    output_surface_xprime: NDArrayF
    a_actual: NDArrayF
    d_actual: NDArrayF
    f_actual: NDArrayF
    a_fixed: NDArrayF
    d_fixed: NDArrayF
    f_fixed: NDArrayF
    predicted_output_x_actual_um: NDArrayF
    predicted_output_xprime_actual: NDArrayF
    predicted_output_x_fixed_um: NDArrayF
    predicted_output_xprime_fixed: NDArrayF
    configured_incidence_angle_deg: float
    mean_actual_incidence_angle_deg: float
    rms_incidence_angle_mismatch_deg: float


@dataclass(frozen=True, slots=True)
class TreacyActualLocalMatrixPoint:
    instance_name: str
    stage_index: int
    is_return_pass: bool
    configured_incidence_angle_deg: float
    mean_actual_incidence_angle_deg: float
    rms_incidence_angle_mismatch_deg: float
    a_fixed_vs_actual_relative_rms: float
    d_fixed_vs_actual_relative_rms: float
    f_fixed_vs_actual_relative_rms: float
    actual_local_x_residual_rms_um: float
    actual_local_xprime_residual_rms: float
    fixed_model_x_residual_rms_um: float
    fixed_model_x_relative_residual_rms: float
    fixed_model_xprime_residual_rms: float
    fixed_model_xprime_relative_residual_rms: float


@dataclass(frozen=True, slots=True)
class TreacyFittedSurfaceMapSeries:
    instance_name: str
    stage_index: int
    is_return_pass: bool
    omega_rad_per_fs: NDArrayF
    input_surface_x_um: NDArrayF
    input_surface_xprime: NDArrayF
    output_surface_x_um: NDArrayF
    output_surface_xprime: NDArrayF
    fitted_a: NDArrayF
    fitted_b_um: NDArrayF
    fitted_c_per_um: NDArrayF
    fitted_d: NDArrayF
    fitted_e_um: NDArrayF
    fitted_f: NDArrayF
    predicted_output_surface_x_um: NDArrayF
    predicted_output_surface_xprime: NDArrayF
    fitted_generator_center_k_rad: NDArrayF
    configured_incidence_angle_deg: float


@dataclass(frozen=True, slots=True)
class TreacyFittedSurfaceMapPoint:
    instance_name: str
    stage_index: int
    is_return_pass: bool
    configured_incidence_angle_deg: float
    fitted_b_rms_um: float
    fitted_e_rms_um: float
    fitted_x_residual_rms_um: float
    fitted_x_relative_residual_rms: float
    fitted_xprime_residual_rms: float
    fitted_xprime_relative_residual_rms: float
    fitted_generator_center_k_rms_rad: float


@dataclass(frozen=True, slots=True)
class _StageTrace:
    index: int
    optic_spec: FrameTransformCfg | FreeSpaceCfg | GratingCfg | ThickLensCfg
    cfg: OpticStageCfg
    state_in: RayState
    state_out: RayState
    contribution: PhaseContribution


def run_treacy_grating_boundary_audit(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyGratingBoundaryPoint, ...]:
    laser = StandaloneLaserSpec(
        pulse=PulseSpec(
            width_fs=float(base_pulse_width_fs),
            center_wavelength_nm=center_wavelength_nm,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        ),
        beam=BeamSpec(radius_mm=float(beam_radius_mm), m2=1.0),
    )
    cfg = treacy_compressor_preset(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        length_to_mirror_um=length_to_mirror_um,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    trace = _trace_cfg_run(cfg, laser)
    return tuple(
        _boundary_point_from_series(_boundary_series_from_grating_trace(item))
        for item in trace
        if isinstance(item.optic_spec, GratingCfg)
    )


def run_section_iv_grating_boundary_series(
    *,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    center_wavelength_nm: float = 1030.0,
    focal_length_um: float = 300_000.0,
    a: float = 0.05,
    a_prime: float = 0.0,
    b: float = 0.0,
    n_samples: int = 256,
    pulse_width_fs: float = 100.0,
    span_scale: float = 1.0,
) -> tuple[SectionIVGratingBoundarySeries, ...]:
    from abcdef_sim.analytics.martinez_debug import (
        _section_iv_exact_grating_matrix,
        _section_iv_first_grating_matrix,
        _section_iv_second_grating_matrix,
        _section_iv_telescope_matrix,
    )

    omega = _section_iv_omega_grid(
        center_wavelength_nm=center_wavelength_nm,
        n_samples=n_samples,
        pulse_width_fs=pulse_width_fs,
        span_scale=span_scale,
    )
    omega0 = float(np.mean(omega))
    theta1_rad = math.radians(float(incidence_angle_deg))
    period_um = 1000.0 / float(line_density_lpmm)
    theta2_rad = _diffraction_angle_from_geometry(
        omega0=omega0,
        period_um=period_um,
        incidence_angle_rad=theta1_rad,
        diffraction_order=-1.0,
    )
    theta2_deg = math.degrees(theta2_rad)
    magnification = float(math.cos(theta2_rad) / math.cos(theta1_rad))
    expected_b_um = -float(focal_length_um) * (float(a) + float(a_prime)) / (magnification**2)

    telescope = _section_iv_telescope_matrix(
        omega.size,
        focal_length_um=float(focal_length_um),
        a=float(a),
        a_prime=float(a_prime),
        b=float(b),
    )
    g1_exact = _section_iv_exact_grating_matrix(
        omega,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
    )
    g2_exact = _section_iv_exact_grating_matrix(
        omega,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=theta2_deg,
    )

    rays0 = np.zeros((omega.size, 3, 1), dtype=np.float64)
    rays0[:, 2, 0] = 1.0
    rays_after_g1 = g1_exact @ rays0
    rays_before_g2 = telescope @ rays_after_g1
    rays_after_g2 = g2_exact @ rays_before_g2
    f_g1_exact = np.asarray(g1_exact[:, 1, 2], dtype=np.float64)
    f_g2_exact = np.asarray(g2_exact[:, 1, 2], dtype=np.float64)
    g1_paired = _section_iv_first_grating_matrix(magnification, f_g1_exact)
    g2_paired = _section_iv_second_grating_matrix(magnification, f_g1_exact)
    rays_after_g1_paired = g1_paired @ rays0
    rays_before_g2_paired = telescope @ rays_after_g1_paired
    rays_after_g2_paired = g2_paired @ rays_before_g2_paired

    k_sample = martinez_k(omega)
    k_center = np.full_like(omega, martinez_k_center(omega), dtype=np.float64)
    f_g2_paired = float(magnification) * f_g1_exact
    x_before_g2_paired = float(expected_b_um) * f_g1_exact
    x_after_g2_paired = x_before_g2_paired / float(magnification)
    zeros = np.zeros_like(omega, dtype=np.float64)

    return (
        SectionIVGratingBoundarySeries(
            stage_name="g1",
            omega_rad_per_fs=np.asarray(omega, dtype=np.float64),
            input_plane_x_um=np.asarray(rays0[:, 0, 0], dtype=np.float64),
            input_plane_xprime=np.asarray(rays0[:, 1, 0], dtype=np.float64),
            output_plane_x_um=np.asarray(rays_after_g1[:, 0, 0], dtype=np.float64),
            output_plane_xprime=np.asarray(rays_after_g1[:, 1, 0], dtype=np.float64),
            paired_output_x_um=np.zeros_like(omega, dtype=np.float64),
            paired_input_x_um=np.zeros_like(omega, dtype=np.float64),
            f_exact=f_g1_exact,
            f_paired=np.asarray(f_g1_exact, dtype=np.float64),
            local_endpoint_exact_sample_k_rad=np.zeros_like(omega, dtype=np.float64),
            local_endpoint_paired_sample_k_rad=np.zeros_like(omega, dtype=np.float64),
            phi3_exact_sample_k_rad=0.5
            * k_sample
            * f_g1_exact
            * np.asarray(rays_after_g1[:, 0, 0], dtype=np.float64),
            phi3_exact_center_k_rad=0.5
            * k_center
            * f_g1_exact
            * np.asarray(rays_after_g1[:, 0, 0], dtype=np.float64),
            phi3_paired_sample_k_rad=np.zeros_like(omega, dtype=np.float64),
            phi3_paired_center_k_rad=np.zeros_like(omega, dtype=np.float64),
            global_endpoint_exact_sample_k_rad=np.zeros_like(omega, dtype=np.float64),
            global_endpoint_paired_sample_k_rad=np.zeros_like(omega, dtype=np.float64),
            phi3_oracle_sample_k_rad=np.zeros_like(omega, dtype=np.float64),
            phi3_oracle_center_k_rad=np.zeros_like(omega, dtype=np.float64),
            configured_incidence_angle_deg=float(incidence_angle_deg),
            magnification=float(magnification),
            expected_b_um=float(expected_b_um),
        ),
        SectionIVGratingBoundarySeries(
            stage_name="g2",
            omega_rad_per_fs=np.asarray(omega, dtype=np.float64),
            input_plane_x_um=np.asarray(rays_before_g2[:, 0, 0], dtype=np.float64),
            input_plane_xprime=np.asarray(rays_before_g2[:, 1, 0], dtype=np.float64),
            output_plane_x_um=np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64),
            output_plane_xprime=np.asarray(rays_after_g2[:, 1, 0], dtype=np.float64),
            paired_output_x_um=np.asarray(x_after_g2_paired, dtype=np.float64),
            paired_input_x_um=np.asarray(x_before_g2_paired, dtype=np.float64),
            f_exact=f_g2_exact,
            f_paired=np.asarray(f_g2_paired, dtype=np.float64),
            local_endpoint_exact_sample_k_rad=0.5
            * k_sample
            * (
                np.asarray(rays_before_g2[:, 0, 0], dtype=np.float64)
                * np.asarray(rays_before_g2[:, 1, 0], dtype=np.float64)
                - np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64)
                * np.asarray(rays_after_g2[:, 1, 0], dtype=np.float64)
            ),
            local_endpoint_paired_sample_k_rad=0.5
            * k_sample
            * (
                np.asarray(rays_before_g2_paired[:, 0, 0], dtype=np.float64)
                * np.asarray(rays_before_g2_paired[:, 1, 0], dtype=np.float64)
                - np.asarray(rays_after_g2_paired[:, 0, 0], dtype=np.float64)
                * np.asarray(rays_after_g2_paired[:, 1, 0], dtype=np.float64)
            ),
            phi3_exact_sample_k_rad=0.5
            * k_sample
            * f_g2_exact
            * np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64),
            phi3_exact_center_k_rad=0.5
            * k_center
            * f_g2_exact
            * np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64),
            phi3_paired_sample_k_rad=0.5
            * k_sample
            * np.asarray(f_g2_paired, dtype=np.float64)
            * np.asarray(x_after_g2_paired, dtype=np.float64),
            phi3_paired_center_k_rad=0.5
            * k_center
            * np.asarray(f_g2_paired, dtype=np.float64)
            * np.asarray(x_after_g2_paired, dtype=np.float64),
            global_endpoint_exact_sample_k_rad=0.5
            * k_sample
            * (0.0 - (np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64) * np.asarray(rays_after_g2[:, 1, 0], dtype=np.float64))),
            global_endpoint_paired_sample_k_rad=0.5
            * k_sample
            * (
                0.0
                - (
                    np.asarray(rays_after_g2_paired[:, 0, 0], dtype=np.float64)
                    * np.asarray(rays_after_g2_paired[:, 1, 0], dtype=np.float64)
                )
            ),
            phi3_oracle_sample_k_rad=0.5
            * k_sample
            * float(expected_b_um)
            * (float(magnification) * f_g1_exact) ** 2,
            phi3_oracle_center_k_rad=0.5
            * k_center
            * float(expected_b_um)
            * (float(magnification) * f_g1_exact) ** 2,
            configured_incidence_angle_deg=float(theta2_deg),
            magnification=float(magnification),
            expected_b_um=float(expected_b_um),
        ),
    )


def run_section_iv_grating_boundary_audit(
    *,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    center_wavelength_nm: float = 1030.0,
    focal_length_um: float = 300_000.0,
    a: float = 0.05,
    a_prime: float = 0.0,
    b: float = 0.0,
    n_samples: int = 256,
    pulse_width_fs: float = 100.0,
    span_scale: float = 1.0,
) -> tuple[SectionIVGratingBoundaryPoint, ...]:
    return tuple(
        _section_iv_boundary_point_from_series(series)
        for series in run_section_iv_grating_boundary_series(
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            center_wavelength_nm=center_wavelength_nm,
            focal_length_um=focal_length_um,
            a=a,
            a_prime=a_prime,
            b=b,
            n_samples=n_samples,
            pulse_width_fs=pulse_width_fs,
            span_scale=span_scale,
        )
    )


def run_treacy_grating_boundary_series(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyGratingBoundarySeries, ...]:
    laser = StandaloneLaserSpec(
        pulse=PulseSpec(
            width_fs=float(base_pulse_width_fs),
            center_wavelength_nm=center_wavelength_nm,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        ),
        beam=BeamSpec(radius_mm=float(beam_radius_mm), m2=1.0),
    )
    cfg = treacy_compressor_preset(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        length_to_mirror_um=length_to_mirror_um,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    trace = _trace_cfg_run(cfg, laser)
    return tuple(
        _boundary_series_from_grating_trace(item)
        for item in trace
        if isinstance(item.optic_spec, GratingCfg)
    )


def run_treacy_adjacent_pair_subsystem_series(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyAdjacentPairSubsystemSeries, ...]:
    laser = StandaloneLaserSpec(
        pulse=PulseSpec(
            width_fs=float(base_pulse_width_fs),
            center_wavelength_nm=center_wavelength_nm,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        ),
        beam=BeamSpec(radius_mm=float(beam_radius_mm), m2=1.0),
    )
    cfg = treacy_compressor_preset(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        length_to_mirror_um=length_to_mirror_um,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    trace = _trace_cfg_run(cfg, laser)
    return tuple(
        _adjacent_pair_subsystem_series(first, second, middle)
        for first, second, middle in _treacy_adjacent_grating_pairs(trace)
    )


def run_treacy_adjacent_pair_subsystem_audit(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyAdjacentPairSubsystemPoint, ...]:
    return tuple(
        _adjacent_pair_subsystem_point_from_series(series)
        for series in run_treacy_adjacent_pair_subsystem_series(
            beam_radius_mm=beam_radius_mm,
            base_pulse_width_fs=base_pulse_width_fs,
            center_wavelength_nm=center_wavelength_nm,
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            separation_um=separation_um,
            length_to_mirror_um=length_to_mirror_um,
            diffraction_order=diffraction_order,
            n_passes=n_passes,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        )
    )


def run_treacy_alignment_audit(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 100_000.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> TreacyAlignmentAudit:
    laser = StandaloneLaserSpec(
        pulse=PulseSpec(
            width_fs=float(base_pulse_width_fs),
            center_wavelength_nm=center_wavelength_nm,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        ),
        beam=BeamSpec(radius_mm=float(beam_radius_mm), m2=1.0),
    )
    cfg = treacy_compressor_preset(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        length_to_mirror_um=length_to_mirror_um,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    trace = _trace_cfg_run(cfg, laser)
    by_name = {item.cfg.instance_name: item.cfg for item in trace}
    for required in ("g1", "g2", "g2_return", "g1_return", "to_fold", "from_fold"):
        if required not in by_name:
            raise ValueError(f"Treacy alignment audit requires stage {required!r}")

    g1 = by_name["g1"]
    g2 = by_name["g2"]
    g2_return = by_name["g2_return"]
    g1_return = by_name["g1_return"]
    to_fold = by_name["to_fold"]
    from_fold = by_name["from_fold"]
    if (
        to_fold.local_axis_angle_rad is None
        or to_fold.next_surface_point_x_um is None
        or to_fold.next_surface_point_z_um is None
        or from_fold.local_axis_angle_rad is None
        or from_fold.entrance_surface_point_x_um is None
        or from_fold.entrance_surface_point_z_um is None
    ):
        raise ValueError("Treacy alignment audit requires complete fold geometry metadata")

    return TreacyAlignmentAudit(
        g2_return_plane_normal_offset_um=_parallel_plane_normal_offset(g2, g2_return),
        g2_return_plane_angle_diff_mod_pi_rad=_parallel_plane_angle_diff_mod_pi(g2, g2_return),
        g1_return_plane_normal_offset_um=_parallel_plane_normal_offset(g1, g1_return),
        g1_return_plane_angle_diff_mod_pi_rad=_parallel_plane_angle_diff_mod_pi(g1, g1_return),
        fold_retro_axis_error_rad=_wrapped_angle_abs(
            float(from_fold.local_axis_angle_rad) - float(to_fold.local_axis_angle_rad) - math.pi
        ),
        fold_point_roundtrip_error_um=math.hypot(
            float(from_fold.entrance_surface_point_x_um) - float(to_fold.next_surface_point_x_um),
            float(from_fold.entrance_surface_point_z_um) - float(to_fold.next_surface_point_z_um),
        ),
    )


def run_treacy_actual_local_matrix_series(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyActualLocalMatrixSeries, ...]:
    laser = StandaloneLaserSpec(
        pulse=PulseSpec(
            width_fs=float(base_pulse_width_fs),
            center_wavelength_nm=center_wavelength_nm,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        ),
        beam=BeamSpec(radius_mm=float(beam_radius_mm), m2=1.0),
    )
    cfg = treacy_compressor_preset(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        length_to_mirror_um=length_to_mirror_um,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    trace = _trace_cfg_run(cfg, laser)
    return tuple(
        _actual_local_matrix_series_from_trace(item)
        for item in trace
        if isinstance(item.optic_spec, GratingCfg)
    )


def run_treacy_actual_local_matrix_audit(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyActualLocalMatrixPoint, ...]:
    return tuple(
        _actual_local_matrix_point_from_series(series)
        for series in run_treacy_actual_local_matrix_series(
            beam_radius_mm=beam_radius_mm,
            base_pulse_width_fs=base_pulse_width_fs,
            center_wavelength_nm=center_wavelength_nm,
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            separation_um=separation_um,
            length_to_mirror_um=length_to_mirror_um,
            diffraction_order=diffraction_order,
            n_passes=n_passes,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        )
    )


def run_treacy_fitted_surface_map_series(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyFittedSurfaceMapSeries, ...]:
    laser = StandaloneLaserSpec(
        pulse=PulseSpec(
            width_fs=float(base_pulse_width_fs),
            center_wavelength_nm=center_wavelength_nm,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        ),
        beam=BeamSpec(radius_mm=float(beam_radius_mm), m2=1.0),
    )
    cfg = treacy_compressor_preset(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        length_to_mirror_um=length_to_mirror_um,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    trace = _trace_cfg_run(cfg, laser)
    return tuple(
        _fitted_surface_map_series_from_trace(item)
        for item in trace
        if isinstance(item.optic_spec, GratingCfg)
    )


def run_treacy_fitted_surface_map_audit(
    *,
    beam_radius_mm: float = 1000.0,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyFittedSurfaceMapPoint, ...]:
    return tuple(
        _fitted_surface_map_point_from_series(series)
        for series in run_treacy_fitted_surface_map_series(
            beam_radius_mm=beam_radius_mm,
            base_pulse_width_fs=base_pulse_width_fs,
            center_wavelength_nm=center_wavelength_nm,
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            separation_um=separation_um,
            length_to_mirror_um=length_to_mirror_um,
            diffraction_order=diffraction_order,
            n_passes=n_passes,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        )
    )


def run_treacy_endpoint_decomposition_study(
    *,
    beam_radii_mm: tuple[float, ...] = (10.0, 100.0, 1000.0),
    mirror_lengths_um: tuple[float, ...] = (0.0, 100_000.0),
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyEndpointDecompositionPoint, ...]:
    points: list[TreacyEndpointDecompositionPoint] = []
    for length_to_mirror_um in mirror_lengths_um:
        analytic = compute_treacy_analytic_metrics(
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            separation_um=separation_um,
            wavelength_nm=center_wavelength_nm,
            diffraction_order=diffraction_order,
            n_passes=n_passes,
        )
        for beam_radius_mm in beam_radii_mm:
            laser = StandaloneLaserSpec(
                pulse=PulseSpec(
                    width_fs=float(base_pulse_width_fs),
                    center_wavelength_nm=center_wavelength_nm,
                    n_samples=n_samples,
                    time_window_fs=time_window_fs,
                ),
                beam=BeamSpec(radius_mm=float(beam_radius_mm), m2=1.0),
            )
            cfg = treacy_compressor_preset(
                line_density_lpmm=line_density_lpmm,
                incidence_angle_deg=incidence_angle_deg,
                separation_um=separation_um,
                length_to_mirror_um=length_to_mirror_um,
                diffraction_order=diffraction_order,
                n_passes=n_passes,
            )
            result = run_abcdef(cfg, laser)
            trace = _trace_cfg_run(cfg, laser)
            omega = np.asarray(result.pipeline_result.omega, dtype=np.float64)
            delta = np.asarray(result.pipeline_result.delta_omega_rad_per_fs, dtype=np.float64)
            omega0 = float(result.pipeline_result.omega0_rad_per_fs)
            weights = np.asarray(result.fit.weights, dtype=np.float64)
            phase_terms = _treacy_nonlocal_phase_terms(result)
            phi2_exact_center = _treacy_global_endpoint_phase(trace, k_mode="center")
            phi2_exact_sample = _treacy_global_endpoint_phase(trace, k_mode="sample")
            phi3_exact_center = _treacy_exact_local_phi3_total(trace, k_mode="center")
            phi3_exact_sample = _treacy_exact_local_phi3_total(trace, k_mode="sample")
            phi3_surface_hit_center = _treacy_surface_hit_phi3_total(trace, k_mode="center")
            phi3_surface_hit_sample = _treacy_surface_hit_phi3_total(trace, k_mode="sample")
            phi_fitted_surface_generator_center = _treacy_fitted_surface_generator_total(
                trace,
                k_mode="center",
            )
            phi_surface_anchored_pair_kernel_center = _treacy_surface_anchored_pair_kernel_total(
                trace,
                k_mode="center",
            )
            phi3_canonical_surface_sym_center = _treacy_surface_canonical_phi_total(
                trace,
                k_mode="center",
                image="sym",
            )
            phi3_canonical_surface_sym_sample = _treacy_surface_canonical_phi_total(
                trace,
                k_mode="sample",
                image="sym",
            )
            phi_canonical_surface_raw_center = _treacy_surface_canonical_phi_total(
                trace,
                k_mode="center",
                image="raw",
            )
            phi3_surface_anchored_runtime_center = _treacy_surface_anchored_phi3_total(
                trace,
                k_mode="center",
                f_mode="runtime_phase",
            )
            phi3_surface_anchored_transport_center = _treacy_surface_anchored_phi3_total(
                trace,
                k_mode="center",
                f_mode="transport_exact",
            )
            phi3_configured_surface_slope_stage_center = _treacy_configured_surface_slope_phi3_total(
                trace,
                k_mode="center",
                x_mode="stage",
            )
            phi3_configured_surface_slope_anchored_center = (
                _treacy_configured_surface_slope_phi3_total(
                    trace,
                    k_mode="center",
                    x_mode="surface_anchored",
                )
            )
            phi3_actual_local_f_stage_center = _treacy_actual_local_f_phi3_total(
                trace,
                k_mode="center",
                x_mode="stage",
            )

            variant_map = {
                "current": np.asarray(result.pipeline_result.phi_total_rad, dtype=np.float64),
                "exact_local_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_exact_center,
                    phase_terms["phi4"],
                ),
                "exact_local_sample_global_sample": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_sample,
                    phi3_exact_sample,
                    phase_terms["phi4"],
                ),
                "exact_local_center_global_sample": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_sample,
                    phi3_exact_center,
                    phase_terms["phi4"],
                ),
                "exact_local_sample_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_exact_sample,
                    phase_terms["phi4"],
                ),
                "surface_hit_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_surface_hit_center,
                    phase_terms["phi4"],
                ),
                "surface_hit_sample_global_sample": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_sample,
                    phi3_surface_hit_sample,
                    phase_terms["phi4"],
                ),
                "fitted_surface_generator_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi_fitted_surface_generator_center,
                    phase_terms["phi4"],
                ),
                "fitted_surface_generator_plus_anchored_pairs_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi_surface_anchored_pair_kernel_center,
                    phi_fitted_surface_generator_center,
                    phase_terms["phi4"],
                ),
                "fitted_surface_generator_plus_anchored_pairs_plus_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center + phi_surface_anchored_pair_kernel_center,
                    phi_fitted_surface_generator_center,
                    phase_terms["phi4"],
                ),
                "canonical_surface_sym_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_canonical_surface_sym_center,
                    phase_terms["phi4"],
                ),
                "canonical_surface_sym_sample_global_sample": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_sample,
                    phi3_canonical_surface_sym_sample,
                    phase_terms["phi4"],
                ),
                "canonical_surface_raw_center_no_global": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi_canonical_surface_raw_center,
                    phase_terms["phi4"],
                ),
                "surface_anchored_runtime_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_surface_anchored_runtime_center,
                    phase_terms["phi4"],
                ),
                "surface_anchored_transport_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_surface_anchored_transport_center,
                    phase_terms["phi4"],
                ),
                "configured_surface_slope_stage_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_configured_surface_slope_stage_center,
                    phase_terms["phi4"],
                ),
                "configured_surface_slope_anchored_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_configured_surface_slope_anchored_center,
                    phase_terms["phi4"],
                ),
                "actual_local_f_stage_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    phi3_actual_local_f_stage_center,
                    phase_terms["phi4"],
                ),
                "incoming_state_g2_surface_center_global_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    phi2_exact_center,
                    _treacy_incoming_state_phi3_total(
                        trace,
                        k_mode="center",
                        target_instances=("g2",),
                    ),
                    phase_terms["phi4"],
                ),
                "conjugate_pair_f2sq_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    _treacy_conjugate_pair_phase_total(trace, mode="f2_exact_sq"),
                    phase_terms["phi4"],
                ),
                "conjugate_pair_mf1sq_center": combine_phi_total_rad(
                    phase_terms["phi_geom"],
                    phase_terms["filter_phase"],
                    phase_terms["phi1"],
                    _treacy_conjugate_pair_phase_total(trace, mode="mf1_sq"),
                    phase_terms["phi4"],
                ),
            }
            for variant, phase in variant_map.items():
                fit = fit_phase_taylor_affine_detrended(
                    delta,
                    phase,
                    omega0_rad_per_fs=omega0,
                    weights=weights,
                    order=4,
                )
                gdd = float(fit.coefficients_rad[2])
                tod = float(fit.coefficients_rad[3])
                points.append(
                    TreacyEndpointDecompositionPoint(
                        beam_radius_mm=float(beam_radius_mm),
                        length_to_mirror_um=float(length_to_mirror_um),
                        variant=variant,
                        analytic_gdd_fs2=float(analytic.gdd_fs2),
                        analytic_tod_fs3=float(analytic.tod_fs3),
                        raw_abcdef_gdd_fs2=gdd,
                        raw_abcdef_tod_fs3=tod,
                        raw_abcdef_gdd_rel_error=_relative_error(gdd, analytic.gdd_fs2),
                        raw_abcdef_tod_rel_error=_relative_error(tod, analytic.tod_fs3),
                        weighted_rms_rad=float(fit.weighted_rms_rad),
                    )
                )
    return tuple(points)


def _trace_cfg_run(
    cfg: AbcdefCfg,
    laser: StandaloneLaserSpec,
) -> tuple[_StageTrace, ...]:
    initial_laser_state = build_standalone_laser_state(laser)
    internal_laser_spec = _laser_spec_from_state(initial_laser_state)
    resolved_cfg = _resolve_runtime_cfg(cfg, internal_laser_spec)
    assembler = SystemAssembler(
        factory=OpticFactory.default(),
        cfg_gen=OpticStageCfgGenerator(cache=NullCacheBackend()),
    )
    stage_cfgs = assembler.build_optic_cfgs(
        resolved_cfg.to_preset(),
        internal_laser_spec,
        policy=None,
    )
    resolved_optics = tuple(resolved_cfg.optics)
    state = _initial_ray_state(cfg=resolved_cfg, laser=internal_laser_spec)
    trace: list[_StageTrace] = []
    for index, (stage_cfg, optic_spec) in enumerate(zip(stage_cfgs, resolved_optics, strict=True)):
        state_in = state
        state, contribution = apply_cfg(state, stage_cfg, policy=None)
        trace.append(
            _StageTrace(
                index=index,
                optic_spec=optic_spec,
                cfg=stage_cfg,
                state_in=state_in,
                state_out=state,
                contribution=contribution,
            )
        )
    return tuple(trace)


def _boundary_series_from_grating_trace(item: _StageTrace) -> TreacyGratingBoundarySeries:
    if not isinstance(item.optic_spec, GratingCfg):
        raise TypeError(f"Expected GratingCfg, got {type(item.optic_spec).__name__}")

    cfg = item.cfg
    optic = item.optic_spec
    omega = np.asarray(cfg.omega, dtype=np.float64).reshape(-1)
    x_in = np.asarray(item.state_in.rays[:, 0, 0], dtype=np.float64)
    x_prime_in = np.asarray(item.state_in.rays[:, 1, 0], dtype=np.float64)
    x_out = np.asarray(item.state_out.rays[:, 0, 0], dtype=np.float64)

    surface_state = _evaluate_grating_surface_state(
        item,
        x_in=x_in,
        x_prime_in=x_prime_in,
    )
    surface_coordinate = np.asarray(surface_state["surface_coordinate_input_frame_um"], dtype=np.float64)
    surface_coordinate_output_frame = np.asarray(
        surface_state["surface_coordinate_output_frame_um"],
        dtype=np.float64,
    )
    surface_distance = np.asarray(surface_state["surface_intersection_distance_um"], dtype=np.float64)
    surface_hit_plane_residual = np.asarray(
        surface_state["surface_hit_plane_residual_um"],
        dtype=np.float64,
    )
    theta_in = np.asarray(surface_state["incidence_angle_rad"], dtype=np.float64)
    theta_out = np.asarray(surface_state["diffraction_angle_rad"], dtype=np.float64)
    surface_output_xprime = np.asarray(surface_state["surface_output_xprime"], dtype=np.float64)
    surface_local_d = np.asarray(surface_state["surface_local_d"], dtype=np.float64)
    incoming_state_f_local = surface_output_xprime - (surface_local_d * x_prime_in)
    tangential_phase_match_residual = (
        martinez_k(omega)
        * (surface_state["dir_t_out"] + surface_state["dir_t_in"])
    ) - (float(optic.diffraction_order) * (2.0 * np.pi / float(surface_state["period_um"])))
    groove_phase_term_rad = (
        float(optic.diffraction_order)
        * (2.0 * np.pi / float(surface_state["period_um"]))
        * surface_coordinate_output_frame
    )
    surface_phase_match_term_rad = (
        martinez_k(omega)
        * (surface_state["dir_t_out"] + surface_state["dir_t_in"])
        * surface_coordinate_output_frame
    )
    transport_like_phi3 = _zero_or_array(
        item.contribution.phi3_transport_like_rad,
        template=omega,
    )
    runtime_phase_phi3 = _zero_or_array(
        item.contribution.phi3_phase_rad,
        template=omega,
    )

    return TreacyGratingBoundarySeries(
        instance_name=str(cfg.instance_name),
        stage_index=int(item.index),
        is_return_pass=bool(cfg.instance_name.endswith("_return")),
        omega_rad_per_fs=np.asarray(omega, dtype=np.float64),
        input_plane_x_um=np.asarray(x_in, dtype=np.float64),
        input_plane_xprime=np.asarray(x_prime_in, dtype=np.float64),
        output_plane_x_um=np.asarray(x_out, dtype=np.float64),
        surface_output_xprime=np.asarray(surface_output_xprime, dtype=np.float64),
        surface_coordinate_input_frame_um=np.asarray(surface_coordinate, dtype=np.float64),
        surface_coordinate_output_frame_um=np.asarray(surface_coordinate_output_frame, dtype=np.float64),
        surface_intersection_distance_um=np.asarray(surface_distance, dtype=np.float64),
        surface_hit_plane_residual_um=np.asarray(surface_hit_plane_residual, dtype=np.float64),
        tangential_k_in_per_um=martinez_k(omega) * surface_state["dir_t_in"],
        tangential_k_out_per_um=martinez_k(omega) * surface_state["dir_t_out"],
        surface_local_d=np.asarray(surface_local_d, dtype=np.float64),
        incoming_state_f_local=np.asarray(incoming_state_f_local, dtype=np.float64),
        groove_momentum_per_um=np.full_like(
            omega,
            float(optic.diffraction_order) * (2.0 * np.pi / float(surface_state["period_um"])),
            dtype=np.float64,
        ),
        tangential_phase_match_residual_per_um=np.asarray(
            tangential_phase_match_residual,
            dtype=np.float64,
        ),
        surface_phase_match_term_rad=np.asarray(surface_phase_match_term_rad, dtype=np.float64),
        groove_phase_term_rad=np.asarray(groove_phase_term_rad, dtype=np.float64),
        transport_like_phi3_rad=np.asarray(transport_like_phi3, dtype=np.float64),
        runtime_phase_phi3_rad=np.asarray(runtime_phase_phi3, dtype=np.float64),
        incidence_angle_rad=np.asarray(theta_in, dtype=np.float64),
        diffraction_angle_rad=np.asarray(theta_out, dtype=np.float64),
        configured_incidence_angle_deg=float(optic.incidence_angle_deg),
    )


def _evaluate_grating_surface_state(
    item: _StageTrace,
    *,
    x_in: NDArrayF,
    x_prime_in: NDArrayF,
) -> dict[str, NDArrayF | float]:
    if not isinstance(item.optic_spec, GratingCfg):
        raise TypeError(f"Expected GratingCfg, got {type(item.optic_spec).__name__}")

    cfg = item.cfg
    optic = item.optic_spec
    if (
        cfg.local_axis_angle_rad is None
        or cfg.entrance_surface_point_x_um is None
        or cfg.entrance_surface_point_z_um is None
        or cfg.entrance_surface_normal_angle_rad is None
    ):
        raise ValueError("grating boundary audit requires complete surface metadata")

    omega = np.asarray(cfg.omega, dtype=np.float64).reshape(-1)
    refractive_index = np.asarray(cfg.refractive_index, dtype=np.float64).reshape(-1)
    x_in_arr = np.asarray(x_in, dtype=np.float64).reshape(-1)
    x_prime_arr = np.asarray(x_prime_in, dtype=np.float64).reshape(-1)
    if x_in_arr.shape != omega.shape or x_prime_arr.shape != omega.shape:
        raise ValueError("x_in and x_prime_in must match cfg.omega shape")

    axis_angle = float(cfg.local_axis_angle_rad)
    surface_point_x = float(cfg.entrance_surface_point_x_um)
    surface_point_z = float(cfg.entrance_surface_point_z_um)
    normal_angle = float(cfg.entrance_surface_normal_angle_rad)
    tangent_x = math.cos(normal_angle)
    tangent_z = -math.sin(normal_angle)
    normal_x = math.sin(normal_angle)
    normal_z = math.cos(normal_angle)

    x_hat_x = math.cos(axis_angle)
    x_hat_z = -math.sin(axis_angle)
    z_hat_x = math.sin(axis_angle)
    z_hat_z = math.cos(axis_angle)

    input_origin_x = surface_point_x + (x_in_arr * x_hat_x)
    input_origin_z = surface_point_z + (x_in_arr * x_hat_z)

    local_slope = x_prime_arr / refractive_index
    dir_x = z_hat_x + (local_slope * x_hat_x)
    dir_z = z_hat_z + (local_slope * x_hat_z)
    dir_norm = np.hypot(dir_x, dir_z)
    dir_x = dir_x / dir_norm
    dir_z = dir_z / dir_norm

    numerator = normal_x * (surface_point_x - input_origin_x) + normal_z * (
        surface_point_z - input_origin_z
    )
    denominator = (normal_x * dir_x) + (normal_z * dir_z)
    surface_distance = numerator / denominator
    hit_x = input_origin_x + (surface_distance * dir_x)
    hit_z = input_origin_z + (surface_distance * dir_z)

    surface_coordinate_input = ((hit_x - surface_point_x) * tangent_x) + (
        (hit_z - surface_point_z) * tangent_z
    )
    surface_hit_plane_residual = ((hit_x - surface_point_x) * normal_x) + (
        (hit_z - surface_point_z) * normal_z
    )

    dir_t_in = (dir_x * tangent_x) + (dir_z * tangent_z)
    dir_n_in = (dir_x * normal_x) + (dir_z * normal_z)
    theta_in = np.arctan2(-dir_t_in, dir_n_in)

    period_um = 1000.0 / float(optic.line_density_lpmm)
    wavelength_um = (2.0 * np.pi * _C_UM_PER_FS) / omega
    diffraction_arg = -float(optic.diffraction_order) * (wavelength_um / period_um) - np.sin(theta_in)
    theta_out = np.arcsin(np.clip(diffraction_arg, -1.0, 1.0))

    dir_t_out = -np.sin(theta_out)
    theta_out_center = _diffraction_angle_from_geometry(
        omega0=float(cfg.omega0_rad_per_fs),
        period_um=period_um,
        incidence_angle_rad=math.radians(float(optic.incidence_angle_deg)),
        diffraction_order=float(optic.diffraction_order),
    )
    output_axis_angle = normal_angle - float(theta_out_center)
    output_x_hat_x = math.cos(output_axis_angle)
    output_x_hat_z = -math.sin(output_axis_angle)
    output_z_hat_x = math.sin(output_axis_angle)
    output_z_hat_z = math.cos(output_axis_angle)
    dir_out_x = (normal_x * np.cos(theta_out)) + (tangent_x * (-np.sin(theta_out)))
    dir_out_z = (normal_z * np.cos(theta_out)) + (tangent_z * (-np.sin(theta_out)))
    dir_out_t = (dir_out_x * output_x_hat_x) + (dir_out_z * output_x_hat_z)
    dir_out_n = (dir_out_x * output_z_hat_x) + (dir_out_z * output_z_hat_z)
    surface_output_xprime = refractive_index * (dir_out_t / dir_out_n)
    surface_coordinate_output = ((hit_x - surface_point_x) * output_x_hat_x) + (
        (hit_z - surface_point_z) * output_x_hat_z
    )
    surface_local_d = np.cos(theta_in) / np.cos(theta_out)

    return {
        "surface_coordinate_input_frame_um": np.asarray(surface_coordinate_input, dtype=np.float64),
        "surface_coordinate_output_frame_um": np.asarray(surface_coordinate_output, dtype=np.float64),
        "surface_intersection_distance_um": np.asarray(surface_distance, dtype=np.float64),
        "surface_hit_plane_residual_um": np.asarray(surface_hit_plane_residual, dtype=np.float64),
        "incidence_angle_rad": np.asarray(theta_in, dtype=np.float64),
        "diffraction_angle_rad": np.asarray(theta_out, dtype=np.float64),
        "surface_output_xprime": np.asarray(surface_output_xprime, dtype=np.float64),
        "surface_local_d": np.asarray(surface_local_d, dtype=np.float64),
        "dir_t_in": np.asarray(dir_t_in, dtype=np.float64),
        "dir_t_out": np.asarray(dir_t_out, dtype=np.float64),
        "period_um": float(period_um),
    }


def _linearized_grating_surface_input_map(
    item: _StageTrace,
    *,
    x_step_um: float = 1.0,
    xprime_step: float = 1.0e-3,
) -> NDArrayF:
    zero = np.zeros_like(np.asarray(item.cfg.omega, dtype=np.float64).reshape(-1), dtype=np.float64)
    base = _evaluate_grating_surface_state(item, x_in=zero, x_prime_in=zero)
    dx = _evaluate_grating_surface_state(
        item,
        x_in=np.full_like(zero, float(x_step_um), dtype=np.float64),
        x_prime_in=zero,
    )
    dp = _evaluate_grating_surface_state(
        item,
        x_in=zero,
        x_prime_in=np.full_like(zero, float(xprime_step), dtype=np.float64),
    )
    return _fit_affine_map_from_outputs(
        base_x=np.asarray(base["surface_coordinate_input_frame_um"], dtype=np.float64),
        base_xprime=zero,
        dx_x=np.asarray(dx["surface_coordinate_input_frame_um"], dtype=np.float64),
        dx_xprime=zero,
        dp_x=np.asarray(dp["surface_coordinate_input_frame_um"], dtype=np.float64),
        dp_xprime=np.full_like(zero, float(xprime_step), dtype=np.float64),
        x_step_um=float(x_step_um),
        xprime_step=float(xprime_step),
    )


def _linearized_grating_surface_output_map(
    item: _StageTrace,
    *,
    x_step_um: float = 1.0,
    xprime_step: float = 1.0e-3,
) -> NDArrayF:
    zero = np.zeros_like(np.asarray(item.cfg.omega, dtype=np.float64).reshape(-1), dtype=np.float64)
    base = _evaluate_grating_surface_state(item, x_in=zero, x_prime_in=zero)
    dx = _evaluate_grating_surface_state(
        item,
        x_in=np.full_like(zero, float(x_step_um), dtype=np.float64),
        x_prime_in=zero,
    )
    dp = _evaluate_grating_surface_state(
        item,
        x_in=zero,
        x_prime_in=np.full_like(zero, float(xprime_step), dtype=np.float64),
    )
    return _fit_affine_map_from_outputs(
        base_x=np.asarray(base["surface_coordinate_output_frame_um"], dtype=np.float64),
        base_xprime=np.asarray(base["surface_output_xprime"], dtype=np.float64),
        dx_x=np.asarray(dx["surface_coordinate_output_frame_um"], dtype=np.float64),
        dx_xprime=np.asarray(dx["surface_output_xprime"], dtype=np.float64),
        dp_x=np.asarray(dp["surface_coordinate_output_frame_um"], dtype=np.float64),
        dp_xprime=np.asarray(dp["surface_output_xprime"], dtype=np.float64),
        x_step_um=float(x_step_um),
        xprime_step=float(xprime_step),
    )


def _fit_affine_map_from_outputs(
    *,
    base_x: NDArrayF,
    base_xprime: NDArrayF,
    dx_x: NDArrayF,
    dx_xprime: NDArrayF,
    dp_x: NDArrayF,
    dp_xprime: NDArrayF,
    x_step_um: float,
    xprime_step: float,
) -> NDArrayF:
    matrices = np.zeros((base_x.size, 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = (dx_x - base_x) / float(x_step_um)
    matrices[:, 0, 1] = (dp_x - base_x) / float(xprime_step)
    matrices[:, 0, 2] = base_x
    matrices[:, 1, 0] = (dx_xprime - base_xprime) / float(x_step_um)
    matrices[:, 1, 1] = (dp_xprime - base_xprime) / float(xprime_step)
    matrices[:, 1, 2] = base_xprime
    matrices[:, 2, 2] = 1.0
    return matrices


def _boundary_point_from_series(series: TreacyGratingBoundarySeries) -> TreacyGratingBoundaryPoint:
    return TreacyGratingBoundaryPoint(
        instance_name=series.instance_name,
        stage_index=series.stage_index,
        is_return_pass=series.is_return_pass,
        configured_incidence_angle_deg=series.configured_incidence_angle_deg,
        input_plane_x_rms_um=_rms(series.input_plane_x_um),
        surface_coordinate_rms_um=_rms(series.surface_coordinate_input_frame_um),
        surface_coordinate_minus_input_x_rms_um=_rms(
            series.surface_coordinate_input_frame_um - series.input_plane_x_um
        ),
        output_plane_x_rms_um=_rms(series.output_plane_x_um),
        surface_coordinate_output_frame_rms_um=_rms(series.surface_coordinate_output_frame_um),
        surface_coordinate_minus_output_x_rms_um=_rms(
            series.surface_coordinate_output_frame_um - series.output_plane_x_um
        ),
        surface_intersection_distance_rms_um=_rms(series.surface_intersection_distance_um),
        max_abs_surface_hit_plane_residual_um=float(
            np.max(np.abs(series.surface_hit_plane_residual_um))
        ),
        rms_surface_hit_plane_residual_um=_rms(series.surface_hit_plane_residual_um),
        max_abs_tangential_phase_match_residual_per_um=float(
            np.max(np.abs(series.tangential_phase_match_residual_per_um))
        ),
        rms_tangential_phase_match_residual_per_um=_rms(
            series.tangential_phase_match_residual_per_um
        ),
        surface_phase_match_term_rms_rad=_rms(series.surface_phase_match_term_rad),
        groove_phase_term_rms_rad=_rms(series.groove_phase_term_rad),
        transport_like_phi3_rms_rad=_rms(series.transport_like_phi3_rad),
        runtime_phase_phi3_rms_rad=_rms(series.runtime_phase_phi3_rad),
        mean_incidence_angle_deg=float(np.degrees(np.mean(series.incidence_angle_rad))),
        mean_diffraction_angle_deg=float(np.degrees(np.mean(series.diffraction_angle_rad))),
        rms_incidence_angle_mismatch_deg=_rms(
            np.degrees(series.incidence_angle_rad) - float(series.configured_incidence_angle_deg)
        ),
        max_abs_incidence_angle_mismatch_deg=float(
            np.max(
                np.abs(
                    np.degrees(series.incidence_angle_rad)
                    - float(series.configured_incidence_angle_deg)
                )
            )
        ),
    )


def _actual_local_matrix_series_from_trace(item: _StageTrace) -> TreacyActualLocalMatrixSeries:
    boundary = _boundary_series_from_grating_trace(item)
    a_fixed = np.asarray(item.cfg.abcdef[:, 0, 0], dtype=np.float64)
    d_fixed = np.asarray(item.cfg.abcdef[:, 1, 1], dtype=np.float64)
    f_fixed = np.asarray(item.cfg.abcdef[:, 1, 2], dtype=np.float64)
    x_in = np.asarray(boundary.surface_coordinate_input_frame_um, dtype=np.float64)
    xprime_in = np.asarray(boundary.input_plane_xprime, dtype=np.float64)
    x_out = np.asarray(boundary.surface_coordinate_output_frame_um, dtype=np.float64)
    xprime_out = np.asarray(boundary.surface_output_xprime, dtype=np.float64)
    # The traced surface coordinates remain expressed in the configured local
    # grating frames, so an angle-rebuilt A = cos(theta_out)/cos(theta_in) is
    # expected to diagnose slope-side inconsistency more directly than x-scaling.
    a_actual = np.cos(np.asarray(boundary.diffraction_angle_rad, dtype=np.float64)) / np.cos(
        np.asarray(boundary.incidence_angle_rad, dtype=np.float64)
    )
    d_actual = np.asarray(boundary.surface_local_d, dtype=np.float64)
    f_actual = np.asarray(boundary.incoming_state_f_local, dtype=np.float64)
    return TreacyActualLocalMatrixSeries(
        instance_name=boundary.instance_name,
        stage_index=boundary.stage_index,
        is_return_pass=boundary.is_return_pass,
        omega_rad_per_fs=np.asarray(boundary.omega_rad_per_fs, dtype=np.float64),
        input_surface_x_um=x_in,
        input_surface_xprime=xprime_in,
        output_surface_x_um=x_out,
        output_surface_xprime=xprime_out,
        a_actual=a_actual,
        d_actual=d_actual,
        f_actual=f_actual,
        a_fixed=a_fixed,
        d_fixed=d_fixed,
        f_fixed=f_fixed,
        predicted_output_x_actual_um=a_actual * x_in,
        predicted_output_xprime_actual=(d_actual * xprime_in) + f_actual,
        predicted_output_x_fixed_um=a_fixed * x_in,
        predicted_output_xprime_fixed=(d_fixed * xprime_in) + f_fixed,
        configured_incidence_angle_deg=boundary.configured_incidence_angle_deg,
        mean_actual_incidence_angle_deg=float(np.degrees(np.mean(boundary.incidence_angle_rad))),
        rms_incidence_angle_mismatch_deg=_rms(
            np.degrees(boundary.incidence_angle_rad) - float(boundary.configured_incidence_angle_deg)
        ),
    )


def _actual_local_matrix_point_from_series(
    series: TreacyActualLocalMatrixSeries,
) -> TreacyActualLocalMatrixPoint:
    return TreacyActualLocalMatrixPoint(
        instance_name=series.instance_name,
        stage_index=series.stage_index,
        is_return_pass=series.is_return_pass,
        configured_incidence_angle_deg=series.configured_incidence_angle_deg,
        mean_actual_incidence_angle_deg=series.mean_actual_incidence_angle_deg,
        rms_incidence_angle_mismatch_deg=series.rms_incidence_angle_mismatch_deg,
        a_fixed_vs_actual_relative_rms=_normalized_rms_difference(series.a_fixed, series.a_actual),
        d_fixed_vs_actual_relative_rms=_normalized_rms_difference(series.d_fixed, series.d_actual),
        f_fixed_vs_actual_relative_rms=_normalized_rms_difference(series.f_fixed, series.f_actual),
        actual_local_x_residual_rms_um=_rms(
            series.output_surface_x_um - series.predicted_output_x_actual_um
        ),
        actual_local_xprime_residual_rms=_rms(
            series.output_surface_xprime - series.predicted_output_xprime_actual
        ),
        fixed_model_x_residual_rms_um=_rms(
            series.output_surface_x_um - series.predicted_output_x_fixed_um
        ),
        fixed_model_x_relative_residual_rms=_normalized_rms_difference(
            series.output_surface_x_um,
            series.predicted_output_x_fixed_um,
        ),
        fixed_model_xprime_residual_rms=_rms(
            series.output_surface_xprime - series.predicted_output_xprime_fixed
        ),
        fixed_model_xprime_relative_residual_rms=_normalized_rms_difference(
            series.output_surface_xprime,
            series.predicted_output_xprime_fixed,
        ),
    )


def _fitted_surface_map_from_trace(item: _StageTrace) -> NDArrayF:
    surface_input_map = _linearized_grating_surface_input_map(item)
    surface_output_map = _linearized_grating_surface_output_map(item)
    return surface_output_map @ _invert_batched_maps(surface_input_map)


def _fitted_surface_map_series_from_trace(item: _StageTrace) -> TreacyFittedSurfaceMapSeries:
    boundary = _boundary_series_from_grating_trace(item)
    fitted_map = _fitted_surface_map_from_trace(item)
    x_in = np.asarray(boundary.surface_coordinate_input_frame_um, dtype=np.float64)
    xprime_in = np.asarray(boundary.input_plane_xprime, dtype=np.float64)
    x_out = np.asarray(boundary.surface_coordinate_output_frame_um, dtype=np.float64)
    xprime_out = np.asarray(boundary.surface_output_xprime, dtype=np.float64)
    x_out_pred = (
        (np.asarray(fitted_map[:, 0, 0], dtype=np.float64) * x_in)
        + (np.asarray(fitted_map[:, 0, 1], dtype=np.float64) * xprime_in)
        + np.asarray(fitted_map[:, 0, 2], dtype=np.float64)
    )
    xprime_out_pred = (
        (np.asarray(fitted_map[:, 1, 0], dtype=np.float64) * x_in)
        + (np.asarray(fitted_map[:, 1, 1], dtype=np.float64) * xprime_in)
        + np.asarray(fitted_map[:, 1, 2], dtype=np.float64)
    )
    omega = np.asarray(boundary.omega_rad_per_fs, dtype=np.float64)
    generator = _type2_generating_phase_from_map(
        k_arr=np.full_like(omega, martinez_k_center(omega), dtype=np.float64),
        a=np.asarray(fitted_map[:, 0, 0], dtype=np.float64),
        c=np.asarray(fitted_map[:, 1, 0], dtype=np.float64),
        f=np.asarray(fitted_map[:, 1, 2], dtype=np.float64),
        x_in=x_in,
        xprime_in=xprime_in,
        x_out=x_out,
        xprime_out=xprime_out,
        image="sym",
    )
    return TreacyFittedSurfaceMapSeries(
        instance_name=boundary.instance_name,
        stage_index=boundary.stage_index,
        is_return_pass=boundary.is_return_pass,
        omega_rad_per_fs=omega,
        input_surface_x_um=x_in,
        input_surface_xprime=xprime_in,
        output_surface_x_um=x_out,
        output_surface_xprime=xprime_out,
        fitted_a=np.asarray(fitted_map[:, 0, 0], dtype=np.float64),
        fitted_b_um=np.asarray(fitted_map[:, 0, 1], dtype=np.float64),
        fitted_c_per_um=np.asarray(fitted_map[:, 1, 0], dtype=np.float64),
        fitted_d=np.asarray(fitted_map[:, 1, 1], dtype=np.float64),
        fitted_e_um=np.asarray(fitted_map[:, 0, 2], dtype=np.float64),
        fitted_f=np.asarray(fitted_map[:, 1, 2], dtype=np.float64),
        predicted_output_surface_x_um=np.asarray(x_out_pred, dtype=np.float64),
        predicted_output_surface_xprime=np.asarray(xprime_out_pred, dtype=np.float64),
        fitted_generator_center_k_rad=np.asarray(generator, dtype=np.float64),
        configured_incidence_angle_deg=boundary.configured_incidence_angle_deg,
    )


def _fitted_surface_map_point_from_series(
    series: TreacyFittedSurfaceMapSeries,
) -> TreacyFittedSurfaceMapPoint:
    return TreacyFittedSurfaceMapPoint(
        instance_name=series.instance_name,
        stage_index=series.stage_index,
        is_return_pass=series.is_return_pass,
        configured_incidence_angle_deg=series.configured_incidence_angle_deg,
        fitted_b_rms_um=_rms(series.fitted_b_um),
        fitted_e_rms_um=_rms(series.fitted_e_um),
        fitted_x_residual_rms_um=_rms(
            series.output_surface_x_um - series.predicted_output_surface_x_um
        ),
        fitted_x_relative_residual_rms=_normalized_rms_difference(
            series.output_surface_x_um,
            series.predicted_output_surface_x_um,
        ),
        fitted_xprime_residual_rms=_rms(
            series.output_surface_xprime - series.predicted_output_surface_xprime
        ),
        fitted_xprime_relative_residual_rms=_normalized_rms_difference(
            series.output_surface_xprime,
            series.predicted_output_surface_xprime,
        ),
        fitted_generator_center_k_rms_rad=_rms(series.fitted_generator_center_k_rad),
    )


def _adjacent_pair_subsystem_series(
    first: _StageTrace,
    second: _StageTrace,
    middle: NDArrayF,
) -> TreacyAdjacentPairSubsystemSeries:
    first_boundary = _boundary_series_from_grating_trace(first)
    second_boundary = _boundary_series_from_grating_trace(second)
    omega = np.asarray(first.cfg.omega, dtype=np.float64).reshape(-1)
    middle_arr = np.asarray(middle, dtype=np.float64)
    x1 = np.asarray(first_boundary.surface_coordinate_output_frame_um, dtype=np.float64)
    p1 = np.asarray(first_boundary.surface_output_xprime, dtype=np.float64)
    x2 = np.asarray(second_boundary.surface_coordinate_input_frame_um, dtype=np.float64)
    p2 = np.asarray(second_boundary.input_plane_xprime, dtype=np.float64)
    a = np.asarray(middle_arr[:, 0, 0], dtype=np.float64)
    b = np.asarray(middle_arr[:, 0, 1], dtype=np.float64)
    c = np.asarray(middle_arr[:, 1, 0], dtype=np.float64)
    d = np.asarray(middle_arr[:, 1, 1], dtype=np.float64)
    e = np.asarray(middle_arr[:, 0, 2], dtype=np.float64)
    f = np.asarray(middle_arr[:, 1, 2], dtype=np.float64)
    x2_pred = (a * x1) + (b * p1) + e
    p2_pred = (c * x1) + (d * p1) + f
    x_residual = x2 - x2_pred
    p2_residual = p2 - p2_pred
    first_surface_output_map = _linearized_grating_surface_output_map(first)
    second_surface_input_map = _linearized_grating_surface_input_map(second)
    first_stage_out_from_surface = np.asarray(first.cfg.abcdef, dtype=np.float64) @ _invert_batched_maps(
        first_surface_output_map
    )
    surface_anchored = second_surface_input_map @ middle_arr @ first_stage_out_from_surface
    x2_anchor = (
        (surface_anchored[:, 0, 0] * x1)
        + (surface_anchored[:, 0, 1] * p1)
        + surface_anchored[:, 0, 2]
    )
    p2_anchor = (
        (surface_anchored[:, 1, 0] * x1)
        + (surface_anchored[:, 1, 1] * p1)
        + surface_anchored[:, 1, 2]
    )
    x_anchor_residual = x2 - x2_anchor
    p2_anchor_residual = p2 - p2_anchor
    canonical_kernel = np.zeros_like(omega, dtype=np.float64)
    valid_b = np.abs(b) > 1e-15
    if np.any(valid_b):
        canonical_kernel[valid_b] = martinez_k(omega[valid_b]) * (
            (
                (a[valid_b] * (x1[valid_b] ** 2))
                - (2.0 * x1[valid_b] * x2[valid_b])
                + (d[valid_b] * (x2[valid_b] ** 2))
            )
            / (2.0 * b[valid_b])
        )
    return TreacyAdjacentPairSubsystemSeries(
        pair_name=f"{first_boundary.instance_name}->{second_boundary.instance_name}",
        first_instance_name=first_boundary.instance_name,
        second_instance_name=second_boundary.instance_name,
        omega_rad_per_fs=np.asarray(omega, dtype=np.float64),
        a=a,
        b_um=b,
        c_per_um=c,
        d=d,
        e_um=e,
        f=f,
        first_surface_output_x_um=x1,
        first_surface_output_xprime=p1,
        second_surface_input_x_um=x2,
        second_surface_input_xprime=p2,
        predicted_second_surface_x_um=x2_pred,
        predicted_second_surface_xprime=p2_pred,
        x_residual_um=x_residual,
        xprime_residual=p2_residual,
        surface_anchored_a=np.asarray(surface_anchored[:, 0, 0], dtype=np.float64),
        surface_anchored_b_um=np.asarray(surface_anchored[:, 0, 1], dtype=np.float64),
        surface_anchored_c_per_um=np.asarray(surface_anchored[:, 1, 0], dtype=np.float64),
        surface_anchored_d=np.asarray(surface_anchored[:, 1, 1], dtype=np.float64),
        surface_anchored_e_um=np.asarray(surface_anchored[:, 0, 2], dtype=np.float64),
        surface_anchored_f=np.asarray(surface_anchored[:, 1, 2], dtype=np.float64),
        anchored_predicted_second_surface_x_um=np.asarray(x2_anchor, dtype=np.float64),
        anchored_predicted_second_surface_xprime=np.asarray(p2_anchor, dtype=np.float64),
        anchored_x_residual_um=np.asarray(x_anchor_residual, dtype=np.float64),
        anchored_xprime_residual=np.asarray(p2_anchor_residual, dtype=np.float64),
        canonical_kernel_sample_k_rad=canonical_kernel,
    )


def _adjacent_pair_subsystem_point_from_series(
    series: TreacyAdjacentPairSubsystemSeries,
) -> TreacyAdjacentPairSubsystemPoint:
    return TreacyAdjacentPairSubsystemPoint(
        pair_name=series.pair_name,
        first_instance_name=series.first_instance_name,
        second_instance_name=series.second_instance_name,
        b_rms_um=_rms(series.b_um),
        x_residual_rms_um=_rms(series.x_residual_um),
        x_relative_residual_rms=_normalized_rms_difference(
            series.second_surface_input_x_um,
            series.predicted_second_surface_x_um,
        ),
        xprime_residual_rms=_rms(series.xprime_residual),
        xprime_relative_residual_rms=_normalized_rms_difference(
            series.second_surface_input_xprime,
            series.predicted_second_surface_xprime,
        ),
        anchored_x_residual_rms_um=_rms(series.anchored_x_residual_um),
        anchored_x_relative_residual_rms=_normalized_rms_difference(
            series.second_surface_input_x_um,
            series.anchored_predicted_second_surface_x_um,
        ),
        anchored_xprime_residual_rms=_rms(series.anchored_xprime_residual),
        anchored_xprime_relative_residual_rms=_normalized_rms_difference(
            series.second_surface_input_xprime,
            series.anchored_predicted_second_surface_xprime,
        ),
        canonical_kernel_sample_k_rms_rad=_rms(series.canonical_kernel_sample_k_rad),
    )


def _section_iv_boundary_point_from_series(
    series: SectionIVGratingBoundarySeries,
) -> SectionIVGratingBoundaryPoint:
    return SectionIVGratingBoundaryPoint(
        stage_name=series.stage_name,
        configured_incidence_angle_deg=series.configured_incidence_angle_deg,
        input_plane_x_rms_um=_rms(series.input_plane_x_um),
        output_plane_x_rms_um=_rms(series.output_plane_x_um),
        paired_output_x_rms_um=_rms(series.paired_output_x_um),
        output_x_pair_mismatch_rms_um=_rms(series.output_plane_x_um - series.paired_output_x_um),
        f_exact_rms=_rms(series.f_exact),
        f_paired_rms=_rms(series.f_paired),
        f_pair_mismatch_rms=_rms(series.f_exact - series.f_paired),
        output_x_pair_relative_rms=_normalized_rms_difference(
            series.output_plane_x_um,
            series.paired_output_x_um,
        ),
        f_pair_relative_rms=_normalized_rms_difference(series.f_exact, series.f_paired),
        local_endpoint_exact_sample_k_rms_rad=_rms(series.local_endpoint_exact_sample_k_rad),
        local_endpoint_paired_sample_k_rms_rad=_rms(series.local_endpoint_paired_sample_k_rad),
        global_endpoint_exact_sample_k_rms_rad=_rms(series.global_endpoint_exact_sample_k_rad),
        global_endpoint_paired_sample_k_rms_rad=_rms(series.global_endpoint_paired_sample_k_rad),
        phi3_exact_sample_k_rms_rad=_rms(series.phi3_exact_sample_k_rad),
        phi3_paired_sample_k_rms_rad=_rms(series.phi3_paired_sample_k_rad),
        phi3_oracle_sample_k_rms_rad=_rms(series.phi3_oracle_sample_k_rad),
        phi3_pair_mismatch_rms_rad=_rms(
            series.phi3_exact_sample_k_rad - series.phi3_paired_sample_k_rad
        ),
        phi3_oracle_vs_paired_mismatch_rms_rad=_rms(
            series.phi3_oracle_sample_k_rad - series.phi3_paired_sample_k_rad
        ),
        phi3_oracle_vs_exact_plus_global_endpoint_rms_rad=_rms(
            series.phi3_oracle_sample_k_rad
            - (series.phi3_exact_sample_k_rad + series.global_endpoint_exact_sample_k_rad)
        ),
    )


def _treacy_nonlocal_phase_terms(result: object) -> dict[str, NDArrayF]:
    from abcdef_sim.data_models.results import AbcdefRunResult

    if not isinstance(result, AbcdefRunResult):
        raise TypeError(f"Expected AbcdefRunResult, got {type(result).__name__}")

    phi_geom = np.asarray(result.pipeline_result.phi_geom_total_rad, dtype=np.float64)
    phi1 = _zero_or_array(result.pipeline_result.phi1_rad, template=phi_geom)
    phi4 = _zero_or_array(result.pipeline_result.phi4_rad, template=phi_geom)
    filter_phase = np.sum(
        [
            _zero_or_array(contribution.filter_phase_rad, template=phi_geom)
            for contribution in result.pipeline_result.contributions
        ],
        axis=0,
    )
    return {
        "phi_geom": phi_geom,
        "phi1": phi1,
        "phi4": phi4,
        "filter_phase": filter_phase,
    }


def _treacy_global_endpoint_phase(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    rays_in = np.asarray(trace[0].state_in.rays, dtype=np.float64)
    rays_out = np.asarray(trace[-1].state_out.rays, dtype=np.float64)
    return 0.5 * k_arr * (
        (rays_in[:, 0, 0] * rays_in[:, 1, 0]) - (rays_out[:, 0, 0] * rays_out[:, 1, 0])
    )


def _treacy_exact_local_phi3_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    total = np.zeros_like(omega, dtype=np.float64)
    for item in trace:
        if not isinstance(item.optic_spec, GratingCfg):
            continue
        f_exact = np.asarray(item.cfg.abcdef[:, 1, 2], dtype=np.float64)
        x_out = np.asarray(item.state_out.rays[:, 0, 0], dtype=np.float64)
        total = total + (0.5 * k_arr * f_exact * x_out)
    return total


def _treacy_surface_hit_phi3_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    total = np.zeros_like(omega, dtype=np.float64)
    for item in trace:
        if not isinstance(item.optic_spec, GratingCfg):
            continue
        boundary_series = _boundary_series_from_grating_trace(item)
        f_exact = np.asarray(item.cfg.abcdef[:, 1, 2], dtype=np.float64)
        x_surface_out = np.asarray(
            boundary_series.surface_coordinate_output_frame_um,
            dtype=np.float64,
        )
        total = total + (0.5 * k_arr * f_exact * x_surface_out)
    return total


def _canonical_kernel_from_subsystem_map(
    *,
    k_arr: NDArrayF,
    a: NDArrayF,
    b_um: NDArrayF,
    d: NDArrayF,
    x_in: NDArrayF,
    x_out: NDArrayF,
) -> NDArrayF:
    kernel = np.zeros_like(k_arr, dtype=np.float64)
    valid_b = np.abs(b_um) > 1e-15
    if np.any(valid_b):
        kernel[valid_b] = k_arr[valid_b] * (
            (
                (a[valid_b] * (x_in[valid_b] ** 2))
                - (2.0 * x_in[valid_b] * x_out[valid_b])
                + (d[valid_b] * (x_out[valid_b] ** 2))
            )
            / (2.0 * b_um[valid_b])
        )
    return kernel


def _treacy_fitted_surface_generator_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    total = np.zeros_like(omega, dtype=np.float64)
    for item in trace:
        if not isinstance(item.optic_spec, GratingCfg):
            continue
        boundary = _boundary_series_from_grating_trace(item)
        fitted_map = _fitted_surface_map_from_trace(item)
        total = total + _type2_generating_phase_from_map(
            k_arr=k_arr,
            a=np.asarray(fitted_map[:, 0, 0], dtype=np.float64),
            c=np.asarray(fitted_map[:, 1, 0], dtype=np.float64),
            f=np.asarray(fitted_map[:, 1, 2], dtype=np.float64),
            x_in=np.asarray(boundary.surface_coordinate_input_frame_um, dtype=np.float64),
            xprime_in=np.asarray(boundary.input_plane_xprime, dtype=np.float64),
            x_out=np.asarray(boundary.surface_coordinate_output_frame_um, dtype=np.float64),
            xprime_out=np.asarray(boundary.surface_output_xprime, dtype=np.float64),
            image="sym",
        )
    return total


def _treacy_surface_anchored_pair_kernel_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    total = np.zeros_like(omega, dtype=np.float64)
    for first, second, middle in _treacy_adjacent_grating_pairs(trace):
        series = _adjacent_pair_subsystem_series(first, second, middle)
        total = total + _canonical_kernel_from_subsystem_map(
            k_arr=k_arr,
            a=np.asarray(series.surface_anchored_a, dtype=np.float64),
            b_um=np.asarray(series.surface_anchored_b_um, dtype=np.float64),
            d=np.asarray(series.surface_anchored_d, dtype=np.float64),
            x_in=np.asarray(series.first_surface_output_x_um, dtype=np.float64),
            x_out=np.asarray(series.second_surface_input_x_um, dtype=np.float64),
        )
    return total


def _treacy_surface_canonical_phi_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
    image: Literal["sym", "raw"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    total = np.zeros_like(omega, dtype=np.float64)
    for item in trace:
        if not isinstance(item.optic_spec, GratingCfg):
            continue
        boundary_series = _boundary_series_from_grating_trace(item)
        total = total + _type2_generating_phase_from_map(
            k_arr=k_arr,
            a=1.0 / np.asarray(boundary_series.surface_local_d, dtype=np.float64),
            c=np.zeros_like(omega, dtype=np.float64),
            f=np.asarray(boundary_series.incoming_state_f_local, dtype=np.float64),
            x_in=np.asarray(boundary_series.surface_coordinate_input_frame_um, dtype=np.float64),
            xprime_in=np.asarray(boundary_series.input_plane_xprime, dtype=np.float64),
            x_out=np.asarray(boundary_series.surface_coordinate_output_frame_um, dtype=np.float64),
            xprime_out=np.asarray(boundary_series.surface_output_xprime, dtype=np.float64),
            image=image,
        )
    return total


def _treacy_surface_anchored_phi3_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
    f_mode: Literal["runtime_phase", "transport_exact"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    total = np.zeros_like(omega, dtype=np.float64)
    grating_items = [item for item in trace if isinstance(item.optic_spec, GratingCfg)]
    if not grating_items:
        return total

    pair_maps = _treacy_surface_anchored_pair_maps(trace)
    current_surface_out: NDArrayF | None = None
    previous_name: str | None = None

    for item in grating_items:
        surface_input_map = _linearized_grating_surface_input_map(item)
        surface_output_map = _linearized_grating_surface_output_map(item)
        surface_output_from_input = surface_output_map @ _invert_batched_maps(surface_input_map)
        if current_surface_out is None:
            boundary = _boundary_series_from_grating_trace(item)
            current_surface_in = np.zeros((omega.size, 3, 1), dtype=np.float64)
            current_surface_in[:, 0, 0] = np.asarray(
                boundary.surface_coordinate_input_frame_um,
                dtype=np.float64,
            )
            current_surface_in[:, 1, 0] = np.asarray(boundary.input_plane_xprime, dtype=np.float64)
            current_surface_in[:, 2, 0] = 1.0
        else:
            if previous_name is None:
                raise ValueError("previous_name must be populated when current_surface_out is set")
            current_surface_in = pair_maps[(previous_name, str(item.cfg.instance_name))] @ current_surface_out

        current_surface_out = surface_output_from_input @ current_surface_in
        if f_mode == "runtime_phase":
            f_phase = _phase_f_array(item.cfg)
        else:
            f_phase = np.asarray(item.cfg.abcdef[:, 1, 2], dtype=np.float64)
        total = total + (0.5 * k_arr * f_phase * np.asarray(current_surface_out[:, 0, 0], dtype=np.float64))
        previous_name = str(item.cfg.instance_name)
    return total


def _treacy_actual_local_f_phi3_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
    x_mode: Literal["stage", "surface"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    total = np.zeros_like(omega, dtype=np.float64)
    for item in trace:
        if not isinstance(item.optic_spec, GratingCfg):
            continue
        boundary = _boundary_series_from_grating_trace(item)
        if x_mode == "stage":
            x_use = np.asarray(item.state_out.rays[:, 0, 0], dtype=np.float64)
        else:
            x_use = np.asarray(boundary.surface_coordinate_output_frame_um, dtype=np.float64)
        total = total + (
            0.5 * k_arr * np.asarray(boundary.incoming_state_f_local, dtype=np.float64) * x_use
        )
    return total


def _treacy_configured_surface_slope_phi3_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
    x_mode: Literal["stage", "surface_anchored"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    total = np.zeros_like(omega, dtype=np.float64)
    if x_mode == "stage":
        for item in trace:
            if not isinstance(item.optic_spec, GratingCfg):
                continue
            boundary = _boundary_series_from_grating_trace(item)
            d_fixed = np.asarray(item.cfg.abcdef[:, 1, 1], dtype=np.float64)
            f_phase = np.asarray(boundary.surface_output_xprime, dtype=np.float64) - (
                d_fixed * np.asarray(boundary.input_plane_xprime, dtype=np.float64)
            )
            x_use = np.asarray(item.state_out.rays[:, 0, 0], dtype=np.float64)
            total = total + (0.5 * k_arr * f_phase * x_use)
        return total

    pair_maps = _treacy_surface_anchored_pair_maps(trace)
    grating_items = tuple(item for item in trace if isinstance(item.optic_spec, GratingCfg))
    current_surface_out: NDArrayF | None = None
    previous_name: str | None = None
    for item in grating_items:
        surface_input_map = _linearized_grating_surface_input_map(item)
        surface_output_map = _linearized_grating_surface_output_map(item)
        surface_output_from_input = surface_output_map @ _invert_batched_maps(surface_input_map)
        boundary = _boundary_series_from_grating_trace(item)
        if current_surface_out is None:
            current_surface_in = np.zeros((omega.size, 3, 1), dtype=np.float64)
            current_surface_in[:, 0, 0] = np.asarray(
                boundary.surface_coordinate_input_frame_um,
                dtype=np.float64,
            )
            current_surface_in[:, 1, 0] = np.asarray(boundary.input_plane_xprime, dtype=np.float64)
            current_surface_in[:, 2, 0] = 1.0
        else:
            if previous_name is None:
                raise ValueError("previous_name must be populated when current_surface_out is set")
            current_surface_in = pair_maps[(previous_name, str(item.cfg.instance_name))] @ current_surface_out
        current_surface_out = surface_output_from_input @ current_surface_in
        d_fixed = np.asarray(item.cfg.abcdef[:, 1, 1], dtype=np.float64)
        f_phase = np.asarray(boundary.surface_output_xprime, dtype=np.float64) - (
            d_fixed * np.asarray(boundary.input_plane_xprime, dtype=np.float64)
        )
        total = total + (0.5 * k_arr * f_phase * np.asarray(current_surface_out[:, 0, 0], dtype=np.float64))
        previous_name = str(item.cfg.instance_name)
    return total


def _treacy_incoming_state_phi3_total(
    trace: tuple[_StageTrace, ...],
    *,
    k_mode: Literal["center", "sample"],
    target_instances: tuple[str, ...],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array(omega, k_mode=k_mode)
    targets = set(target_instances)
    total = np.zeros_like(omega, dtype=np.float64)
    for item in trace:
        if not isinstance(item.optic_spec, GratingCfg):
            continue
        if item.cfg.instance_name in targets:
            boundary_series = _boundary_series_from_grating_trace(item)
            f_phase = np.asarray(boundary_series.incoming_state_f_local, dtype=np.float64)
            x_surface_out = np.asarray(
                boundary_series.surface_coordinate_output_frame_um,
                dtype=np.float64,
            )
            total = total + (0.5 * k_arr * f_phase * x_surface_out)
            continue
        f_exact = np.asarray(item.cfg.abcdef[:, 1, 2], dtype=np.float64)
        x_out = np.asarray(item.state_out.rays[:, 0, 0], dtype=np.float64)
        total = total + (0.5 * k_arr * f_exact * x_out)
    return total


def _treacy_conjugate_pair_phase_total(
    trace: tuple[_StageTrace, ...],
    *,
    mode: Literal["f2_exact_sq", "mf1_sq"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = np.full_like(omega, martinez_k_center(omega), dtype=np.float64)
    total = np.zeros_like(omega, dtype=np.float64)
    for first, second, middle in _treacy_grating_pairs(trace):
        b_pair = np.asarray(middle[:, 0, 1], dtype=np.float64)
        f1 = np.asarray(first.cfg.abcdef[:, 1, 2], dtype=np.float64)
        f2 = np.asarray(second.cfg.abcdef[:, 1, 2], dtype=np.float64)
        m_pair = float(np.mean(np.asarray(first.cfg.abcdef[:, 0, 0], dtype=np.float64)))
        if mode == "f2_exact_sq":
            f_use = f2
        else:
            f_use = float(m_pair) * f1
        total = total + (0.5 * k_arr * b_pair * (f_use**2))
    return total


def _treacy_grating_pairs(
    trace: tuple[_StageTrace, ...],
) -> tuple[tuple[_StageTrace, _StageTrace, NDArrayF], ...]:
    grating_indices = [index for index, item in enumerate(trace) if isinstance(item.optic_spec, GratingCfg)]
    if len(grating_indices) % 2 != 0:
        raise ValueError("expected an even number of Treacy grating hits")

    pairs: list[tuple[_StageTrace, _StageTrace, NDArrayF]] = []
    identity = np.eye(3, dtype=np.float64)
    for pair_start in range(0, len(grating_indices), 2):
        i0 = grating_indices[pair_start]
        i1 = grating_indices[pair_start + 1]
        middle = np.repeat(identity[None, ...], trace[i0].cfg.omega.size, axis=0)
        for item in trace[i0 + 1 : i1]:
            middle = np.asarray(item.cfg.abcdef, dtype=np.float64) @ middle
        pairs.append((trace[i0], trace[i1], middle))
    return tuple(pairs)


def _treacy_adjacent_grating_pairs(
    trace: tuple[_StageTrace, ...],
) -> tuple[tuple[_StageTrace, _StageTrace, NDArrayF], ...]:
    grating_indices = [index for index, item in enumerate(trace) if isinstance(item.optic_spec, GratingCfg)]
    if len(grating_indices) < 2:
        raise ValueError("expected at least two Treacy grating hits")

    pairs: list[tuple[_StageTrace, _StageTrace, NDArrayF]] = []
    identity = np.eye(3, dtype=np.float64)
    for i0, i1 in zip(grating_indices, grating_indices[1:], strict=False):
        middle = np.repeat(identity[None, ...], trace[i0].cfg.omega.size, axis=0)
        for item in trace[i0 + 1 : i1]:
            middle = np.asarray(item.cfg.abcdef, dtype=np.float64) @ middle
        pairs.append((trace[i0], trace[i1], middle))
    return tuple(pairs)


def _treacy_surface_anchored_pair_maps(
    trace: tuple[_StageTrace, ...],
) -> dict[tuple[str, str], NDArrayF]:
    pair_maps: dict[tuple[str, str], NDArrayF] = {}
    for first, second, middle in _treacy_adjacent_grating_pairs(trace):
        first_surface_output_map = _linearized_grating_surface_output_map(first)
        second_surface_input_map = _linearized_grating_surface_input_map(second)
        first_stage_out_from_surface = np.asarray(first.cfg.abcdef, dtype=np.float64) @ _invert_batched_maps(
            first_surface_output_map
        )
        pair_maps[(str(first.cfg.instance_name), str(second.cfg.instance_name))] = (
            second_surface_input_map @ np.asarray(middle, dtype=np.float64) @ first_stage_out_from_surface
        )
    return pair_maps


def _parallel_plane_normal_offset(a: OpticStageCfg, b: OpticStageCfg) -> float:
    if (
        a.entrance_surface_point_x_um is None
        or a.entrance_surface_point_z_um is None
        or a.entrance_surface_normal_angle_rad is None
        or b.entrance_surface_point_x_um is None
        or b.entrance_surface_point_z_um is None
        or b.entrance_surface_normal_angle_rad is None
    ):
        raise ValueError("plane comparison requires complete entrance surface metadata")
    normal_x = math.sin(float(a.entrance_surface_normal_angle_rad))
    normal_z = math.cos(float(a.entrance_surface_normal_angle_rad))
    delta_x = float(b.entrance_surface_point_x_um) - float(a.entrance_surface_point_x_um)
    delta_z = float(b.entrance_surface_point_z_um) - float(a.entrance_surface_point_z_um)
    return (normal_x * delta_x) + (normal_z * delta_z)


def _parallel_plane_angle_diff_mod_pi(a: OpticStageCfg, b: OpticStageCfg) -> float:
    if a.entrance_surface_normal_angle_rad is None or b.entrance_surface_normal_angle_rad is None:
        raise ValueError("plane comparison requires complete entrance surface metadata")
    return min(
        _wrapped_angle_abs(
            float(b.entrance_surface_normal_angle_rad) - float(a.entrance_surface_normal_angle_rad)
        ),
        _wrapped_angle_abs(
            float(b.entrance_surface_normal_angle_rad) - float(a.entrance_surface_normal_angle_rad) - math.pi
        ),
    )


def _wrapped_angle_abs(angle_rad: float) -> float:
    return abs(math.atan2(math.sin(float(angle_rad)), math.cos(float(angle_rad))))


def _invert_batched_maps(matrices: NDArrayF) -> NDArrayF:
    matrix_arr = np.asarray(matrices, dtype=np.float64)
    if matrix_arr.ndim != 3 or matrix_arr.shape[1:] != (3, 3):
        raise ValueError(f"matrices must have shape (N,3,3); got {matrix_arr.shape}")
    return np.linalg.inv(matrix_arr)


def _type2_generating_phase_from_map(
    *,
    k_arr: NDArrayF,
    a: NDArrayF,
    c: NDArrayF,
    f: NDArrayF,
    x_in: NDArrayF,
    xprime_in: NDArrayF,
    x_out: NDArrayF,
    xprime_out: NDArrayF,
    image: Literal["sym", "raw"],
) -> NDArrayF:
    # For a thin B=0 affine map, this is the exact type-2 generator image.
    generator = (a * x_in * xprime_out) - (0.5 * a * c * (x_in**2)) - (a * f * x_in)
    if image == "sym":
        return k_arr * (0.5 * ((x_in * xprime_in) + (x_out * xprime_out)) - generator)
    return k_arr * (generator - (x_out * xprime_out))


def _k_array(omega: NDArrayF, *, k_mode: Literal["center", "sample"]) -> NDArrayF:
    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    if k_mode == "sample":
        return martinez_k(omega_arr)
    return np.full_like(omega_arr, martinez_k_center(omega_arr), dtype=np.float64)


def _rms(values: NDArrayF) -> float:
    array = np.asarray(values, dtype=np.float64)
    return float(np.sqrt(np.mean(array**2)))


def _normalized_rms_difference(values: NDArrayF, reference: NDArrayF) -> float:
    value_arr = np.asarray(values, dtype=np.float64)
    reference_arr = np.asarray(reference, dtype=np.float64)
    reference_rms = _rms(reference_arr)
    diff_rms = _rms(value_arr - reference_arr)
    if reference_rms <= 1e-15:
        return 0.0 if diff_rms <= 1e-15 else float("inf")
    return diff_rms / reference_rms


def _zero_or_array(value: NDArrayF | None, *, template: NDArrayF) -> NDArrayF:
    if value is None:
        return np.zeros_like(np.asarray(template, dtype=np.float64), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _section_iv_omega_grid(
    *,
    center_wavelength_nm: float,
    n_samples: int,
    pulse_width_fs: float,
    span_scale: float,
) -> NDArrayF:
    base_state = build_standalone_laser_state(
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=float(pulse_width_fs) / float(span_scale),
                center_wavelength_nm=float(center_wavelength_nm),
                n_samples=int(n_samples),
                time_window_fs=3000.0,
            ),
            beam=BeamSpec(radius_mm=1.0, m2=1.0),
        )
    )
    laser_spec = _laser_spec_from_state(base_state)
    return np.asarray(laser_spec.omega(), dtype=np.float64)


def _diffraction_angle_from_geometry(
    *,
    omega0: float,
    period_um: float,
    incidence_angle_rad: float,
    diffraction_order: float,
) -> float:
    wavelength_um = (2.0 * math.pi * _C_UM_PER_FS) / float(omega0)
    diff_arg = -float(diffraction_order) * (wavelength_um / float(period_um)) - math.sin(
        float(incidence_angle_rad)
    )
    if diff_arg < -1.0 or diff_arg > 1.0:
        raise ValueError("diffraction geometry is outside the asin domain")
    return math.asin(diff_arg)


def _relative_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return abs(float(value))
    return abs(float(value) - float(reference)) / abs(float(reference))
