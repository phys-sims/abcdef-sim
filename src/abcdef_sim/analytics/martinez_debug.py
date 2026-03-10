from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal, cast

import numpy as np

from abcdef_sim import (
    BeamSpec,
    PulseSpec,
    StandaloneLaserSpec,
    run_abcdef,
    treacy_compressor_preset,
)
from abcdef_sim.cache.backend import NullCacheBackend
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.results import AbcdefRunResult, PhaseContribution
from abcdef_sim.data_models.specs import (
    AbcdefCfg,
    FrameTransformCfg,
    FreeSpaceCfg,
    GratingCfg,
    ThickLensCfg,
)
from abcdef_sim.data_models.states import RayState
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.physics.abcdef.adapters import apply_cfg
from abcdef_sim.physics.abcdef.conventions import (
    extract_A,
    extract_B,
    extract_C,
    extract_D,
    extract_E,
    extract_F,
)
from abcdef_sim.physics.abcdef.dispersion import fit_phase_taylor_affine_detrended
from abcdef_sim.physics.abcdef.phase_terms import (
    combine_phi_total_rad,
    martinez_k,
    martinez_k_center,
)
from abcdef_sim.physics.abcdef.pulse import build_standalone_laser_state
from abcdef_sim.physics.abcdef.treacy import (
    compute_grating_diffraction_angle_deg,
    compute_treacy_analytic_metrics,
    phase_from_treacy_dispersion,
)
from abcdef_sim.pipeline._assembler import SystemAssembler
from abcdef_sim.runner import _initial_ray_state, _laser_spec_from_state, _resolve_runtime_cfg

NDArrayF = np.ndarray
SectionIVReferenceMode = Literal["legacy_linear", "quadratic_bfg2"]

DEFAULT_SPAN_SCALES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)

__all__ = [
    "DEFAULT_SPAN_SCALES",
    "MartinezSectionIVCoefficientPoint",
    "MartinezSectionIVPhasePoint",
    "MartinezSectionIVPhi3VariantPoint",
    "TreacyPartitionPoint",
    "TreacyPhi3PerGratingPoint",
    "TreacyGratingFramePoint",
    "TreacyPhi3SignStagePoint",
    "TreacyPhi3SignVariantPoint",
    "TreacyPhi3VariantPoint",
    "run_martinez_section_iv_coefficient_audit",
    "run_martinez_section_iv_phase_study",
    "run_martinez_section_iv_phi3_variant_study",
    "run_treacy_partition_span_study",
    "run_treacy_grating_frame_audit",
    "run_treacy_phi3_per_grating_budget",
    "run_treacy_phi3_sign_audit",
    "run_treacy_phi3_variant_comparison",
]


@dataclass(frozen=True, slots=True)
class MartinezSectionIVCoefficientPoint:
    delta_omega_rad_per_fs: float
    system_a: float
    expected_a: float
    system_b_um: float
    expected_b_um: float
    system_c_per_um: float
    expected_c_per_um: float
    system_d: float
    expected_d: float
    system_e_um: float
    expected_e_um: float
    system_f: float
    expected_f: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class MartinezSectionIVPhasePoint:
    span_scale: float
    variant: str
    gdd_fs2: float
    tod_fs3: float
    reference_gdd_fs2: float
    reference_tod_fs3: float
    gdd_abs_error_fs2: float
    tod_abs_error_fs3: float
    weighted_rms_rad: float

    def to_dict(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["variant"] = str(self.variant)
        return payload


@dataclass(frozen=True, slots=True)
class MartinezSectionIVPhi3VariantPoint:
    span_scale: float
    variant: str
    total_gdd_fs2: float
    total_tod_fs3: float
    reference_gdd_fs2: float
    reference_tod_fs3: float
    gdd_abs_error_fs2: float
    tod_abs_error_fs3: float
    weighted_rms_rad: float

    def to_dict(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["variant"] = str(self.variant)
        return payload


@dataclass(frozen=True, slots=True)
class TreacyPartitionPoint:
    span_scale: float
    pulse_width_fs: float
    variant: str
    raw_abcdef_gdd_fs2: float
    raw_abcdef_tod_fs3: float
    analytic_gdd_fs2: float
    analytic_tod_fs3: float
    raw_abcdef_gdd_rel_error: float
    raw_abcdef_tod_rel_error: float
    weighted_rms_rad: float

    def to_dict(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["variant"] = str(self.variant)
        return payload


@dataclass(frozen=True, slots=True)
class TreacyPhi3VariantPoint:
    beam_radius_mm: float
    variant: str
    raw_abcdef_gdd_fs2: float
    raw_abcdef_tod_fs3: float
    analytic_gdd_fs2: float
    analytic_tod_fs3: float
    raw_abcdef_gdd_rel_error: float
    raw_abcdef_tod_rel_error: float
    weighted_rms_rad: float
    weighted_xprime_rms: float

    def to_dict(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["variant"] = str(self.variant)
        return payload


@dataclass(frozen=True, slots=True)
class TreacyPhi3PerGratingPoint:
    variant: str
    beam_radius_mm: float
    instance_name: str
    stage_index: int
    local_axis_angle_deg: float
    is_return_pass: bool
    f_phase_linear_coeff: float
    f_phase_rms: float
    x_after_rms_um: float
    phi3_gdd_fs2: float
    phi3_tod_fs3: float
    required_residual_projection: float
    required_residual_gdd_fs2: float
    required_residual_tod_fs3: float

    def to_dict(self) -> dict[str, float | str | bool | int]:
        payload = asdict(self)
        payload["variant"] = str(self.variant)
        payload["instance_name"] = str(self.instance_name)
        return payload


@dataclass(frozen=True, slots=True)
class TreacyGratingFramePoint:
    instance_name: str
    stage_index: int
    following_frame_instance_name: str
    frame_stage_index: int
    max_x_delta_um: float
    max_xprime_flip_residual: float
    x_rms_um: float
    xprime_rms: float

    def to_dict(self) -> dict[str, float | str | int]:
        payload = asdict(self)
        payload["instance_name"] = str(self.instance_name)
        payload["following_frame_instance_name"] = str(self.following_frame_instance_name)
        return payload


@dataclass(frozen=True, slots=True)
class TreacyPhi3SignStagePoint:
    instance_name: str
    stage_index: int
    local_axis_angle_deg: float
    positive_x_axis_angle_deg: float
    positive_z_axis_angle_deg: float
    is_return_pass: bool
    return_flip_hypothesis: bool
    f_phase_linear_coeff: float

    def to_dict(self) -> dict[str, float | str | bool | int]:
        payload = asdict(self)
        payload["instance_name"] = str(self.instance_name)
        return payload


@dataclass(frozen=True, slots=True)
class TreacyPhi3SignVariantPoint:
    variant: str
    sign_case: str
    beam_radius_mm: float
    raw_abcdef_gdd_rel_error: float
    raw_abcdef_tod_rel_error: float

    def to_dict(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["variant"] = str(self.variant)
        payload["sign_case"] = str(self.sign_case)
        return payload


@dataclass(frozen=True, slots=True)
class _StageTrace:
    index: int
    optic_spec: FrameTransformCfg | FreeSpaceCfg | GratingCfg | ThickLensCfg
    cfg: OpticStageCfg
    state_in: RayState
    state_out: RayState
    contribution: PhaseContribution


@dataclass(frozen=True, slots=True)
class _SectionIVSystem:
    omega0_rad_per_fs: float
    system: NDArrayF
    expected_b_um: float
    expected_c_per_um: float
    expected_e_um: NDArrayF
    phi3_sample_k_series2_rad: NDArrayF
    phi3_center_k_series2_rad: NDArrayF
    phi3_sample_k_series1_rad: NDArrayF
    phi3_center_k_series1_rad: NDArrayF
    reference_phase_rad: NDArrayF
    reference_phase_quadratic_rad: NDArrayF


@dataclass(frozen=True, slots=True)
class _Phi3VariantConfig:
    name: str
    f_mode: Literal["transport_exact", "linear_f0", "series2"]
    x_mode: Literal["runtime", "phase_ad"]
    k_mode: Literal["center", "sample"]
    boundary_mode: Literal["none", "terminal_bpre_residual"] = "none"
    boundary_weight: float = 0.0


@dataclass(frozen=True, slots=True)
class _GratingPhaseSeries:
    f_exact: NDArrayF
    f_linear: NDArrayF
    f_series2: NDArrayF
    a_series1: NDArrayF
    d_series1: NDArrayF


PHI3_VARIANTS: tuple[_Phi3VariantConfig, ...] = (
    _Phi3VariantConfig(
        name="runtime_exact_centerk",
        f_mode="transport_exact",
        x_mode="runtime",
        k_mode="center",
    ),
    _Phi3VariantConfig(
        name="martinez_f0_centerk",
        f_mode="linear_f0",
        x_mode="runtime",
        k_mode="center",
    ),
    _Phi3VariantConfig(
        name="martinez_series2_centerk",
        f_mode="series2",
        x_mode="runtime",
        k_mode="center",
    ),
    _Phi3VariantConfig(
        name="martinez_series2_phasead_centerk",
        f_mode="series2",
        x_mode="phase_ad",
        k_mode="center",
    ),
    _Phi3VariantConfig(
        name="martinez_series2_phasead_samplek",
        f_mode="series2",
        x_mode="phase_ad",
        k_mode="sample",
    ),
)

# Probe-only variant for matched Treacy studies. This keeps the current runtime
# phi3 term intact, then adds a small upstream-B boundary residual on terminal
# compressor gratings. It is intentionally not part of the Section IV study or
# the runtime defaults because the coefficient is not yet a fully derived
# grating-boundary law.
TREACY_PHI3_VARIANTS: tuple[_Phi3VariantConfig, ...] = PHI3_VARIANTS + (
    _Phi3VariantConfig(
        name="martinez_series2_terminal_bpre_probe",
        f_mode="series2",
        x_mode="runtime",
        k_mode="center",
        boundary_mode="terminal_bpre_residual",
        boundary_weight=0.019,
    ),
)


def run_martinez_section_iv_coefficient_audit(
    *,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    center_wavelength_nm: float = 1030.0,
    focal_length_um: float = 300_000.0,
    a: float = 0.05,
    a_prime: float = 0.0,
    b: float = 0.0,
    n_samples: int = 256,
    span_scale: float = 1.0,
    pulse_width_fs: float = 100.0,
) -> tuple[MartinezSectionIVCoefficientPoint, ...]:
    omega = _scaled_omega_grid(
        center_wavelength_nm=center_wavelength_nm,
        n_samples=n_samples,
        pulse_width_fs=pulse_width_fs,
        span_scale=span_scale,
    )
    delta_omega = omega - float(np.mean(omega))
    section = _section_iv_system(
        omega=omega,
        delta_omega=delta_omega,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        center_wavelength_nm=center_wavelength_nm,
        focal_length_um=focal_length_um,
        a=a,
        a_prime=a_prime,
        b=b,
    )
    return tuple(
        MartinezSectionIVCoefficientPoint(
            delta_omega_rad_per_fs=float(delta_omega_i),
            system_a=float(extract_A(section.system)[idx]),
            expected_a=-1.0,
            system_b_um=float(extract_B(section.system)[idx]),
            expected_b_um=float(section.expected_b_um),
            system_c_per_um=float(extract_C(section.system)[idx]),
            expected_c_per_um=float(section.expected_c_per_um),
            system_d=float(extract_D(section.system)[idx]),
            expected_d=-1.0,
            system_e_um=float(extract_E(section.system)[idx]),
            expected_e_um=float(section.expected_e_um[idx]),
            system_f=float(extract_F(section.system)[idx]),
            expected_f=0.0,
        )
        for idx, delta_omega_i in enumerate(delta_omega)
    )


def run_martinez_section_iv_phase_study(
    *,
    span_scales: tuple[float, ...] = DEFAULT_SPAN_SCALES,
    reference_mode: SectionIVReferenceMode = "legacy_linear",
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    center_wavelength_nm: float = 1030.0,
    focal_length_um: float = 300_000.0,
    a: float = 0.05,
    a_prime: float = 0.0,
    b: float = 0.0,
    n_samples: int = 256,
    pulse_width_fs: float = 100.0,
) -> tuple[MartinezSectionIVPhasePoint, ...]:
    points: list[MartinezSectionIVPhasePoint] = []
    for span_scale in span_scales:
        omega = _scaled_omega_grid(
            center_wavelength_nm=center_wavelength_nm,
            n_samples=n_samples,
            pulse_width_fs=pulse_width_fs,
            span_scale=span_scale,
        )
        delta_omega = omega - float(np.mean(omega))
        section = _section_iv_system(
            omega=omega,
            delta_omega=delta_omega,
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            center_wavelength_nm=center_wavelength_nm,
            focal_length_um=focal_length_um,
            a=a,
            a_prime=a_prime,
            b=b,
        )
        reference_phase = _section_iv_reference_phase(section, reference_mode=reference_mode)
        weights = np.ones_like(reference_phase, dtype=np.float64)
        reference_fit = fit_phase_taylor_affine_detrended(
            delta_omega,
            reference_phase,
            omega0_rad_per_fs=float(section.omega0_rad_per_fs),
            weights=weights,
            order=4,
        )

        variant_map = {
            "sample_k_series2": np.asarray(section.phi3_sample_k_series2_rad, dtype=np.float64),
            "center_k_series2": np.asarray(section.phi3_center_k_series2_rad, dtype=np.float64),
            "sample_k_series1": np.asarray(section.phi3_sample_k_series1_rad, dtype=np.float64),
            "center_k_series1": np.asarray(section.phi3_center_k_series1_rad, dtype=np.float64),
        }
        for variant, phase in variant_map.items():
            fit = fit_phase_taylor_affine_detrended(
                delta_omega,
                phase,
                omega0_rad_per_fs=float(section.omega0_rad_per_fs),
                weights=weights,
                order=4,
            )
            points.append(
                MartinezSectionIVPhasePoint(
                    span_scale=float(span_scale),
                    variant=variant,
                    gdd_fs2=float(fit.coefficients_rad[2]),
                    tod_fs3=float(fit.coefficients_rad[3]),
                    reference_gdd_fs2=float(reference_fit.coefficients_rad[2]),
                    reference_tod_fs3=float(reference_fit.coefficients_rad[3]),
                    gdd_abs_error_fs2=float(
                        abs(
                            float(fit.coefficients_rad[2])
                            - float(reference_fit.coefficients_rad[2])
                        )
                    ),
                    tod_abs_error_fs3=float(
                        abs(
                            float(fit.coefficients_rad[3])
                            - float(reference_fit.coefficients_rad[3])
                        )
                    ),
                    weighted_rms_rad=float(fit.weighted_rms_rad),
                )
            )
    return tuple(points)


def run_treacy_partition_span_study(
    *,
    span_scales: tuple[float, ...] = DEFAULT_SPAN_SCALES,
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
) -> tuple[TreacyPartitionPoint, ...]:
    points: list[TreacyPartitionPoint] = []
    analytic = compute_treacy_analytic_metrics(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        wavelength_nm=center_wavelength_nm,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )

    for span_scale in span_scales:
        pulse_width_fs = float(base_pulse_width_fs) / float(span_scale)
        laser = StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=pulse_width_fs,
                center_wavelength_nm=center_wavelength_nm,
                n_samples=n_samples,
                time_window_fs=time_window_fs,
            ),
            beam=BeamSpec(radius_mm=beam_radius_mm, m2=1.0),
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
        weights = np.asarray(result.fit.weights, dtype=np.float64)
        delta_omega = np.asarray(result.pipeline_result.delta_omega_rad_per_fs, dtype=np.float64)
        omega0 = float(result.pipeline_result.omega0_rad_per_fs)
        phi1 = _zero_if_none(result.pipeline_result.phi1_rad)
        phi2 = _zero_if_none(result.pipeline_result.phi2_rad)
        phi4 = _zero_if_none(result.pipeline_result.phi4_rad)
        phi0_axial = np.asarray(result.pipeline_result.phi0_axial_total_rad, dtype=np.float64)
        phi_geom = np.asarray(result.pipeline_result.phi_geom_total_rad, dtype=np.float64)
        phi3_current = np.asarray(result.pipeline_result.phi3_total_rad, dtype=np.float64)
        phi3_series1 = _phi3_variant_from_trace(trace, order=1, k_mode="center")
        phi3_series2 = _phi3_variant_from_trace(trace, order=2, k_mode="center")

        variant_map = {
            "axial_current": combine_phi_total_rad(phi0_axial, phi1, phi2, phi3_current, phi4),
            "geom_current": combine_phi_total_rad(phi_geom, phi1, phi2, phi3_current, phi4),
            "geom_series1": combine_phi_total_rad(phi_geom, phi1, phi2, phi3_series1, phi4),
            "geom_series2": combine_phi_total_rad(phi_geom, phi1, phi2, phi3_series2, phi4),
        }

        for variant, phase in variant_map.items():
            fit = fit_phase_taylor_affine_detrended(
                delta_omega,
                phase,
                omega0_rad_per_fs=omega0,
                weights=weights,
                order=4,
            )
            gdd = float(fit.coefficients_rad[2])
            tod = float(fit.coefficients_rad[3])
            points.append(
                TreacyPartitionPoint(
                    span_scale=float(span_scale),
                    pulse_width_fs=float(pulse_width_fs),
                    variant=variant,
                    raw_abcdef_gdd_fs2=gdd,
                    raw_abcdef_tod_fs3=tod,
                    analytic_gdd_fs2=float(analytic.gdd_fs2),
                    analytic_tod_fs3=float(analytic.tod_fs3),
                    raw_abcdef_gdd_rel_error=_relative_error(gdd, analytic.gdd_fs2),
                    raw_abcdef_tod_rel_error=_relative_error(tod, analytic.tod_fs3),
                    weighted_rms_rad=float(fit.weighted_rms_rad),
                )
            )
    return tuple(points)


def run_martinez_section_iv_phi3_variant_study(
    *,
    span_scales: tuple[float, ...] = DEFAULT_SPAN_SCALES,
    reference_mode: SectionIVReferenceMode = "legacy_linear",
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    center_wavelength_nm: float = 1030.0,
    focal_length_um: float = 300_000.0,
    a: float = 0.05,
    a_prime: float = 0.0,
    b: float = 0.0,
    n_samples: int = 256,
    pulse_width_fs: float = 100.0,
) -> tuple[MartinezSectionIVPhi3VariantPoint, ...]:
    points: list[MartinezSectionIVPhi3VariantPoint] = []
    for span_scale in span_scales:
        omega = _scaled_omega_grid(
            center_wavelength_nm=center_wavelength_nm,
            n_samples=n_samples,
            pulse_width_fs=pulse_width_fs,
            span_scale=span_scale,
        )
        delta_omega = omega - float(np.mean(omega))
        section = _section_iv_system(
            omega=omega,
            delta_omega=delta_omega,
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            center_wavelength_nm=center_wavelength_nm,
            focal_length_um=focal_length_um,
            a=a,
            a_prime=a_prime,
            b=b,
        )
        reference_phase = _section_iv_reference_phase(section, reference_mode=reference_mode)
        weights = np.ones_like(reference_phase, dtype=np.float64)
        reference_fit = fit_phase_taylor_affine_detrended(
            delta_omega,
            reference_phase,
            omega0_rad_per_fs=float(section.omega0_rad_per_fs),
            weights=weights,
            order=4,
        )

        for variant in PHI3_VARIANTS:
            phase = _section_iv_phi3_variant_phase(
                omega=np.asarray(omega, dtype=np.float64),
                delta_omega=np.asarray(delta_omega, dtype=np.float64),
                variant=variant,
                line_density_lpmm=line_density_lpmm,
                incidence_angle_deg=incidence_angle_deg,
                center_wavelength_nm=center_wavelength_nm,
                focal_length_um=focal_length_um,
                a=a,
                a_prime=a_prime,
                b=b,
            )
            fit = fit_phase_taylor_affine_detrended(
                delta_omega,
                phase,
                omega0_rad_per_fs=float(section.omega0_rad_per_fs),
                weights=weights,
                order=4,
            )
            points.append(
                MartinezSectionIVPhi3VariantPoint(
                    span_scale=float(span_scale),
                    variant=variant.name,
                    total_gdd_fs2=float(fit.coefficients_rad[2]),
                    total_tod_fs3=float(fit.coefficients_rad[3]),
                    reference_gdd_fs2=float(reference_fit.coefficients_rad[2]),
                    reference_tod_fs3=float(reference_fit.coefficients_rad[3]),
                    gdd_abs_error_fs2=float(
                        abs(
                            float(fit.coefficients_rad[2])
                            - float(reference_fit.coefficients_rad[2])
                        )
                    ),
                    tod_abs_error_fs3=float(
                        abs(
                            float(fit.coefficients_rad[3])
                            - float(reference_fit.coefficients_rad[3])
                        )
                    ),
                    weighted_rms_rad=float(
                        _weighted_rms(
                            _remove_affine_phase(
                                np.asarray(delta_omega, dtype=np.float64),
                                np.asarray(phase - reference_phase, dtype=np.float64),
                                weights=weights,
                            ),
                            weights,
                        )
                    ),
                )
            )
    return tuple(points)


def run_treacy_phi3_variant_comparison(
    *,
    beam_radii_mm: tuple[float, ...] = (1.0, 10.0, 100.0, 1000.0),
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
) -> tuple[TreacyPhi3VariantPoint, ...]:
    points: list[TreacyPhi3VariantPoint] = []
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
        weights = np.asarray(result.fit.weights, dtype=np.float64)
        delta_omega = np.asarray(result.pipeline_result.delta_omega_rad_per_fs, dtype=np.float64)
        omega0 = float(result.pipeline_result.omega0_rad_per_fs)
        analytic_phase = phase_from_treacy_dispersion(
            delta_omega,
            gdd_fs2=analytic.gdd_fs2,
            tod_fs3=analytic.tod_fs3,
        )
        base_terms = _treacy_non_phi3_phase_terms(result)
        weighted_xprime_rms = _weighted_xprime_rms_from_result(result)
        for variant in TREACY_PHI3_VARIANTS:
            variant_payload = _phi3_variant_payload_from_trace(trace, variant=variant)
            total_phase = combine_phi_total_rad(
                base_terms["phi_geom"],
                base_terms["filter_phase"],
                base_terms["phi1"],
                base_terms["phi2"],
                variant_payload["phi3_total"],
                base_terms["phi4"],
            )
            fit = fit_phase_taylor_affine_detrended(
                delta_omega,
                total_phase,
                omega0_rad_per_fs=omega0,
                weights=weights,
                order=4,
            )
            points.append(
                TreacyPhi3VariantPoint(
                    beam_radius_mm=float(beam_radius_mm),
                    variant=variant.name,
                    raw_abcdef_gdd_fs2=float(fit.coefficients_rad[2]),
                    raw_abcdef_tod_fs3=float(fit.coefficients_rad[3]),
                    analytic_gdd_fs2=float(analytic.gdd_fs2),
                    analytic_tod_fs3=float(analytic.tod_fs3),
                    raw_abcdef_gdd_rel_error=_relative_error(
                        float(fit.coefficients_rad[2]), analytic.gdd_fs2
                    ),
                    raw_abcdef_tod_rel_error=_relative_error(
                        float(fit.coefficients_rad[3]), analytic.tod_fs3
                    ),
                    weighted_rms_rad=float(
                        _weighted_rms(
                            _remove_affine_phase(
                                delta_omega,
                                total_phase - analytic_phase,
                                weights=weights,
                            ),
                            weights,
                        )
                    ),
                    weighted_xprime_rms=float(weighted_xprime_rms),
                )
            )
    return tuple(points)


def run_treacy_grating_frame_audit(
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
) -> tuple[TreacyGratingFramePoint, ...]:
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
    points: list[TreacyGratingFramePoint] = []
    for idx, item in enumerate(trace[:-1]):
        following = trace[idx + 1]
        if not isinstance(item.optic_spec, GratingCfg) or not isinstance(
            following.optic_spec, FrameTransformCfg
        ):
            continue
        x_after_grating = np.asarray(item.state_out.rays[:, 0, 0], dtype=np.float64)
        x_after_frame = np.asarray(following.state_out.rays[:, 0, 0], dtype=np.float64)
        xprime_after_grating = np.asarray(item.state_out.rays[:, 1, 0], dtype=np.float64)
        xprime_after_frame = np.asarray(following.state_out.rays[:, 1, 0], dtype=np.float64)
        points.append(
            TreacyGratingFramePoint(
                instance_name=item.cfg.instance_name,
                stage_index=int(item.index),
                following_frame_instance_name=following.cfg.instance_name,
                frame_stage_index=int(following.index),
                max_x_delta_um=float(np.max(np.abs(x_after_frame - x_after_grating))),
                max_xprime_flip_residual=float(
                    np.max(np.abs(xprime_after_frame + xprime_after_grating))
                ),
                x_rms_um=float(np.sqrt(np.mean(x_after_grating**2))),
                xprime_rms=float(np.sqrt(np.mean(xprime_after_grating**2))),
            )
        )
    return tuple(points)


def run_treacy_phi3_per_grating_budget(
    *,
    beam_radius_mm: float = 1000.0,
    variants: tuple[str, ...] | None = None,
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
) -> tuple[TreacyPhi3PerGratingPoint, ...]:
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
    weights = np.asarray(result.fit.weights, dtype=np.float64)
    delta_omega = np.asarray(result.pipeline_result.delta_omega_rad_per_fs, dtype=np.float64)
    omega0 = float(result.pipeline_result.omega0_rad_per_fs)
    analytic = compute_treacy_analytic_metrics(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        wavelength_nm=center_wavelength_nm,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    analytic_phase = phase_from_treacy_dispersion(
        delta_omega,
        gdd_fs2=analytic.gdd_fs2,
        tod_fs3=analytic.tod_fs3,
    )
    base_terms = _treacy_non_phi3_phase_terms(result)
    chosen_variants = (
        variants
        if variants is not None
        else (
            TREACY_PHI3_VARIANTS[0].name,
            _best_phi3_variant_name(
                run_treacy_phi3_variant_comparison(
                    beam_radii_mm=(10.0, 100.0, 1000.0),
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
            ),
        )
    )

    points: list[TreacyPhi3PerGratingPoint] = []
    for variant_name in chosen_variants:
        variant = _phi3_variant_config_by_name(variant_name)
        payload = _phi3_variant_payload_from_trace(trace, variant=variant)
        total_phase = combine_phi_total_rad(
            base_terms["phi_geom"],
            base_terms["filter_phase"],
            base_terms["phi1"],
            base_terms["phi2"],
            payload["phi3_total"],
            base_terms["phi4"],
        )
        residual = _remove_affine_phase(
            delta_omega,
            analytic_phase - total_phase,
            weights=weights,
        )
        per_stage_payload = cast(dict[str, dict[str, NDArrayF]], payload["per_stage"])
        for stage_name, stage_payload in per_stage_payload.items():
            phase_i = np.asarray(stage_payload["phi3_rad"], dtype=np.float64)
            phase_i_detrended = _remove_affine_phase(delta_omega, phase_i, weights=weights)
            denom = float(np.sum(weights * phase_i_detrended * phase_i_detrended))
            projection = (
                0.0
                if denom <= 0.0
                else float(np.sum(weights * residual * phase_i_detrended) / denom)
            )
            required_contrib = projection * phase_i
            fit = fit_phase_taylor_affine_detrended(
                delta_omega,
                phase_i,
                omega0_rad_per_fs=omega0,
                weights=weights,
                order=4,
            )
            required_fit = fit_phase_taylor_affine_detrended(
                delta_omega,
                required_contrib,
                omega0_rad_per_fs=omega0,
                weights=weights,
                order=4,
            )
            linear_coeff = np.polynomial.polynomial.polyfit(
                delta_omega,
                np.asarray(stage_payload["f_phase"], dtype=np.float64),
                deg=1,
            )[1]
            stage_trace = next(item for item in trace if item.cfg.instance_name == stage_name)
            points.append(
                TreacyPhi3PerGratingPoint(
                    variant=variant.name,
                    beam_radius_mm=float(beam_radius_mm),
                    instance_name=stage_name,
                    stage_index=int(stage_trace.index),
                    local_axis_angle_deg=float(
                        math.degrees(float(stage_trace.cfg.local_axis_angle_rad or 0.0))
                    ),
                    is_return_pass=_is_return_pass_stage(stage_trace),
                    f_phase_linear_coeff=float(linear_coeff),
                    f_phase_rms=float(
                        _weighted_rms(
                            np.asarray(stage_payload["f_phase"], dtype=np.float64),
                            weights,
                        )
                    ),
                    x_after_rms_um=float(
                        _weighted_rms(
                            np.asarray(stage_payload["x_after"], dtype=np.float64),
                            weights,
                        )
                    ),
                    phi3_gdd_fs2=float(fit.coefficients_rad[2]),
                    phi3_tod_fs3=float(fit.coefficients_rad[3]),
                    required_residual_projection=float(projection),
                    required_residual_gdd_fs2=float(required_fit.coefficients_rad[2]),
                    required_residual_tod_fs3=float(required_fit.coefficients_rad[3]),
                )
            )
    return tuple(points)


def run_treacy_phi3_sign_audit(
    *,
    beam_radius_mm: float = 1000.0,
    variant_name: str | None = None,
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
) -> tuple[tuple[TreacyPhi3SignStagePoint, ...], tuple[TreacyPhi3SignVariantPoint, ...]]:
    chosen_variant_name = (
        variant_name
        if variant_name is not None
        else _best_phi3_variant_name(
            run_treacy_phi3_variant_comparison(
                beam_radii_mm=(10.0, 100.0, 1000.0),
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
    )
    variant = _phi3_variant_config_by_name(chosen_variant_name)
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
    weights = np.asarray(result.fit.weights, dtype=np.float64)
    delta_omega = np.asarray(result.pipeline_result.delta_omega_rad_per_fs, dtype=np.float64)
    omega0 = float(result.pipeline_result.omega0_rad_per_fs)
    analytic = compute_treacy_analytic_metrics(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        wavelength_nm=center_wavelength_nm,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    base_terms = _treacy_non_phi3_phase_terms(result)

    sign_stage_points: list[TreacyPhi3SignStagePoint] = []
    base_payload = _phi3_variant_payload_from_trace(trace, variant=variant)
    base_per_stage = cast(dict[str, dict[str, NDArrayF]], base_payload["per_stage"])
    for item in _grating_stage_traces(trace):
        f_phase = np.asarray(base_per_stage[item.cfg.instance_name]["f_phase"], dtype=np.float64)
        linear_coeff = np.polynomial.polynomial.polyfit(delta_omega, f_phase, deg=1)[1]
        frame_angle = float(item.cfg.local_axis_angle_rad or 0.0)
        positive_x_axis_angle_deg = math.degrees(frame_angle) - 90.0
        sign_stage_points.append(
            TreacyPhi3SignStagePoint(
                instance_name=item.cfg.instance_name,
                stage_index=int(item.index),
                local_axis_angle_deg=float(math.degrees(frame_angle)),
                positive_x_axis_angle_deg=float(positive_x_axis_angle_deg),
                positive_z_axis_angle_deg=float(math.degrees(frame_angle)),
                is_return_pass=_is_return_pass_stage(item),
                return_flip_hypothesis=_is_return_pass_stage(item),
                f_phase_linear_coeff=float(linear_coeff),
            )
        )

    sign_variant_points: list[TreacyPhi3SignVariantPoint] = []
    for sign_case in ("current", "flip_return_pass", "flip_all_gratings"):
        payload = _phi3_variant_payload_from_trace(trace, variant=variant, sign_case=sign_case)
        total_phase = combine_phi_total_rad(
            base_terms["phi_geom"],
            base_terms["filter_phase"],
            base_terms["phi1"],
            base_terms["phi2"],
            payload["phi3_total"],
            base_terms["phi4"],
        )
        fit = fit_phase_taylor_affine_detrended(
            delta_omega,
            total_phase,
            omega0_rad_per_fs=omega0,
            weights=weights,
            order=4,
        )
        sign_variant_points.append(
            TreacyPhi3SignVariantPoint(
                variant=variant.name,
                sign_case=sign_case,
                beam_radius_mm=float(beam_radius_mm),
                raw_abcdef_gdd_rel_error=_relative_error(
                    float(fit.coefficients_rad[2]), analytic.gdd_fs2
                ),
                raw_abcdef_tod_rel_error=_relative_error(
                    float(fit.coefficients_rad[3]), analytic.tod_fs3
                ),
            )
        )

    return tuple(sign_stage_points), tuple(sign_variant_points)


def _section_iv_system(
    *,
    omega: NDArrayF,
    delta_omega: NDArrayF,
    line_density_lpmm: float,
    incidence_angle_deg: float,
    center_wavelength_nm: float,
    focal_length_um: float,
    a: float,
    a_prime: float,
    b: float,
) -> _SectionIVSystem:
    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    delta_arr = np.asarray(delta_omega, dtype=np.float64).reshape(-1)
    omega0 = float(np.mean(omega_arr))
    theta1 = np.deg2rad(float(incidence_angle_deg))
    wavelength_um = float(center_wavelength_nm) / 1000.0
    period_um = 1000.0 / float(line_density_lpmm)
    argument = (wavelength_um / period_um) - np.sin(theta1)
    theta2 = float(np.arcsin(argument))
    magnification = float(np.cos(theta2) / np.cos(theta1))

    fg_series1 = _truncate_grating_f(
        _exact_local_grating_f(
            omega_arr,
            omega0=omega0,
            period_um=period_um,
            incidence_angle_rad=theta1,
        ),
        delta_arr,
        order=1,
    )
    fg_series2 = _truncate_grating_f(
        _exact_local_grating_f(
            omega_arr,
            omega0=omega0,
            period_um=period_um,
            incidence_angle_rad=theta1,
        ),
        delta_arr,
        order=2,
    )

    telescope = _section_iv_telescope_matrix(
        omega_arr.size,
        focal_length_um=float(focal_length_um),
        a=float(a),
        a_prime=float(a_prime),
        b=float(b),
    )
    g1_series1 = _section_iv_first_grating_matrix(magnification, fg_series1)
    g2_series1 = _section_iv_second_grating_matrix(magnification, fg_series1)
    system_series1 = g2_series1 @ telescope @ g1_series1
    g1_series2 = _section_iv_first_grating_matrix(magnification, fg_series2)
    g2_series2 = _section_iv_second_grating_matrix(magnification, fg_series2)
    system_series2 = g2_series2 @ telescope @ g1_series2

    expected_b = -float(focal_length_um) * (float(a) + float(a_prime)) / (magnification**2)
    expected_c = 2.0 * (magnification**2) * float(b) / float(focal_length_um)
    expected_e = -fg_series2 * float(focal_length_um) * (float(a) + float(a_prime)) / magnification
    k_sample = martinez_k(omega_arr)
    k_center = martinez_k_center(omega_arr)
    e_series2 = extract_E(system_series2)
    e_series1 = extract_E(system_series1)
    phi3_sample_k_series2 = 0.5 * k_sample * (magnification * fg_series2) * e_series2
    phi3_center_k_series2 = 0.5 * k_center * (magnification * fg_series2) * e_series2
    phi3_sample_k_series1 = 0.5 * k_sample * (magnification * fg_series1) * e_series1
    phi3_center_k_series1 = 0.5 * k_center * (magnification * fg_series1) * e_series1
    reference_phase = (
        -0.5 * k_sample * fg_series2 * float(focal_length_um) * (float(a) + float(a_prime))
    )
    reference_phase_quadratic = 0.5 * k_sample * expected_b * (magnification * fg_series2) ** 2

    return _SectionIVSystem(
        omega0_rad_per_fs=omega0,
        system=system_series2,
        expected_b_um=expected_b,
        expected_c_per_um=expected_c,
        expected_e_um=expected_e,
        phi3_sample_k_series2_rad=phi3_sample_k_series2,
        phi3_center_k_series2_rad=phi3_center_k_series2,
        phi3_sample_k_series1_rad=phi3_sample_k_series1,
        phi3_center_k_series1_rad=phi3_center_k_series1,
        reference_phase_rad=reference_phase,
        reference_phase_quadratic_rad=reference_phase_quadratic,
    )


def _section_iv_telescope_matrix(
    batch_size: int,
    *,
    focal_length_um: float,
    a: float,
    a_prime: float,
    b: float,
) -> NDArrayF:
    matrices = np.repeat(np.eye(3, dtype=np.float64)[None, ...], batch_size, axis=0)

    def _propagation(length_um: float) -> NDArrayF:
        matrix = np.eye(3, dtype=np.float64)
        matrix[0, 1] = float(length_um)
        return matrix

    def _lens() -> NDArrayF:
        matrix = np.eye(3, dtype=np.float64)
        matrix[1, 0] = -1.0 / float(focal_length_um)
        return matrix

    for matrix in (
        _propagation(float(focal_length_um) * (1.0 + float(a_prime))),
        _lens(),
        _propagation(2.0 * float(focal_length_um) * (1.0 + float(b))),
        _lens(),
        _propagation(float(focal_length_um) * (1.0 + float(a))),
    ):
        matrices = np.broadcast_to(matrix, matrices.shape) @ matrices
    return matrices


def _section_iv_reference_phase(
    section: _SectionIVSystem,
    *,
    reference_mode: SectionIVReferenceMode,
) -> NDArrayF:
    if reference_mode == "legacy_linear":
        return np.asarray(section.reference_phase_rad, dtype=np.float64)
    return np.asarray(section.reference_phase_quadratic_rad, dtype=np.float64)


def _section_iv_first_grating_matrix(magnification: float, fg: NDArrayF) -> NDArrayF:
    matrices = np.zeros((fg.size, 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = float(magnification)
    matrices[:, 1, 1] = 1.0 / float(magnification)
    matrices[:, 1, 2] = fg
    matrices[:, 2, 2] = 1.0
    return matrices


def _section_iv_second_grating_matrix(magnification: float, fg: NDArrayF) -> NDArrayF:
    matrices = np.zeros((fg.size, 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1.0 / float(magnification)
    matrices[:, 1, 1] = float(magnification)
    matrices[:, 1, 2] = float(magnification) * fg
    matrices[:, 2, 2] = 1.0
    return matrices


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


def _phi3_variant_from_trace(
    trace: tuple[_StageTrace, ...],
    *,
    order: int,
    k_mode: Literal["center", "sample"],
) -> NDArrayF:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    delta_omega = trace[0].cfg.delta_omega_rad_per_fs
    if delta_omega is None:
        raise ValueError("stage cfg delta_omega_rad_per_fs must not be None")
    delta = np.asarray(delta_omega, dtype=np.float64).reshape(-1)
    if k_mode == "center":
        k_arr = np.full_like(omega, martinez_k_center(omega), dtype=np.float64)
    else:
        k_arr = martinez_k(omega)
    total = np.zeros_like(omega)
    for item in trace:
        f_exact = np.asarray(item.cfg.abcdef[:, 1, 2], dtype=np.float64)
        if np.max(np.abs(f_exact)) <= 0.0:
            continue
        f_use = _truncate_grating_f(f_exact, delta, order=order)
        x_after = np.asarray(item.state_out.rays[:, 0, 0], dtype=np.float64)
        total = total + (0.5 * k_arr * f_use * x_after)
    return total


def _section_iv_phi3_variant_phase(
    *,
    omega: NDArrayF,
    delta_omega: NDArrayF,
    variant: _Phi3VariantConfig,
    line_density_lpmm: float,
    incidence_angle_deg: float,
    center_wavelength_nm: float,
    focal_length_um: float,
    a: float,
    a_prime: float,
    b: float,
) -> NDArrayF:
    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    delta_arr = np.asarray(delta_omega, dtype=np.float64).reshape(-1)
    omega0 = float(np.mean(omega_arr))
    theta1_rad = math.radians(float(incidence_angle_deg))
    theta2_deg = compute_grating_diffraction_angle_deg(
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        wavelength_nm=float(center_wavelength_nm),
        diffraction_order=-1,
    )
    theta2_rad = math.radians(float(theta2_deg))
    telescope = _section_iv_telescope_matrix(
        omega_arr.size,
        focal_length_um=float(focal_length_um),
        a=float(a),
        a_prime=float(a_prime),
        b=float(b),
    )
    g1_exact = _section_iv_exact_grating_matrix(
        omega_arr,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
    )
    g2_exact = _section_iv_exact_grating_matrix(
        omega_arr,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=float(theta2_deg),
    )
    rays0 = np.zeros((omega_arr.size, 3, 1), dtype=np.float64)
    rays0[:, 2, 0] = 1.0

    series_g1 = _section_iv_phase_series(
        omega_arr,
        delta_arr,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_rad=theta1_rad,
    )
    series_g2 = _section_iv_phase_series(
        omega_arr,
        delta_arr,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_rad=theta2_rad,
    )
    if variant.x_mode == "runtime":
        rays_after_g1 = g1_exact @ rays0
        rays_after_g2 = g2_exact @ (telescope @ rays_after_g1)
        x_after_map = {
            "g1": np.asarray(rays_after_g1[:, 0, 0], dtype=np.float64),
            "g2": np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64),
        }
    else:
        g1_phase = _section_iv_phase_matrix(
            variant=variant,
            phase_series=series_g1,
            exact_matrix=g1_exact,
        )
        g2_phase = _section_iv_phase_matrix(
            variant=variant,
            phase_series=series_g2,
            exact_matrix=g2_exact,
        )
        rays_after_g1 = g1_phase @ rays0
        rays_after_g2 = g2_phase @ (telescope @ rays_after_g1)
        x_after_map = {
            "g1": np.asarray(rays_after_g1[:, 0, 0], dtype=np.float64),
            "g2": np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64),
        }

    k_arr = _k_array_for_variant(omega_arr, variant=variant)
    total = np.zeros_like(omega_arr)
    for stage_name, phase_series in (("g1", series_g1), ("g2", series_g2)):
        f_phase = _select_f_phase(phase_series, variant=variant)
        total = total + (0.5 * k_arr * f_phase * x_after_map[stage_name])
    return total


def _section_iv_exact_grating_matrix(
    omega: NDArrayF,
    *,
    omega0: float,
    line_density_lpmm: float,
    incidence_angle_deg: float,
) -> NDArrayF:
    from abcdef_sim.optics.grating import Grating

    return Grating(
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        diffraction_order=-1,
        immersion_refractive_index=1.0,
    ).matrix(omega, omega0=omega0)


def _section_iv_phase_matrix(
    *,
    variant: _Phi3VariantConfig,
    phase_series: _GratingPhaseSeries,
    exact_matrix: NDArrayF,
) -> NDArrayF:
    if variant.x_mode != "phase_ad":
        return np.asarray(exact_matrix, dtype=np.float64)
    matrices = np.asarray(exact_matrix, dtype=np.float64).copy()
    matrices[:, 0, 0] = phase_series.a_series1
    matrices[:, 1, 1] = phase_series.d_series1
    matrices[:, 1, 2] = _select_f_phase(phase_series, variant=variant)
    return matrices


def _section_iv_phase_series(
    omega: NDArrayF,
    delta_omega: NDArrayF,
    *,
    omega0: float,
    line_density_lpmm: float,
    incidence_angle_rad: float,
) -> _GratingPhaseSeries:
    period_um = 1000.0 / float(line_density_lpmm)
    f_exact = _exact_local_grating_f(
        omega,
        omega0=omega0,
        period_um=period_um,
        incidence_angle_rad=incidence_angle_rad,
    )
    return _martinez_phase_series_from_exact(
        f_exact=f_exact,
        delta_omega=delta_omega,
        omega0=omega0,
        period_um=period_um,
        incidence_angle_rad=incidence_angle_rad,
        immersion_refractive_index=1.0,
    )


def _phi3_variant_payload_from_trace(
    trace: tuple[_StageTrace, ...],
    *,
    variant: _Phi3VariantConfig,
    sign_case: Literal["current", "flip_return_pass", "flip_all_gratings"] = "current",
) -> dict[str, object]:
    if not trace:
        raise ValueError("trace must contain at least one stage")
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = _k_array_for_variant(omega, variant=variant)
    x_after_map = (
        _runtime_x_after_map(trace)
        if variant.x_mode == "runtime"
        else _phase_side_x_after_map(trace, variant=variant, sign_case=sign_case)
    )

    total = np.zeros_like(omega)
    per_stage: dict[str, dict[str, NDArrayF]] = {}
    for item in _grating_stage_traces(trace):
        series = _phase_series_for_trace_item(item)
        f_phase = _select_f_phase(series, variant=variant)
        if sign_case == "flip_return_pass" and _is_return_pass_stage(item):
            f_phase = -f_phase
        elif sign_case == "flip_all_gratings":
            f_phase = -f_phase
        x_after = np.asarray(x_after_map[item.cfg.instance_name], dtype=np.float64)
        phi3_i = 0.5 * k_arr * f_phase * x_after
        boundary_correction = _boundary_probe_correction_term(
            item,
            f_phase=f_phase,
            variant=variant,
        )
        phi3_i = phi3_i + boundary_correction
        total = total + phi3_i
        per_stage[item.cfg.instance_name] = {
            "f_phase": np.asarray(f_phase, dtype=np.float64),
            "x_after": x_after,
            "boundary_correction_rad": boundary_correction,
            "phi3_rad": phi3_i,
        }
    return {"phi3_total": total, "per_stage": per_stage}


def _boundary_probe_correction_term(
    item: _StageTrace,
    *,
    f_phase: NDArrayF,
    variant: _Phi3VariantConfig,
) -> NDArrayF:
    if variant.boundary_mode == "none":
        return np.zeros_like(np.asarray(f_phase, dtype=np.float64), dtype=np.float64)
    if not _has_nonunit_upstream_magnification(item):
        return np.zeros_like(np.asarray(f_phase, dtype=np.float64), dtype=np.float64)

    omega = np.asarray(item.cfg.omega, dtype=np.float64).reshape(-1)
    b_pre = np.asarray(item.state_in.system[:, 0, 1], dtype=np.float64)
    return (
        float(variant.boundary_weight)
        * 0.5
        * martinez_k(omega)
        * np.asarray(f_phase, dtype=np.float64)
        * b_pre
    )


def _has_nonunit_upstream_magnification(item: _StageTrace) -> bool:
    a_pre = np.asarray(item.state_in.system[:, 0, 0], dtype=np.float64)
    d_pre = np.asarray(item.state_in.system[:, 1, 1], dtype=np.float64)
    return not (
        np.allclose(a_pre, 1.0, rtol=0.0, atol=1e-12)
        and np.allclose(d_pre, 1.0, rtol=0.0, atol=1e-12)
    )


def _runtime_x_after_map(trace: tuple[_StageTrace, ...]) -> dict[str, NDArrayF]:
    return {
        item.cfg.instance_name: np.asarray(item.state_out.rays[:, 0, 0], dtype=np.float64)
        for item in _grating_stage_traces(trace)
    }


def _phase_side_x_after_map(
    trace: tuple[_StageTrace, ...],
    *,
    variant: _Phi3VariantConfig,
    sign_case: Literal["current", "flip_return_pass", "flip_all_gratings"] = "current",
) -> dict[str, NDArrayF]:
    rays = np.asarray(trace[0].state_in.rays, dtype=np.float64)
    x_after: dict[str, NDArrayF] = {}
    for item in trace:
        matrix = np.asarray(item.cfg.abcdef, dtype=np.float64)
        if isinstance(item.optic_spec, GratingCfg):
            series = _phase_series_for_trace_item(item)
            f_phase = _select_f_phase(series, variant=variant)
            if sign_case == "flip_return_pass" and _is_return_pass_stage(item):
                f_phase = -f_phase
            elif sign_case == "flip_all_gratings":
                f_phase = -f_phase
            matrix = matrix.copy()
            matrix[:, 0, 0] = series.a_series1
            matrix[:, 1, 1] = series.d_series1
            matrix[:, 1, 2] = f_phase
        rays = matrix @ rays
        if isinstance(item.optic_spec, GratingCfg):
            x_after[item.cfg.instance_name] = np.asarray(rays[:, 0, 0], dtype=np.float64)
    return x_after


def _phase_series_for_trace_item(item: _StageTrace) -> _GratingPhaseSeries:
    if not isinstance(item.optic_spec, GratingCfg):
        raise TypeError(f"Expected GratingCfg, got {type(item.optic_spec).__name__}")
    delta_omega = np.asarray(item.cfg.delta_omega_rad_per_fs, dtype=np.float64).reshape(-1)
    omega0 = float(item.cfg.omega0_rad_per_fs)
    period_um = 1000.0 / float(item.optic_spec.line_density_lpmm)
    incidence_angle_rad = math.radians(float(item.optic_spec.incidence_angle_deg))
    f_exact = np.asarray(item.cfg.abcdef[:, 1, 2], dtype=np.float64)
    return _martinez_phase_series_from_exact(
        f_exact=f_exact,
        delta_omega=delta_omega,
        omega0=omega0,
        period_um=period_um,
        incidence_angle_rad=incidence_angle_rad,
        immersion_refractive_index=float(item.optic_spec.immersion_refractive_index),
    )


def _martinez_phase_series_from_exact(
    *,
    f_exact: NDArrayF,
    delta_omega: NDArrayF,
    omega0: float,
    period_um: float,
    incidence_angle_rad: float,
    immersion_refractive_index: float,
) -> _GratingPhaseSeries:
    delta = np.asarray(delta_omega, dtype=np.float64).reshape(-1)
    f_exact_arr = np.asarray(f_exact, dtype=np.float64).reshape(-1)
    if f_exact_arr.shape != delta.shape:
        raise ValueError("f_exact and delta_omega must share one shape")
    theta2_ref = _diffraction_angle_from_geometry(
        omega0=omega0,
        period_um=period_um,
        incidence_angle_rad=incidence_angle_rad,
        diffraction_order=-1.0,
    )
    n_medium = float(immersion_refractive_index)
    f0 = (
        -(
            (2.0 * np.pi * 0.299792458 * n_medium**2)
            / (period_um * omega0**2 * math.cos(theta2_ref))
        )
        * delta
    )
    if float(np.dot(f0, f_exact_arr)) < 0.0:
        f0 = -f0
    f_series2 = f0 * (1.0 - (2.0 * delta / float(omega0)) + (0.5 * math.tan(theta2_ref) * f0))
    a0 = math.cos(theta2_ref) / math.cos(float(incidence_angle_rad))
    d0 = math.cos(float(incidence_angle_rad)) / math.cos(theta2_ref)
    d_series1 = d0 * (1.0 - ((1.0 / n_medium) * math.tan(theta2_ref) * f0))
    a_series1 = a0 * (1.0 - ((1.0 / n_medium) * math.sin(theta2_ref) * f0))
    return _GratingPhaseSeries(
        f_exact=f_exact_arr,
        f_linear=f0,
        f_series2=f_series2,
        a_series1=np.full_like(f_exact_arr, a0, dtype=np.float64)
        if np.max(np.abs(f0)) <= 0.0
        else np.asarray(a_series1, dtype=np.float64),
        d_series1=np.full_like(f_exact_arr, d0, dtype=np.float64)
        if np.max(np.abs(f0)) <= 0.0
        else np.asarray(d_series1, dtype=np.float64),
    )


def _select_f_phase(series: _GratingPhaseSeries, *, variant: _Phi3VariantConfig) -> NDArrayF:
    if variant.f_mode == "transport_exact":
        return np.asarray(series.f_exact, dtype=np.float64)
    if variant.f_mode == "linear_f0":
        return np.asarray(series.f_linear, dtype=np.float64)
    return np.asarray(series.f_series2, dtype=np.float64)


def _k_array_for_variant(omega: NDArrayF, *, variant: _Phi3VariantConfig) -> NDArrayF:
    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    if variant.k_mode == "sample":
        return martinez_k(omega_arr)
    return np.full_like(omega_arr, martinez_k_center(omega_arr), dtype=np.float64)


def _grating_stage_traces(trace: tuple[_StageTrace, ...]) -> tuple[_StageTrace, ...]:
    return tuple(item for item in trace if isinstance(item.optic_spec, GratingCfg))


def _is_return_pass_stage(item: _StageTrace) -> bool:
    return item.cfg.instance_name.endswith("_return")


def _diffraction_angle_from_geometry(
    *,
    omega0: float,
    period_um: float,
    incidence_angle_rad: float,
    diffraction_order: float,
) -> float:
    wavelength_um = (2.0 * np.pi * 0.299792458) / float(omega0)
    diff_arg = -float(diffraction_order) * (wavelength_um / float(period_um)) - math.sin(
        float(incidence_angle_rad)
    )
    if diff_arg < -1.0 or diff_arg > 1.0:
        raise ValueError("diffraction geometry is outside the asin domain")
    return math.asin(diff_arg)


def _treacy_non_phi3_phase_terms(result: AbcdefRunResult) -> dict[str, NDArrayF]:
    phi_geom = np.asarray(result.pipeline_result.phi_geom_total_rad, dtype=np.float64)
    phi1 = _zero_or_array(result.pipeline_result.phi1_rad, phi_geom)
    phi2 = _zero_or_array(result.pipeline_result.phi2_rad, phi_geom)
    phi4 = _zero_or_array(result.pipeline_result.phi4_rad, phi_geom)
    filter_phase = np.sum(
        [
            np.zeros_like(phi_geom, dtype=np.float64)
            if contribution.filter_phase_rad is None
            else np.asarray(contribution.filter_phase_rad, dtype=np.float64)
            for contribution in result.pipeline_result.contributions
        ],
        axis=0,
    )
    return {
        "phi_geom": phi_geom,
        "phi1": phi1,
        "phi2": phi2,
        "phi4": phi4,
        "filter_phase": filter_phase,
    }


def _weighted_xprime_rms_from_result(result: AbcdefRunResult) -> float:
    rays = np.asarray(result.pipeline_result.final_state.rays, dtype=np.float64)
    weights = np.asarray(result.fit.weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    return float(np.sqrt(np.sum(weights * rays[:, 1, 0] ** 2)))


def _remove_affine_phase(delta_omega: NDArrayF, phase: NDArrayF, *, weights: NDArrayF) -> NDArrayF:
    delta_arr = np.asarray(delta_omega, dtype=np.float64).reshape(-1)
    phase_arr = np.asarray(phase, dtype=np.float64).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    design = np.column_stack([np.ones_like(delta_arr), delta_arr])
    sqrt_weights = np.sqrt(weights_arr)
    weighted_design = design * sqrt_weights[:, None]
    weighted_phase = phase_arr * sqrt_weights
    coeffs, *_ = np.linalg.lstsq(weighted_design, weighted_phase, rcond=None)
    return phase_arr - (design @ coeffs)


def _weighted_rms(values: NDArrayF, weights: NDArrayF) -> float:
    value_arr = np.asarray(values, dtype=np.float64).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    weights_arr = weights_arr / np.sum(weights_arr)
    return float(np.sqrt(np.sum(weights_arr * value_arr**2)))


def _phi3_variant_config_by_name(name: str) -> _Phi3VariantConfig:
    for variant in TREACY_PHI3_VARIANTS:
        if variant.name == name:
            return variant
    raise KeyError(f"Unknown phi3 variant {name!r}")


def _best_phi3_variant_name(points: tuple[TreacyPhi3VariantPoint, ...]) -> str:
    grouped: dict[str, list[TreacyPhi3VariantPoint]] = {}
    for point in points:
        if point.beam_radius_mm < 10.0:
            continue
        grouped.setdefault(point.variant, []).append(point)
    if not grouped:
        raise ValueError("points must contain at least one beam_radius_mm >= 10.0")

    def _score(items: list[TreacyPhi3VariantPoint]) -> float:
        return float(
            np.mean(
                [point.raw_abcdef_gdd_rel_error + point.raw_abcdef_tod_rel_error for point in items]
            )
        )

    return min(grouped.items(), key=lambda item: _score(item[1]))[0]


def _truncate_grating_f(f_exact: NDArrayF, delta_omega: NDArrayF, *, order: int) -> NDArrayF:
    coeffs = np.polynomial.polynomial.polyfit(delta_omega, f_exact, deg=order)
    coeffs[0] = 0.0
    return np.polynomial.polynomial.polyval(delta_omega, coeffs)


def _exact_local_grating_f(
    omega: NDArrayF,
    *,
    omega0: float,
    period_um: float,
    incidence_angle_rad: float,
) -> NDArrayF:
    c_um_per_fs = 0.299792458
    wavelength_um = (2.0 * np.pi * c_um_per_fs) / np.asarray(omega, dtype=np.float64)
    wavelength0_um = (2.0 * np.pi * c_um_per_fs) / float(omega0)
    theta_ref = np.arcsin((wavelength0_um / float(period_um)) - np.sin(float(incidence_angle_rad)))
    theta = np.arcsin((wavelength_um / float(period_um)) - np.sin(float(incidence_angle_rad)))
    return theta_ref - theta


def _scaled_omega_grid(
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


def _relative_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return abs(float(value))
    return abs(float(value) - float(reference)) / abs(float(reference))


def _zero_or_array(value: object | None, template: NDArrayF) -> NDArrayF:
    if value is None:
        return np.zeros_like(np.asarray(template, dtype=np.float64), dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _zero_if_none(value: object | None) -> NDArrayF:
    if value is None:
        raise ValueError("expected a phase term array, got None")
    return np.asarray(value, dtype=np.float64)
