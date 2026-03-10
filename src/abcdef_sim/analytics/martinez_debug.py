from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np

from abcdef_sim import BeamSpec, PulseSpec, StandaloneLaserSpec, run_abcdef, treacy_compressor_preset
from abcdef_sim.cache.backend import NullCacheBackend
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.specs import AbcdefCfg
from abcdef_sim.data_models.states import RayState
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.physics.abcdef.adapters import apply_cfg
from abcdef_sim.physics.abcdef.conventions import extract_A, extract_B, extract_C, extract_D, extract_E, extract_F
from abcdef_sim.physics.abcdef.dispersion import fit_phase_taylor_affine_detrended
from abcdef_sim.physics.abcdef.phase_terms import combine_phi_total_rad, martinez_k, martinez_k_center
from abcdef_sim.physics.abcdef.treacy import compute_treacy_analytic_metrics
from abcdef_sim.pipeline._assembler import SystemAssembler
from abcdef_sim.runner import _initial_ray_state, _laser_spec_from_state, _resolve_runtime_cfg
from abcdef_sim.physics.abcdef.pulse import build_standalone_laser_state

NDArrayF = np.ndarray

DEFAULT_SPAN_SCALES: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)

__all__ = [
    "DEFAULT_SPAN_SCALES",
    "MartinezSectionIVCoefficientPoint",
    "MartinezSectionIVPhasePoint",
    "TreacyPartitionPoint",
    "run_martinez_section_iv_coefficient_audit",
    "run_martinez_section_iv_phase_study",
    "run_treacy_partition_span_study",
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
class _StageTrace:
    cfg: OpticStageCfg
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
        reference_phase = np.asarray(section.reference_phase_rad, dtype=np.float64)
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
                        abs(float(fit.coefficients_rad[2]) - float(reference_fit.coefficients_rad[2]))
                    ),
                    tod_abs_error_fs3=float(
                        abs(float(fit.coefficients_rad[3]) - float(reference_fit.coefficients_rad[3]))
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
    phi3_sample_k_series2 = 0.5 * k_sample * (magnification * fg_series2) * extract_E(system_series2)
    phi3_center_k_series2 = 0.5 * k_center * (magnification * fg_series2) * extract_E(system_series2)
    phi3_sample_k_series1 = 0.5 * k_sample * (magnification * fg_series1) * extract_E(system_series1)
    phi3_center_k_series1 = 0.5 * k_center * (magnification * fg_series1) * extract_E(system_series1)
    reference_phase = -0.5 * k_sample * fg_series2 * float(focal_length_um) * (float(a) + float(a_prime))

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
    state = _initial_ray_state(cfg=resolved_cfg, laser=internal_laser_spec)
    trace: list[_StageTrace] = []
    for stage_cfg in stage_cfgs:
        state, contribution = apply_cfg(state, stage_cfg, policy=None)
        trace.append(_StageTrace(cfg=stage_cfg, state_out=state, contribution=contribution))
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


def _zero_if_none(value: object | None) -> NDArrayF:
    if value is None:
        raise ValueError("expected a phase term array, got None")
    return np.asarray(value, dtype=np.float64)
