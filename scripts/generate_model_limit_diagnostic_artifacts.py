#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


SCENARIOS: tuple[tuple[str, dict[str, float]], ...] = (
    (
        "baseline_35deg_mirror0",
        {
            "incidence_angle_deg": 35.0,
            "line_density_lpmm": 1200.0,
            "separation_um": 100_000.0,
            "length_to_mirror_um": 0.0,
        },
    ),
    (
        "angle_30deg_mirror0",
        {
            "incidence_angle_deg": 30.0,
            "line_density_lpmm": 1200.0,
            "separation_um": 100_000.0,
            "length_to_mirror_um": 0.0,
        },
    ),
    (
        "baseline_35deg_mirror100mm",
        {
            "incidence_angle_deg": 35.0,
            "line_density_lpmm": 1200.0,
            "separation_um": 100_000.0,
            "length_to_mirror_um": 100_000.0,
        },
    ),
    (
        "density_1000_mirror0",
        {
            "incidence_angle_deg": 35.0,
            "line_density_lpmm": 1000.0,
            "separation_um": 100_000.0,
            "length_to_mirror_um": 0.0,
        },
    ),
)


def generate_artifacts(output_dir: Path) -> tuple[Path, ...]:
    env = _load_env()

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "treacy_k_consistency": {
            name: _treacy_k_consistency(env, params=params) for name, params in SCENARIOS
        },
        "treacy_chief_ray_action": {
            name: _treacy_chief_ray_action_candidates(env, params=params)
            for name, params in SCENARIOS
        },
        "section_iv_surface_candidates": _section_iv_surface_candidates(env),
        "section_iv_second_grating_audit": _section_iv_second_grating_audit(env),
        "notes": {
            "chief_ray_action": (
                "chief_ray_path_only sums exact free-space propagation on the fixed resolved "
                "Treacy surfaces and projects the final point to one common output plane. "
                "chief_ray_path_plus_surface_pm adds a local reflective-grating phase-matching "
                "surface term k * (sin(theta_in) + sin(theta_out)) * s at each grating hit."
            ),
            "k_consistency": (
                "all_center is the current runtime convention; all_sample replaces phi2, phi3, "
                "and phi4 with sample-k variants while leaving phi1 untouched."
            ),
        },
    }

    json_path = output_dir / "model_limit_diagnostics.json"
    k_plot_path = output_dir / "k_consistency_treacy.png"
    chief_ray_plot_path = output_dir / "chief_ray_action_candidates.png"
    section_plot_path = output_dir / "section_iv_surface_candidates.png"

    json_path.write_text(json.dumps(payload, indent=2))
    _plot_k_consistency(env, payload["treacy_k_consistency"], k_plot_path)
    _plot_chief_ray_candidates(env, payload["treacy_chief_ray_action"], chief_ray_plot_path)
    _plot_section_iv_surface_candidates(
        env,
        payload["section_iv_surface_candidates"],
        section_plot_path,
    )
    return (json_path, k_plot_path, chief_ray_plot_path, section_plot_path)


def _load_env() -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    from abcdef_sim import BeamSpec, PulseSpec, StandaloneLaserSpec, run_abcdef, treacy_compressor_preset
    from abcdef_sim.analytics import martinez_debug as md
    from abcdef_sim.cache.backend import NullCacheBackend
    from abcdef_sim.cfg_generator import OpticStageCfgGenerator
    from abcdef_sim.data_models.specs import FrameTransformCfg, FreeSpaceCfg, GratingCfg
    from abcdef_sim.optics.grating import _diffraction_angle_rad
    from abcdef_sim.optics.registry import OpticFactory
    from abcdef_sim.physics.abcdef.dispersion import fit_phase_taylor_affine_detrended
    from abcdef_sim.physics.abcdef.phase_terms import combine_phi_total_rad, martinez_k
    from abcdef_sim.physics.abcdef.treacy import (
        compute_grating_diffraction_angle_deg,
        compute_treacy_analytic_metrics,
    )
    from abcdef_sim.physics.geometry.surfaces import SurfacePlane1D
    from abcdef_sim.pipeline._assembler import SystemAssembler

    return {
        "plt": plt,
        "np": np,
        "BeamSpec": BeamSpec,
        "PulseSpec": PulseSpec,
        "StandaloneLaserSpec": StandaloneLaserSpec,
        "run_abcdef": run_abcdef,
        "treacy_compressor_preset": treacy_compressor_preset,
        "fit_phase_taylor_affine_detrended": fit_phase_taylor_affine_detrended,
        "combine_phi_total_rad": combine_phi_total_rad,
        "martinez_k": martinez_k,
        "compute_grating_diffraction_angle_deg": compute_grating_diffraction_angle_deg,
        "compute_treacy_analytic_metrics": compute_treacy_analytic_metrics,
        "OpticFactory": OpticFactory,
        "OpticStageCfgGenerator": OpticStageCfgGenerator,
        "NullCacheBackend": NullCacheBackend,
        "SystemAssembler": SystemAssembler,
        "FrameTransformCfg": FrameTransformCfg,
        "FreeSpaceCfg": FreeSpaceCfg,
        "GratingCfg": GratingCfg,
        "SurfacePlane1D": SurfacePlane1D,
        "_diffraction_angle_rad": _diffraction_angle_rad,
        "md": md,
    }


def _zero_or_array(value: Any, template: Any) -> Any:
    np = __import__("numpy")
    if value is None:
        return np.zeros_like(template, dtype=np.float64)
    return np.asarray(value, dtype=np.float64)


def _filter_phase_total(result: Any) -> Any:
    np = __import__("numpy")
    phi_geom = np.asarray(result.pipeline_result.phi_geom_total_rad, dtype=np.float64)
    total = np.zeros_like(phi_geom)
    for contribution in result.pipeline_result.contributions:
        if contribution.filter_phase_rad is not None:
            total = total + np.asarray(contribution.filter_phase_rad, dtype=np.float64)
    return total


def _fit_metrics(
    env: dict[str, Any],
    *,
    total_phase: Any,
    result: Any,
    analytic: Any,
) -> dict[str, float]:
    np = env["np"]
    fit = env["fit_phase_taylor_affine_detrended"](
        np.asarray(result.pipeline_result.delta_omega_rad_per_fs, dtype=np.float64),
        np.asarray(total_phase, dtype=np.float64),
        omega0_rad_per_fs=float(result.pipeline_result.omega0_rad_per_fs),
        weights=np.asarray(result.fit.weights, dtype=np.float64),
        order=4,
    )
    gdd = float(fit.coefficients_rad[2])
    tod = float(fit.coefficients_rad[3])
    return {
        "gdd_fs2": gdd,
        "tod_fs3": tod,
        "gdd_rel_error": abs((gdd - analytic.gdd_fs2) / analytic.gdd_fs2),
        "tod_rel_error": abs((tod - analytic.tod_fs3) / analytic.tod_fs3),
        "weighted_rms_rad": float(fit.weighted_rms_rad),
    }


def _treacy_result_and_trace(
    env: dict[str, Any],
    *,
    params: dict[str, float],
) -> tuple[Any, Any, Any, Any, Any]:
    laser = env["StandaloneLaserSpec"](
        pulse=env["PulseSpec"](
            width_fs=100.0,
            center_wavelength_nm=1030.0,
            n_samples=256,
            time_window_fs=3000.0,
        ),
        beam=env["BeamSpec"](radius_mm=1000.0, m2=1.0),
    )
    cfg = env["treacy_compressor_preset"](
        line_density_lpmm=params["line_density_lpmm"],
        incidence_angle_deg=params["incidence_angle_deg"],
        separation_um=params["separation_um"],
        length_to_mirror_um=params["length_to_mirror_um"],
        diffraction_order=-1,
        n_passes=2,
    )
    result = env["run_abcdef"](cfg, laser)
    trace = env["md"]._trace_cfg_run(cfg, laser)
    analytic = env["compute_treacy_analytic_metrics"](
        line_density_lpmm=params["line_density_lpmm"],
        incidence_angle_deg=params["incidence_angle_deg"],
        separation_um=params["separation_um"],
        wavelength_nm=1030.0,
        diffraction_order=-1,
        n_passes=2,
    )
    return laser, cfg, result, trace, analytic


def _phi2_sample(initial_rays: Any, final_rays: Any, omega: Any, *, martinez_k: Any, np: Any) -> Any:
    k_arr = martinez_k(omega)
    x_in = np.asarray(initial_rays[:, 0, 0], dtype=np.float64)
    x_prime_in = np.asarray(initial_rays[:, 1, 0], dtype=np.float64)
    x_out = np.asarray(final_rays[:, 0, 0], dtype=np.float64)
    x_prime_out = np.asarray(final_rays[:, 1, 0], dtype=np.float64)
    return 0.5 * k_arr * (x_in * x_prime_in - x_out * x_prime_out)


def _phi4_sample(x_out: Any, q_out: Any, omega: Any, *, martinez_k: Any, np: Any) -> Any:
    return np.real(martinez_k(omega) * np.asarray(x_out, dtype=np.float64) ** 2 / (2.0 * q_out))


def _treacy_k_consistency(
    env: dict[str, Any],
    *,
    params: dict[str, float],
) -> dict[str, dict[str, float]]:
    np = env["np"]
    md = env["md"]
    _, _, result, trace, analytic = _treacy_result_and_trace(env, params=params)

    omega = np.asarray(result.pipeline_result.omega, dtype=np.float64)
    phi_geom = np.asarray(result.pipeline_result.phi_geom_total_rad, dtype=np.float64)
    phi1 = _zero_or_array(result.pipeline_result.phi1_rad, phi_geom)
    filter_phase = _filter_phase_total(result)
    phi2_center = _zero_or_array(result.pipeline_result.phi2_rad, phi_geom)
    phi4_center = _zero_or_array(result.pipeline_result.phi4_rad, phi_geom)

    variant_center = md._Phi3VariantConfig(
        name="diag_center",
        f_mode="series2",
        x_mode="runtime",
        k_mode="center",
    )
    variant_sample = md._Phi3VariantConfig(
        name="diag_sample",
        f_mode="series2",
        x_mode="runtime",
        k_mode="sample",
    )
    phi3_center = np.asarray(
        md._phi3_variant_payload_from_trace(trace, variant=variant_center)["phi3_total"],
        dtype=np.float64,
    )
    phi3_sample = np.asarray(
        md._phi3_variant_payload_from_trace(trace, variant=variant_sample)["phi3_total"],
        dtype=np.float64,
    )

    omega_for_beam = np.asarray(result.pipeline_result.omega, dtype=np.float64)
    q_out = env["md"]._beam_phase_inputs if False else None
    from abcdef_sim.runner import _beam_phase_inputs

    _, _, _, q_out = _beam_phase_inputs(
        omega=omega_for_beam,
        beam_radius_mm=1000.0,
        m2=1.0,
        final_system=np.asarray(result.pipeline_result.final_state.system, dtype=np.float64),
    )
    phi2_sample = _phi2_sample(
        trace[0].state_in.rays,
        result.pipeline_result.final_state.rays,
        omega,
        martinez_k=env["martinez_k"],
        np=np,
    )
    phi4_sample = _phi4_sample(
        result.pipeline_result.final_state.rays[:, 0, 0],
        q_out,
        omega,
        martinez_k=env["martinez_k"],
        np=np,
    )

    total_map = {
        "all_center": env["combine_phi_total_rad"](
            phi_geom,
            filter_phase,
            phi1,
            phi2_center,
            phi3_center,
            phi4_center,
        ),
        "sample_phi3_only": env["combine_phi_total_rad"](
            phi_geom,
            filter_phase,
            phi1,
            phi2_center,
            phi3_sample,
            phi4_center,
        ),
        "sample_phi2_phi4_only": env["combine_phi_total_rad"](
            phi_geom,
            filter_phase,
            phi1,
            phi2_sample,
            phi3_center,
            phi4_sample,
        ),
        "all_sample": env["combine_phi_total_rad"](
            phi_geom,
            filter_phase,
            phi1,
            phi2_sample,
            phi3_sample,
            phi4_sample,
        ),
    }
    return {
        name: _fit_metrics(env, total_phase=phase, result=result, analytic=analytic)
        for name, phase in total_map.items()
    }


def _surface_tangent_unit(normal_angle_rad: float) -> tuple[float, float]:
    return (math.cos(normal_angle_rad), -math.sin(normal_angle_rad))


def _signed_surface_angle_rad(*, dir_x: Any, dir_z: Any, normal_angle_rad: float, np: Any) -> Any:
    tangent_x, tangent_z = _surface_tangent_unit(normal_angle_rad)
    tangent_projection = (dir_x * tangent_x) + (dir_z * tangent_z)
    return np.arcsin(np.clip(-tangent_projection, -1.0, 1.0))


def _direction_from_surface_angle(
    *,
    normal_angle_rad: float,
    theta_local_rad: Any,
    np: Any,
) -> tuple[Any, Any]:
    global_angle = float(normal_angle_rad) - np.asarray(theta_local_rad, dtype=np.float64)
    return (np.sin(global_angle), np.cos(global_angle))


def _intersect_distance_to_plane(
    *,
    point_x_um: Any,
    point_z_um: Any,
    dir_x: Any,
    dir_z: Any,
    plane: Any,
    np: Any,
) -> Any:
    normal_x, normal_z = plane.normal_unit
    numerator = normal_x * (float(plane.point_x_um) - point_x_um) + normal_z * (
        float(plane.point_z_um) - point_z_um
    )
    denominator = (normal_x * dir_x) + (normal_z * dir_z)
    return numerator / denominator


def _treacy_chief_ray_action_phase(
    env: dict[str, Any],
    *,
    trace: Any,
    include_surface_phase_matching: bool,
) -> Any:
    np = env["np"]
    FreeSpaceCfg = env["FreeSpaceCfg"]
    GratingCfg = env["GratingCfg"]
    FrameTransformCfg = env["FrameTransformCfg"]
    SurfacePlane1D = env["SurfacePlane1D"]

    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    k_arr = env["martinez_k"](omega)

    point_x = np.zeros_like(omega)
    point_z = np.zeros_like(omega)
    dir_x = np.zeros_like(omega)
    dir_z = np.ones_like(omega)
    total_phase = np.zeros_like(omega)

    for item in trace:
        optic_spec = item.optic_spec
        cfg = item.cfg
        if isinstance(optic_spec, FrameTransformCfg):
            if optic_spec.geometry_role == "reflect_chief_ray":
                normal_angle = float(cfg.entrance_surface_normal_angle_rad or 0.0)
                axis_angle = np.arctan2(dir_x, dir_z)
                reflected_axis = (2.0 * normal_angle) - axis_angle + math.pi
                dir_x = np.sin(reflected_axis)
                dir_z = np.cos(reflected_axis)
            continue

        if isinstance(optic_spec, FreeSpaceCfg):
            if (
                cfg.next_surface_point_x_um is None
                or cfg.next_surface_point_z_um is None
                or cfg.next_surface_normal_angle_rad is None
            ):
                raise ValueError("chief-ray action oracle requires surface metadata on free space")
            next_surface = SurfacePlane1D(
                point_x_um=float(cfg.next_surface_point_x_um),
                point_z_um=float(cfg.next_surface_point_z_um),
                normal_angle_rad=float(cfg.next_surface_normal_angle_rad),
            )
            path_length = _intersect_distance_to_plane(
                point_x_um=point_x,
                point_z_um=point_z,
                dir_x=dir_x,
                dir_z=dir_z,
                plane=next_surface,
                np=np,
            )
            total_phase = total_phase + (
                k_arr * float(optic_spec.medium_refractive_index) * np.asarray(path_length)
            )
            point_x = point_x + (path_length * dir_x)
            point_z = point_z + (path_length * dir_z)
            continue

        if isinstance(optic_spec, GratingCfg):
            if (
                cfg.entrance_surface_point_x_um is None
                or cfg.entrance_surface_point_z_um is None
                or cfg.entrance_surface_normal_angle_rad is None
            ):
                raise ValueError("chief-ray action oracle requires grating surface metadata")
            normal_angle = float(cfg.entrance_surface_normal_angle_rad)
            tangent_x, tangent_z = _surface_tangent_unit(normal_angle)
            surface_point_x = float(cfg.entrance_surface_point_x_um)
            surface_point_z = float(cfg.entrance_surface_point_z_um)
            s_hit_um = ((point_x - surface_point_x) * tangent_x) + (
                (point_z - surface_point_z) * tangent_z
            )
            if include_surface_phase_matching:
                theta_in = _signed_surface_angle_rad(
                    dir_x=dir_x,
                    dir_z=dir_z,
                    normal_angle_rad=normal_angle,
                    np=np,
                )
                period_um = 1000.0 / float(optic_spec.line_density_lpmm)
                wavelength_um = (2.0 * np.pi * 0.299792458) / omega
                diff_arg = (
                    -float(optic_spec.diffraction_order) * (wavelength_um / period_um)
                ) - np.sin(theta_in)
                theta_out = np.arcsin(np.clip(diff_arg, -1.0, 1.0))
                total_phase = total_phase + (
                    k_arr * (np.sin(theta_in) + np.sin(theta_out)) * s_hit_um
                )
            else:
                theta_in = _signed_surface_angle_rad(
                    dir_x=dir_x,
                    dir_z=dir_z,
                    normal_angle_rad=normal_angle,
                    np=np,
                )
                period_um = 1000.0 / float(optic_spec.line_density_lpmm)
                wavelength_um = (2.0 * np.pi * 0.299792458) / omega
                diff_arg = (
                    -float(optic_spec.diffraction_order) * (wavelength_um / period_um)
                ) - np.sin(theta_in)
                theta_out = np.arcsin(np.clip(diff_arg, -1.0, 1.0))
            dir_x, dir_z = _direction_from_surface_angle(
                normal_angle_rad=normal_angle,
                theta_local_rad=theta_out,
                np=np,
            )
            continue

    center_idx = int(np.argmin(np.abs(omega - float(np.mean(omega)))))
    output_plane = SurfacePlane1D(
        point_x_um=float(point_x[center_idx]),
        point_z_um=float(point_z[center_idx]),
        normal_angle_rad=float(np.arctan2(dir_x[center_idx], dir_z[center_idx])),
    )
    final_path = _intersect_distance_to_plane(
        point_x_um=point_x,
        point_z_um=point_z,
        dir_x=dir_x,
        dir_z=dir_z,
        plane=output_plane,
        np=np,
    )
    return total_phase + (k_arr * final_path)


def _treacy_chief_ray_action_candidates(
    env: dict[str, Any],
    *,
    params: dict[str, float],
) -> dict[str, dict[str, float]]:
    _, _, result, trace, analytic = _treacy_result_and_trace(env, params=params)
    current_phase = env["combine_phi_total_rad"](
        result.pipeline_result.phi_geom_total_rad,
        _filter_phase_total(result),
        result.pipeline_result.phi1_rad,
        result.pipeline_result.phi2_rad,
        result.pipeline_result.phi3_total_rad,
        result.pipeline_result.phi4_rad,
    )
    chief_path = _treacy_chief_ray_action_phase(
        env,
        trace=trace,
        include_surface_phase_matching=False,
    )
    chief_path_surface = _treacy_chief_ray_action_phase(
        env,
        trace=trace,
        include_surface_phase_matching=True,
    )
    phase_map = {
        "current_runtime": current_phase,
        "chief_ray_path_only": chief_path,
        "chief_ray_path_plus_surface_pm": chief_path_surface,
    }
    return {
        name: _fit_metrics(env, total_phase=phase, result=result, analytic=analytic)
        for name, phase in phase_map.items()
    }


def _section_iv_surface_candidates(env: dict[str, Any]) -> dict[str, dict[str, float]]:
    np = env["np"]
    md = env["md"]

    omega = md._scaled_omega_grid(
        center_wavelength_nm=1030.0,
        n_samples=256,
        pulse_width_fs=100.0,
        span_scale=1.0,
    )
    omega = np.asarray(omega, dtype=np.float64)
    delta_omega = omega - float(np.mean(omega))
    omega0 = float(np.mean(omega))
    line_density_lpmm = 1200.0
    incidence_angle_deg = 35.0
    focal_length_um = 300_000.0
    a = 0.05
    a_prime = 0.0
    b = 0.0

    theta1_rad = math.radians(float(incidence_angle_deg))
    theta2_deg = env["compute_grating_diffraction_angle_deg"](
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        wavelength_nm=1030.0,
        diffraction_order=-1,
    )
    theta2_rad = math.radians(float(theta2_deg))
    telescope = md._section_iv_telescope_matrix(
        omega.size,
        focal_length_um=float(focal_length_um),
        a=float(a),
        a_prime=float(a_prime),
        b=float(b),
    )
    g1_exact = md._section_iv_exact_grating_matrix(
        omega,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
    )
    rays0 = np.zeros((omega.size, 3, 1), dtype=np.float64)
    rays0[:, 2, 0] = 1.0
    rays_after_g1 = g1_exact @ rays0
    rays_before_g2 = telescope @ rays_after_g1

    period_um = 1000.0 / float(line_density_lpmm)
    section = md._section_iv_system(
        omega=omega,
        delta_omega=delta_omega,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        center_wavelength_nm=1030.0,
        focal_length_um=focal_length_um,
        a=a,
        a_prime=a_prime,
        b=b,
    )
    reference_phase = md._section_iv_reference_phase(
        section,
        reference_mode="quadratic_bfg2",
    )
    reference_fit = env["fit_phase_taylor_affine_detrended"](
        delta_omega,
        reference_phase,
        omega0_rad_per_fs=omega0,
        weights=np.ones_like(omega),
        order=4,
    )

    def _surface_candidate(*, sign: float, half: bool) -> dict[str, float]:
        coeff = float(sign) * float(-1) * (2.0 * math.pi / period_um)
        if half:
            coeff *= 0.5
        s1 = np.asarray(rays0[:, 0, 0], dtype=np.float64) / math.cos(theta1_rad)
        s2 = np.asarray(rays_before_g2[:, 0, 0], dtype=np.float64) / math.cos(theta2_rad)
        phase = coeff * (s1 + s2)
        fit = env["fit_phase_taylor_affine_detrended"](
            delta_omega,
            phase,
            omega0_rad_per_fs=omega0,
            weights=np.ones_like(omega),
            order=4,
        )
        residual = md._remove_affine_phase(
            delta_omega,
            phase - reference_phase,
            weights=np.ones_like(omega),
        )
        return {
            "gdd_abs_error_fs2": abs(
                float(fit.coefficients_rad[2]) - float(reference_fit.coefficients_rad[2])
            ),
            "tod_abs_error_fs3": abs(
                float(fit.coefficients_rad[3]) - float(reference_fit.coefficients_rad[3])
            ),
            "weighted_rms_rad": float(md._weighted_rms(residual, np.ones_like(omega))),
            "phase_gdd_fs2": float(fit.coefficients_rad[2]),
            "phase_tod_fs3": float(fit.coefficients_rad[3]),
        }

    return {
        "mask_mK": _surface_candidate(sign=1.0, half=False),
        "mask_half_mK": _surface_candidate(sign=1.0, half=True),
        "mask_negmK": _surface_candidate(sign=-1.0, half=False),
        "mask_half_negmK": _surface_candidate(sign=-1.0, half=True),
    }


def _section_iv_second_grating_audit(env: dict[str, Any]) -> dict[str, float]:
    np = env["np"]
    md = env["md"]

    omega = md._scaled_omega_grid(
        center_wavelength_nm=1030.0,
        n_samples=256,
        pulse_width_fs=100.0,
        span_scale=1.0,
    )
    omega = np.asarray(omega, dtype=np.float64)
    omega0 = float(np.mean(omega))
    line_density_lpmm = 1200.0
    incidence_angle_deg = 35.0
    focal_length_um = 300_000.0
    a = 0.05
    a_prime = 0.0
    b = 0.0
    period_um = 1000.0 / float(line_density_lpmm)
    theta1_rad = math.radians(float(incidence_angle_deg))
    theta2_deg = env["compute_grating_diffraction_angle_deg"](
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        wavelength_nm=1030.0,
        diffraction_order=-1,
    )
    theta2_rad = math.radians(float(theta2_deg))
    magnification = math.cos(theta2_rad) / math.cos(theta1_rad)
    telescope = md._section_iv_telescope_matrix(
        omega.size,
        focal_length_um=float(focal_length_um),
        a=float(a),
        a_prime=float(a_prime),
        b=float(b),
    )
    g1_exact = md._section_iv_exact_grating_matrix(
        omega,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
    )
    g2_exact = md._section_iv_exact_grating_matrix(
        omega,
        omega0=omega0,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=float(theta2_deg),
    )
    rays0 = np.zeros((omega.size, 3, 1), dtype=np.float64)
    rays0[:, 2, 0] = 1.0
    rays_after_g1 = g1_exact @ rays0
    rays_before_g2 = telescope @ rays_after_g1
    rays_after_g2 = g2_exact @ rays_before_g2

    theta3_rad = env["_diffraction_angle_rad"](
        omega,
        period_um=period_um,
        incidence_angle_rad=theta2_rad,
        diffraction_order=-1.0,
    )
    s2_pre = np.asarray(rays_before_g2[:, 0, 0], dtype=np.float64) / math.cos(theta2_rad)
    s2_post = np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64) / np.cos(theta3_rad)

    fg1_exact = md._exact_local_grating_f(
        omega,
        omega0=omega0,
        period_um=period_um,
        incidence_angle_rad=theta1_rad,
    )
    f2_exact = np.asarray(g2_exact[:, 1, 2], dtype=np.float64)
    paired_f2 = float(magnification) * np.asarray(fg1_exact, dtype=np.float64)
    expected_b = -float(focal_length_um) * (float(a) + float(a_prime)) / (magnification**2)
    paired_x_after = expected_b * np.asarray(fg1_exact, dtype=np.float64) / float(magnification)
    exact_x_after = np.asarray(rays_after_g2[:, 0, 0], dtype=np.float64)
    exact_x_before = np.asarray(rays_before_g2[:, 0, 0], dtype=np.float64)

    return {
        "surface_coordinate_max_residual_um": float(np.max(np.abs(s2_pre - s2_post))),
        "surface_coordinate_rms_residual_um": float(np.sqrt(np.mean((s2_pre - s2_post) ** 2))),
        "f2_exact_vs_mf1_rms": float(np.sqrt(np.mean((f2_exact - paired_f2) ** 2))),
        "f2_exact_vs_mf1_rel_rms": float(
            np.sqrt(np.mean((f2_exact - paired_f2) ** 2)) / np.sqrt(np.mean(paired_f2**2))
        ),
        "x_after_exact_vs_paired_rms_um": float(
            np.sqrt(np.mean((exact_x_after - paired_x_after) ** 2))
        ),
        "x_after_exact_vs_paired_rel_rms": float(
            np.sqrt(np.mean((exact_x_after - paired_x_after) ** 2))
            / np.sqrt(np.mean(paired_x_after**2))
        ),
        "x_before_g2_rms_um": float(np.sqrt(np.mean(exact_x_before**2))),
        "x_after_g2_rms_um": float(np.sqrt(np.mean(exact_x_after**2))),
    }


def _plot_k_consistency(
    env: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    plt = env["plt"]
    np = env["np"]

    variants = ("all_center", "sample_phi3_only", "sample_phi2_phi4_only", "all_sample")
    labels = ("center", "sample phi3", "sample phi2/4", "all sample")
    scenario_names = tuple(metrics.keys())
    x = np.arange(len(scenario_names))
    width = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for idx, (variant, label) in enumerate(zip(variants, labels, strict=True)):
        gdd_vals = [metrics[name][variant]["gdd_rel_error"] for name in scenario_names]
        tod_vals = [metrics[name][variant]["tod_rel_error"] for name in scenario_names]
        axes[0].bar(x + ((idx - 1.5) * width), gdd_vals, width=width, label=label)
        axes[1].bar(x + ((idx - 1.5) * width), tod_vals, width=width, label=label)
    axes[0].set_title("Treacy GDD Error")
    axes[1].set_title("Treacy TOD Error")
    axes[0].set_ylabel("Relative Error")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_yscale("log")
    axes[0].set_xticks(x)
    axes[1].set_xticks(x)
    axes[0].set_xticklabels(("35/0", "30/0", "35/100", "1000lp"))
    axes[1].set_xticklabels(("35/0", "30/0", "35/100", "1000lp"))
    axes[0].legend(loc="upper left", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_chief_ray_candidates(
    env: dict[str, Any],
    metrics: dict[str, dict[str, dict[str, float]]],
    output_path: Path,
) -> None:
    plt = env["plt"]
    np = env["np"]

    variants = ("current_runtime", "chief_ray_path_only", "chief_ray_path_plus_surface_pm")
    labels = ("runtime", "chief path", "chief path + surface")
    scenario_names = tuple(metrics.keys())
    x = np.arange(len(scenario_names))
    width = 0.24

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for idx, (variant, label) in enumerate(zip(variants, labels, strict=True)):
        gdd_vals = [metrics[name][variant]["gdd_rel_error"] for name in scenario_names]
        tod_vals = [metrics[name][variant]["tod_rel_error"] for name in scenario_names]
        axes[0].bar(x + ((idx - 1.0) * width), gdd_vals, width=width, label=label)
        axes[1].bar(x + ((idx - 1.0) * width), tod_vals, width=width, label=label)
    axes[0].set_title("Chief-Ray Action Candidates: GDD")
    axes[1].set_title("Chief-Ray Action Candidates: TOD")
    axes[0].set_ylabel("Relative Error")
    axes[1].set_ylabel("Relative Error")
    axes[1].set_yscale("log")
    axes[0].set_xticks(x)
    axes[1].set_xticks(x)
    axes[0].set_xticklabels(("35/0", "30/0", "35/100", "1000lp"))
    axes[1].set_xticklabels(("35/0", "30/0", "35/100", "1000lp"))
    axes[0].legend(loc="upper left", fontsize=8)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_section_iv_surface_candidates(
    env: dict[str, Any],
    metrics: dict[str, dict[str, float]],
    output_path: Path,
) -> None:
    plt = env["plt"]
    np = env["np"]
    names = tuple(metrics.keys())
    rms = [metrics[name]["weighted_rms_rad"] for name in names]
    gdd = [metrics[name]["gdd_abs_error_fs2"] for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].bar(np.arange(len(names)), rms)
    axes[1].bar(np.arange(len(names)), gdd)
    axes[0].set_xticks(np.arange(len(names)))
    axes[1].set_xticks(np.arange(len(names)))
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_title("Section IV Surface Candidates")
    axes[1].set_title("Section IV Surface Candidates")
    axes[0].set_ylabel("Weighted RMS (rad)")
    axes[1].set_ylabel("|GDD error| (fs^2)")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate model-limit diagnostics for the remaining grating-phase residual. "
            "Outputs JSON plus comparison plots for k-consistency, chief-ray action "
            "candidates, and Section IV surface candidates."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "physics" / "model_limit_diagnostics",
        help="Directory for JSON and PNG outputs.",
    )
    args = parser.parse_args()
    paths = generate_artifacts(args.output_dir)
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
