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


LAMBDA_MIN = -0.08
LAMBDA_MAX = 0.08
LAMBDA_SAMPLES = 321

CURRENT_PHI3_VARIANT = "martinez_series2_centerk"
PRIMARY_BASES: tuple[str, ...] = ("f_bpre", "epre")
ALL_BASES: tuple[str, ...] = (
    "bpre",
    "epre",
    "eout",
    "f_bpre",
    "f_epre",
    "f_eout",
)

BEAM_RADIUS_SWEEP_MM: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0)
ANGLE_SWEEP_DEG: tuple[float, ...] = (25.0, 30.0, 35.0, 37.0, 38.0, 39.0, 40.0, 45.0)
LINE_DENSITY_SWEEP_LPMM: tuple[float, ...] = (600.0, 750.0, 900.0, 1050.0, 1200.0, 1350.0, 1500.0)
SEPARATION_SWEEP_UM: tuple[float, ...] = (25_000.0, 50_000.0, 100_000.0, 200_000.0, 400_000.0)
MIRROR_SWEEP_UM: tuple[float, ...] = (0.0, 25_000.0, 50_000.0, 75_000.0, 100_000.0)

SECTION_IV_A_SWEEP: tuple[float, ...] = (0.01, 0.03, 0.05, 0.07, 0.1)
SECTION_IV_APRIME_SWEEP: tuple[float, ...] = (-0.04, -0.02, 0.0, 0.02, 0.04)
SECTION_IV_B_SWEEP: tuple[float, ...] = (-0.05, -0.025, 0.0, 0.025, 0.05)

GEOMETRY_GROUPS: tuple[tuple[str, str], ...] = (
    ("one_minus_m", "1 - M"),
    ("m_minus_one", "M - 1"),
    ("m_minus_inv_m", "M - 1/M"),
    ("tan_diff", "tan(td) - tan(ti)"),
    ("tan_sum", "tan(td) + tan(ti)"),
)


def generate_artifacts(output_dir: Path) -> tuple[Path, ...]:
    env = _load_env()
    np = env["np"]

    output_dir.mkdir(parents=True, exist_ok=True)

    radius_points = _run_treacy_radius_sweep(env)
    baseline_points = _run_treacy_baseline_basis_scan(env)
    geometry_points = _run_treacy_geometry_sweeps(env)
    section_iv_points = _run_section_iv_family_sweeps(env)
    collapse_points, collapse_scores = _run_geometry_group_collapse(env, geometry_points)

    payload = {
        "plot_description": (
            "Inverse-physics sweeps for the Treacy / Martinez grating-boundary residual. "
            "All fits keep the current runtime phi3 term intact and scan one extra boundary "
            "basis coefficient over a bounded lambda grid."
        ),
        "lambda_grid": np.linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_SAMPLES).tolist(),
        "current_phi3_variant": CURRENT_PHI3_VARIANT,
        "all_bases": list(ALL_BASES),
        "primary_bases": list(PRIMARY_BASES),
        "radius_points": radius_points,
        "baseline_points": baseline_points,
        "geometry_points": geometry_points,
        "section_iv_points": section_iv_points,
        "collapse_points": collapse_points,
        "collapse_scores": collapse_scores,
    }

    json_path = output_dir / "boundary_probe_sweeps.json"
    radius_png = output_dir / "treacy_boundary_basis_vs_radius.png"
    baseline_png = output_dir / "treacy_boundary_basis_baseline.png"
    geometry_png = output_dir / "treacy_boundary_basis_geometry_sweeps.png"
    section_iv_png = output_dir / "section_iv_boundary_basis_sweeps.png"
    collapse_png = output_dir / "boundary_basis_magnification_collapse.png"

    json_path.write_text(json.dumps(payload, indent=2))
    _plot_radius_sweep(env, radius_points, radius_png)
    _plot_baseline_basis_scan(env, baseline_points, baseline_png)
    _plot_geometry_sweeps(env, geometry_points, baseline_points, geometry_png)
    _plot_section_iv_sweeps(env, section_iv_points, section_iv_png)
    _plot_collapse(env, collapse_points, collapse_scores, geometry_points, collapse_png)
    return (json_path, radius_png, baseline_png, geometry_png, section_iv_png, collapse_png)


def _load_env() -> dict[str, Any]:
    import matplotlib.pyplot as plt
    import numpy as np

    from abcdef_sim import (
        BeamSpec,
        PulseSpec,
        StandaloneLaserSpec,
        run_abcdef,
        treacy_compressor_preset,
    )
    from abcdef_sim.analytics import martinez_debug as md
    from abcdef_sim.physics.abcdef.dispersion import fit_phase_taylor_affine_detrended
    from abcdef_sim.physics.abcdef.phase_terms import combine_phi_total_rad, martinez_k
    from abcdef_sim.physics.abcdef.treacy import (
        compute_grating_diffraction_angle_deg,
        compute_treacy_analytic_metrics,
        phase_from_treacy_dispersion,
    )

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
        "phase_from_treacy_dispersion": phase_from_treacy_dispersion,
        "md": md,
    }


def _lambda_grid(env: dict[str, Any]) -> Any:
    return env["np"].linspace(LAMBDA_MIN, LAMBDA_MAX, LAMBDA_SAMPLES)


def _phase_metrics(
    env: dict[str, Any],
    *,
    delta_omega: Any,
    phase: Any,
    omega0: float,
    weights: Any,
    reference_phase: Any,
    reference_gdd_fs2: float,
    reference_tod_fs3: float,
) -> dict[str, float]:
    md = env["md"]
    fit = env["fit_phase_taylor_affine_detrended"](
        delta_omega,
        phase,
        omega0_rad_per_fs=omega0,
        weights=weights,
        order=4,
    )
    gdd = float(fit.coefficients_rad[2])
    tod = float(fit.coefficients_rad[3])
    residual = md._remove_affine_phase(
        delta_omega,
        phase - reference_phase,
        weights=weights,
    )
    rms = float(md._weighted_rms(residual, weights))
    gdd_rel = float(md._relative_error(gdd, reference_gdd_fs2))
    tod_rel = float(md._relative_error(tod, reference_tod_fs3))
    return {
        "gdd_fs2": gdd,
        "tod_fs3": tod,
        "gdd_rel_error": gdd_rel,
        "tod_rel_error": tod_rel,
        "weighted_rms_rad": rms,
        "score": gdd_rel + tod_rel,
    }


def _fit_best_lambda(
    env: dict[str, Any],
    *,
    base_phase: Any,
    basis_phase: Any,
    delta_omega: Any,
    omega0: float,
    weights: Any,
    reference_phase: Any,
    reference_gdd_fs2: float,
    reference_tod_fs3: float,
) -> dict[str, float]:
    np = env["np"]
    basis_arr = np.asarray(basis_phase, dtype=np.float64)
    base_arr = np.asarray(base_phase, dtype=np.float64)

    baseline = _phase_metrics(
        env,
        delta_omega=delta_omega,
        phase=base_arr,
        omega0=omega0,
        weights=weights,
        reference_phase=reference_phase,
        reference_gdd_fs2=reference_gdd_fs2,
        reference_tod_fs3=reference_tod_fs3,
    )
    if np.allclose(basis_arr, 0.0, rtol=0.0, atol=1e-18):
        baseline["best_lambda"] = 0.0
        baseline["hit_scan_edge"] = False
        baseline["improvement_factor"] = 1.0
        return baseline

    best_metrics: dict[str, float] | None = None
    for lambda_value in _lambda_grid(env):
        metrics = _phase_metrics(
            env,
            delta_omega=delta_omega,
            phase=base_arr + float(lambda_value) * basis_arr,
            omega0=omega0,
            weights=weights,
            reference_phase=reference_phase,
            reference_gdd_fs2=reference_gdd_fs2,
            reference_tod_fs3=reference_tod_fs3,
        )
        metrics["best_lambda"] = float(lambda_value)
        if best_metrics is None or (
            metrics["score"],
            metrics["weighted_rms_rad"],
        ) < (
            best_metrics["score"],
            best_metrics["weighted_rms_rad"],
        ):
            best_metrics = metrics
    if best_metrics is None:
        raise RuntimeError("lambda scan produced no candidate metrics")
    best_metrics["hit_scan_edge"] = math.isclose(
        best_metrics["best_lambda"],
        LAMBDA_MIN,
        rel_tol=0.0,
        abs_tol=1e-12,
    ) or math.isclose(
        best_metrics["best_lambda"],
        LAMBDA_MAX,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    baseline_score = baseline["score"]
    best_metrics["improvement_factor"] = (
        1.0 if baseline_score <= 0.0 else baseline_score / best_metrics["score"]
    )
    return best_metrics


def _basis_term(
    env: dict[str, Any],
    *,
    basis_name: str,
    omega: Any,
    f_phase: Any,
    b_pre: Any,
    e_pre: Any,
    e_out: Any,
) -> Any:
    np = env["np"]
    omega_arr = np.asarray(omega, dtype=np.float64)
    k_arr = env["martinez_k"](omega_arr)
    f_arr = np.asarray(f_phase, dtype=np.float64)
    b_pre_arr = np.asarray(b_pre, dtype=np.float64)
    e_pre_arr = np.asarray(e_pre, dtype=np.float64)
    e_out_arr = np.asarray(e_out, dtype=np.float64)

    if basis_name == "bpre":
        return 0.5 * k_arr * b_pre_arr
    if basis_name == "epre":
        return 0.5 * k_arr * e_pre_arr
    if basis_name == "eout":
        return 0.5 * k_arr * e_out_arr
    if basis_name == "f_bpre":
        return 0.5 * k_arr * f_arr * b_pre_arr
    if basis_name == "f_epre":
        return 0.5 * k_arr * f_arr * e_pre_arr
    if basis_name == "f_eout":
        return 0.5 * k_arr * f_arr * e_out_arr
    raise KeyError(f"Unknown basis {basis_name!r}")


def _treacy_geometry_groups(
    env: dict[str, Any],
    *,
    line_density_lpmm: float,
    incidence_angle_deg: float,
    center_wavelength_nm: float,
    diffraction_order: int,
) -> dict[str, float]:
    theta_i_rad = math.radians(float(incidence_angle_deg))
    theta_d_deg = env["compute_grating_diffraction_angle_deg"](
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        wavelength_nm=float(center_wavelength_nm),
        diffraction_order=int(diffraction_order),
    )
    theta_d_rad = math.radians(float(theta_d_deg))
    magnification = math.cos(theta_d_rad) / math.cos(theta_i_rad)
    return {
        "theta_i_deg": float(incidence_angle_deg),
        "theta_d_deg": float(theta_d_deg),
        "magnification": float(magnification),
        "one_minus_m": float(1.0 - magnification),
        "m_minus_one": float(magnification - 1.0),
        "m_minus_inv_m": float(magnification - (1.0 / magnification)),
        "tan_diff": float(math.tan(theta_d_rad) - math.tan(theta_i_rad)),
        "tan_sum": float(math.tan(theta_d_rad) + math.tan(theta_i_rad)),
    }


def _build_treacy_case(
    env: dict[str, Any],
    *,
    beam_radius_mm: float,
    line_density_lpmm: float,
    incidence_angle_deg: float,
    separation_um: float,
    length_to_mirror_um: float,
    diffraction_order: int = -1,
    n_passes: int = 2,
    base_pulse_width_fs: float = 100.0,
    center_wavelength_nm: float = 1030.0,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> dict[str, Any]:
    md = env["md"]
    laser = env["StandaloneLaserSpec"](
        pulse=env["PulseSpec"](
            width_fs=float(base_pulse_width_fs),
            center_wavelength_nm=float(center_wavelength_nm),
            n_samples=int(n_samples),
            time_window_fs=float(time_window_fs),
        ),
        beam=env["BeamSpec"](radius_mm=float(beam_radius_mm), m2=1.0),
    )
    cfg = env["treacy_compressor_preset"](
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        separation_um=float(separation_um),
        length_to_mirror_um=float(length_to_mirror_um),
        diffraction_order=int(diffraction_order),
        n_passes=int(n_passes),
    )
    result = env["run_abcdef"](cfg, laser)
    trace = md._trace_cfg_run(cfg, laser)
    analytic = env["compute_treacy_analytic_metrics"](
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        separation_um=float(separation_um),
        wavelength_nm=float(center_wavelength_nm),
        diffraction_order=int(diffraction_order),
        n_passes=int(n_passes),
    )
    delta_omega = env["np"].asarray(result.pipeline_result.delta_omega_rad_per_fs, dtype=env["np"].float64)
    omega0 = float(result.pipeline_result.omega0_rad_per_fs)
    weights = env["np"].asarray(result.fit.weights, dtype=env["np"].float64)
    reference_phase = env["phase_from_treacy_dispersion"](
        delta_omega,
        gdd_fs2=float(analytic.gdd_fs2),
        tod_fs3=float(analytic.tod_fs3),
    )
    base_terms = md._treacy_non_phi3_phase_terms(result)
    current_variant = md._phi3_variant_config_by_name(CURRENT_PHI3_VARIANT)
    current_phi3 = md._phi3_variant_payload_from_trace(trace, variant=current_variant)["phi3_total"]
    current_total = env["combine_phi_total_rad"](
        base_terms["phi_geom"],
        base_terms["filter_phase"],
        base_terms["phi1"],
        base_terms["phi2"],
        current_phi3,
        base_terms["phi4"],
    )
    return {
        "trace": trace,
        "delta_omega": delta_omega,
        "omega0": omega0,
        "weights": weights,
        "reference_phase": reference_phase,
        "reference_gdd_fs2": float(analytic.gdd_fs2),
        "reference_tod_fs3": float(analytic.tod_fs3),
        "base_phase": current_total,
        "current_variant": current_variant,
        "geometry": _treacy_geometry_groups(
            env,
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            center_wavelength_nm=center_wavelength_nm,
            diffraction_order=diffraction_order,
        ),
    }


def _treacy_basis_series(
    env: dict[str, Any],
    *,
    trace: Any,
    basis_name: str,
    current_variant: Any,
) -> Any:
    md = env["md"]
    np = env["np"]
    omega = np.asarray(trace[0].cfg.omega, dtype=np.float64).reshape(-1)
    total = np.zeros_like(omega)
    for item in md._grating_stage_traces(trace):
        if not md._has_nonunit_upstream_magnification(item):
            continue
        series = md._phase_series_for_trace_item(item)
        f_phase = md._select_f_phase(series, variant=current_variant)
        b_pre = np.asarray(item.state_in.system[:, 0, 1], dtype=np.float64)
        e_pre = np.asarray(item.state_in.system[:, 0, 2], dtype=np.float64)
        e_out = np.asarray(item.state_out.system[:, 0, 2], dtype=np.float64)
        total = total + _basis_term(
            env,
            basis_name=basis_name,
            omega=omega,
            f_phase=f_phase,
            b_pre=b_pre,
            e_pre=e_pre,
            e_out=e_out,
        )
    return total


def _run_treacy_basis_scan(
    env: dict[str, Any],
    *,
    beam_radius_mm: float,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    basis_names: tuple[str, ...] = ALL_BASES,
) -> list[dict[str, float | str | bool]]:
    case = _build_treacy_case(
        env,
        beam_radius_mm=beam_radius_mm,
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        length_to_mirror_um=length_to_mirror_um,
    )
    rows: list[dict[str, float | str | bool]] = []
    for basis_name in basis_names:
        basis = _treacy_basis_series(
            env,
            trace=case["trace"],
            basis_name=basis_name,
            current_variant=case["current_variant"],
        )
        fit = _fit_best_lambda(
            env,
            base_phase=case["base_phase"],
            basis_phase=basis,
            delta_omega=case["delta_omega"],
            omega0=case["omega0"],
            weights=case["weights"],
            reference_phase=case["reference_phase"],
            reference_gdd_fs2=case["reference_gdd_fs2"],
            reference_tod_fs3=case["reference_tod_fs3"],
        )
        row: dict[str, float | str | bool] = {
            "beam_radius_mm": float(beam_radius_mm),
            "basis": basis_name,
            "line_density_lpmm": float(line_density_lpmm),
            "incidence_angle_deg": float(incidence_angle_deg),
            "separation_um": float(separation_um),
            "length_to_mirror_um": float(length_to_mirror_um),
            "reference_gdd_fs2": float(case["reference_gdd_fs2"]),
            "reference_tod_fs3": float(case["reference_tod_fs3"]),
        }
        row.update(case["geometry"])
        row.update(fit)
        rows.append(row)
    return rows


def _run_treacy_radius_sweep(env: dict[str, Any]) -> list[dict[str, float | str | bool]]:
    points: list[dict[str, float | str | bool]] = []
    for beam_radius_mm in BEAM_RADIUS_SWEEP_MM:
        points.extend(
            _run_treacy_basis_scan(
                env,
                beam_radius_mm=float(beam_radius_mm),
                basis_names=PRIMARY_BASES,
            )
        )
    return points


def _run_treacy_baseline_basis_scan(env: dict[str, Any]) -> list[dict[str, float | str | bool]]:
    return _run_treacy_basis_scan(env, beam_radius_mm=1000.0, basis_names=ALL_BASES)


def _run_treacy_geometry_sweeps(env: dict[str, Any]) -> list[dict[str, float | str | bool]]:
    points: list[dict[str, float | str | bool]] = []

    for incidence_angle_deg in ANGLE_SWEEP_DEG:
        try:
            rows = _run_treacy_basis_scan(
                env,
                beam_radius_mm=1000.0,
                incidence_angle_deg=float(incidence_angle_deg),
                basis_names=PRIMARY_BASES,
            )
            for row in rows:
                row["sweep"] = "angle"
                row["sweep_value"] = float(incidence_angle_deg)
            points.extend(rows)
        except ValueError:
            continue

    for line_density_lpmm in LINE_DENSITY_SWEEP_LPMM:
        try:
            rows = _run_treacy_basis_scan(
                env,
                beam_radius_mm=1000.0,
                line_density_lpmm=float(line_density_lpmm),
                basis_names=PRIMARY_BASES,
            )
            for row in rows:
                row["sweep"] = "density"
                row["sweep_value"] = float(line_density_lpmm)
            points.extend(rows)
        except ValueError:
            continue

    for separation_um in SEPARATION_SWEEP_UM:
        rows = _run_treacy_basis_scan(
            env,
            beam_radius_mm=1000.0,
            separation_um=float(separation_um),
            basis_names=PRIMARY_BASES,
        )
        for row in rows:
            row["sweep"] = "separation"
            row["sweep_value"] = float(separation_um)
        points.extend(rows)

    for length_to_mirror_um in MIRROR_SWEEP_UM:
        rows = _run_treacy_basis_scan(
            env,
            beam_radius_mm=1000.0,
            length_to_mirror_um=float(length_to_mirror_um),
            basis_names=PRIMARY_BASES,
        )
        for row in rows:
            row["sweep"] = "mirror"
            row["sweep_value"] = float(length_to_mirror_um)
        points.extend(rows)

    return points


def _build_section_iv_case(
    env: dict[str, Any],
    *,
    a: float,
    a_prime: float,
    b: float,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    center_wavelength_nm: float = 1030.0,
    focal_length_um: float = 300_000.0,
    n_samples: int = 256,
    pulse_width_fs: float = 100.0,
) -> dict[str, Any]:
    md = env["md"]
    np = env["np"]
    omega = md._scaled_omega_grid(
        center_wavelength_nm=float(center_wavelength_nm),
        n_samples=int(n_samples),
        pulse_width_fs=float(pulse_width_fs),
        span_scale=1.0,
    )
    delta_omega = np.asarray(omega - float(np.mean(omega)), dtype=np.float64)
    section = md._section_iv_system(
        omega=omega,
        delta_omega=delta_omega,
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        center_wavelength_nm=float(center_wavelength_nm),
        focal_length_um=float(focal_length_um),
        a=float(a),
        a_prime=float(a_prime),
        b=float(b),
    )
    weights = np.ones_like(omega, dtype=np.float64)
    reference_phase = np.asarray(section.reference_phase_rad, dtype=np.float64)
    reference_fit = env["fit_phase_taylor_affine_detrended"](
        delta_omega,
        reference_phase,
        omega0_rad_per_fs=float(section.omega0_rad_per_fs),
        weights=weights,
        order=4,
    )
    current_variant = md._phi3_variant_config_by_name(CURRENT_PHI3_VARIANT)
    base_phase = md._section_iv_phi3_variant_phase(
        omega=np.asarray(omega, dtype=np.float64),
        delta_omega=np.asarray(delta_omega, dtype=np.float64),
        variant=current_variant,
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        center_wavelength_nm=float(center_wavelength_nm),
        focal_length_um=float(focal_length_um),
        a=float(a),
        a_prime=float(a_prime),
        b=float(b),
    )

    theta_i_rad = math.radians(float(incidence_angle_deg))
    theta_d_deg = env["compute_grating_diffraction_angle_deg"](
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
        wavelength_nm=float(center_wavelength_nm),
        diffraction_order=-1,
    )
    theta_d_rad = math.radians(float(theta_d_deg))
    telescope = md._section_iv_telescope_matrix(
        np.asarray(omega, dtype=np.float64).size,
        focal_length_um=float(focal_length_um),
        a=float(a),
        a_prime=float(a_prime),
        b=float(b),
    )
    g1_exact = md._section_iv_exact_grating_matrix(
        np.asarray(omega, dtype=np.float64),
        omega0=float(section.omega0_rad_per_fs),
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(incidence_angle_deg),
    )
    g2_exact = md._section_iv_exact_grating_matrix(
        np.asarray(omega, dtype=np.float64),
        omega0=float(section.omega0_rad_per_fs),
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_deg=float(theta_d_deg),
    )
    system_pre_g2 = telescope @ g1_exact
    system_out_g2 = g2_exact @ system_pre_g2
    g2_series = md._section_iv_phase_series(
        np.asarray(omega, dtype=np.float64),
        np.asarray(delta_omega, dtype=np.float64),
        omega0=float(section.omega0_rad_per_fs),
        line_density_lpmm=float(line_density_lpmm),
        incidence_angle_rad=float(theta_d_rad),
    )
    f_phase = md._select_f_phase(g2_series, variant=current_variant)
    geometry = {
        "theta_i_deg": float(incidence_angle_deg),
        "theta_d_deg": float(theta_d_deg),
        "magnification": float(math.cos(theta_d_rad) / math.cos(theta_i_rad)),
    }
    return {
        "base_phase": np.asarray(base_phase, dtype=np.float64),
        "delta_omega": np.asarray(delta_omega, dtype=np.float64),
        "omega0": float(section.omega0_rad_per_fs),
        "weights": weights,
        "reference_phase": reference_phase,
        "reference_gdd_fs2": float(reference_fit.coefficients_rad[2]),
        "reference_tod_fs3": float(reference_fit.coefficients_rad[3]),
        "b_pre": np.asarray(system_pre_g2[:, 0, 1], dtype=np.float64),
        "e_pre": np.asarray(system_pre_g2[:, 0, 2], dtype=np.float64),
        "e_out": np.asarray(system_out_g2[:, 0, 2], dtype=np.float64),
        "f_phase": np.asarray(f_phase, dtype=np.float64),
        "omega": np.asarray(omega, dtype=np.float64),
        "geometry": geometry,
    }


def _run_section_iv_basis_scan(
    env: dict[str, Any],
    *,
    a: float,
    a_prime: float,
    b: float,
    basis_names: tuple[str, ...] = PRIMARY_BASES,
) -> list[dict[str, float | str | bool]]:
    case = _build_section_iv_case(env, a=a, a_prime=a_prime, b=b)
    rows: list[dict[str, float | str | bool]] = []
    for basis_name in basis_names:
        basis = _basis_term(
            env,
            basis_name=basis_name,
            omega=case["omega"],
            f_phase=case["f_phase"],
            b_pre=case["b_pre"],
            e_pre=case["e_pre"],
            e_out=case["e_out"],
        )
        fit = _fit_best_lambda(
            env,
            base_phase=case["base_phase"],
            basis_phase=basis,
            delta_omega=case["delta_omega"],
            omega0=case["omega0"],
            weights=case["weights"],
            reference_phase=case["reference_phase"],
            reference_gdd_fs2=case["reference_gdd_fs2"],
            reference_tod_fs3=case["reference_tod_fs3"],
        )
        row: dict[str, float | str | bool] = {
            "family": "section_iv",
            "basis": basis_name,
            "a": float(a),
            "a_prime": float(a_prime),
            "b": float(b),
        }
        row.update(case["geometry"])
        row.update(fit)
        rows.append(row)
    return rows


def _run_section_iv_family_sweeps(env: dict[str, Any]) -> list[dict[str, float | str | bool]]:
    points: list[dict[str, float | str | bool]] = []

    for a in SECTION_IV_A_SWEEP:
        rows = _run_section_iv_basis_scan(env, a=float(a), a_prime=0.0, b=0.0)
        for row in rows:
            row["sweep"] = "a"
            row["sweep_value"] = float(a)
        points.extend(rows)

    for a_prime in SECTION_IV_APRIME_SWEEP:
        rows = _run_section_iv_basis_scan(env, a=0.05, a_prime=float(a_prime), b=0.0)
        for row in rows:
            row["sweep"] = "a_prime"
            row["sweep_value"] = float(a_prime)
        points.extend(rows)

    for b in SECTION_IV_B_SWEEP:
        rows = _run_section_iv_basis_scan(env, a=0.05, a_prime=0.0, b=float(b))
        for row in rows:
            row["sweep"] = "b"
            row["sweep_value"] = float(b)
        points.extend(rows)

    return points


def _run_geometry_group_collapse(
    _env: dict[str, Any],
    geometry_points: list[dict[str, float | str | bool]],
) -> tuple[list[dict[str, float | str | bool]], list[dict[str, float | str]]]:
    relevant = [
        point
        for point in geometry_points
        if point["sweep"] in {"angle", "density"}
        and point["basis"] in PRIMARY_BASES
    ]
    collapse_scores: list[dict[str, float | str]] = []
    for basis_name in PRIMARY_BASES:
        basis_points = [point for point in relevant if point["basis"] == basis_name]
        for key, label in GEOMETRY_GROUPS:
            normalized: list[float] = []
            for point in basis_points:
                group_value = float(point[key])
                if abs(group_value) < 1e-6:
                    continue
                normalized.append(float(point["best_lambda"]) / group_value)
            if not normalized:
                continue
            mean_abs = sum(abs(value) for value in normalized) / len(normalized)
            variance = sum((value - (sum(normalized) / len(normalized))) ** 2 for value in normalized)
            spread = math.sqrt(variance / len(normalized))
            collapse_scores.append(
                {
                    "basis": basis_name,
                    "group_key": key,
                    "group_label": label,
                    "coefficient_of_variation": float(spread / mean_abs)
                    if mean_abs > 0.0
                    else 1.0e12,
                }
            )
    return relevant, collapse_scores


def _plot_radius_sweep(
    env: dict[str, Any],
    points: list[dict[str, float | str | bool]],
    output_path: Path,
) -> None:
    plt = env["plt"]
    np = env["np"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, constrained_layout=True)
    for basis_name, color in (("f_bpre", "tab:green"), ("epre", "tab:orange")):
        basis_points = sorted(
            (point for point in points if point["basis"] == basis_name),
            key=lambda point: float(point["beam_radius_mm"]),
        )
        beam_radii = np.array([point["beam_radius_mm"] for point in basis_points], dtype=np.float64)
        best_lambda = np.array([point["best_lambda"] for point in basis_points], dtype=np.float64)
        scores = np.array([point["score"] for point in basis_points], dtype=np.float64)
        axes[0].plot(beam_radii, best_lambda, marker="o", color=color, label=basis_name)
        axes[1].plot(beam_radii, scores, marker="o", color=color, label=basis_name)

    axes[0].set_ylabel("Best fitted lambda")
    axes[0].set_title("Large-beam regime check")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Beam radius (mm)")
    axes[1].set_ylabel("Best score (GDD rel. + TOD rel.)")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Treacy Boundary-Basis Radius Sweep")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_baseline_basis_scan(
    env: dict[str, Any],
    points: list[dict[str, float | str | bool]],
    output_path: Path,
) -> None:
    plt = env["plt"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)
    ordered = sorted(points, key=lambda point: float(point["score"]))
    labels = [str(point["basis"]) for point in ordered]
    scores = [float(point["score"]) for point in ordered]
    lambdas = [float(point["best_lambda"]) for point in ordered]

    axes[0].bar(labels, scores, color="tab:blue")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Best score")
    axes[0].set_title("Baseline Treacy case")
    axes[0].tick_params(axis="x", rotation=35)
    axes[0].grid(True, which="both", axis="y", alpha=0.3)

    axes[1].bar(labels, lambdas, color="tab:red")
    axes[1].set_ylabel("Best lambda")
    axes[1].set_title("Fitted coefficient by basis")
    axes[1].tick_params(axis="x", rotation=35)
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("Treacy Boundary-Basis Comparison at 1000 mm Radius")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_geometry_sweeps(
    env: dict[str, Any],
    points: list[dict[str, float | str | bool]],
    baseline_points: list[dict[str, float | str | bool]],
    output_path: Path,
) -> None:
    plt = env["plt"]
    np = env["np"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    baseline_scores = {
        str(point["basis"]): float(point["score"])
        for point in baseline_points
        if point["basis"] in PRIMARY_BASES
    }

    sweep_specs = (
        ("incidence_angle_deg", "Incidence angle (deg)", axes[0, 0], 0.0, "angle"),
        ("line_density_lpmm", "Line density (lp/mm)", axes[0, 1], 0.0, "density"),
        ("separation_um", "Separation (mm)", axes[1, 0], 1e-3, "separation"),
        ("length_to_mirror_um", "Mirror leg (mm)", axes[1, 1], 1e-3, "mirror"),
    )

    for key, xlabel, ax, scale, sweep_name in sweep_specs:
        ax2 = ax.twinx()
        for basis_name, color in (("f_bpre", "tab:green"), ("epre", "tab:orange")):
            if sweep_name == "angle":
                series = [
                    point
                    for point in points
                    if point["basis"] == basis_name
                    and point["sweep"] == "angle"
                ]
            elif sweep_name == "density":
                series = [
                    point
                    for point in points
                    if point["basis"] == basis_name
                    and point["sweep"] == "density"
                ]
            elif sweep_name == "separation":
                series = [
                    point
                    for point in points
                    if point["basis"] == basis_name
                    and point["sweep"] == "separation"
                ]
            else:
                series = [
                    point
                    for point in points
                    if point["basis"] == basis_name
                    and point["sweep"] == "mirror"
                ]
            series.sort(key=lambda point: float(point[key]))
            x_values = np.array([point[key] for point in series], dtype=np.float64)
            if scale > 0.0:
                x_values = x_values * scale
            lambdas = np.array([point["best_lambda"] for point in series], dtype=np.float64)
            ax.plot(x_values, lambdas, marker="o", color=color, label=basis_name)
            baseline_score = baseline_scores[basis_name]
            ax2.plot(
                x_values,
                np.array(
                    [float(point["score"]) / baseline_score for point in series],
                    dtype=np.float64,
                ),
                marker="x",
                linestyle="--",
                color=color,
                alpha=0.35,
            )
        ax2.set_yscale("log")
        ax2.set_ylabel("Score / baseline")
        ax2.grid(False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Best lambda")
        ax.grid(True, alpha=0.3)
        if sweep_name == "angle":
            ax.axvline(38.15, color="black", linestyle=":", linewidth=1.0)
        ax.legend(loc="upper left")

    fig.suptitle("Treacy Boundary-Basis Geometry Sweeps")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_section_iv_sweeps(
    env: dict[str, Any],
    points: list[dict[str, float | str | bool]],
    output_path: Path,
) -> None:
    plt = env["plt"]
    np = env["np"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.2), constrained_layout=True)
    sweep_specs = (
        ("a", "a", axes[0, 0], axes[1, 0]),
        ("a_prime", "a'", axes[0, 1], axes[1, 1]),
        ("b", "b", axes[0, 2], axes[1, 2]),
    )
    for sweep_name, xlabel, ax_lambda, ax_score in sweep_specs:
        for basis_name, color in (("f_bpre", "tab:green"), ("epre", "tab:orange")):
            series = sorted(
                (
                    point
                    for point in points
                    if point["sweep"] == sweep_name and point["basis"] == basis_name
                ),
                key=lambda point: float(point["sweep_value"]),
            )
            x_values = np.array([point["sweep_value"] for point in series], dtype=np.float64)
            ax_lambda.plot(
                x_values,
                np.array([point["best_lambda"] for point in series], dtype=np.float64),
                marker="o",
                color=color,
                label=basis_name,
            )
            ax_score.plot(
                x_values,
                np.array([point["score"] for point in series], dtype=np.float64),
                marker="o",
                color=color,
                label=basis_name,
            )
        ax_lambda.set_xlabel(xlabel)
        ax_lambda.set_ylabel("Best lambda")
        ax_lambda.grid(True, alpha=0.3)
        ax_lambda.legend()
        ax_score.set_xlabel(xlabel)
        ax_score.set_ylabel("Best score")
        ax_score.set_yscale("log")
        ax_score.grid(True, which="both", alpha=0.3)
        ax_score.legend()

    fig.suptitle("Martinez Section IV Boundary-Basis Sweeps")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_collapse(
    env: dict[str, Any],
    _collapse_points: list[dict[str, float | str | bool]],
    collapse_scores: list[dict[str, float | str]],
    geometry_points: list[dict[str, float | str | bool]],
    output_path: Path,
) -> None:
    plt = env["plt"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for axis, basis_name in ((axes[0, 0], "f_bpre"), (axes[0, 1], "epre")):
        basis_scores = [score for score in collapse_scores if score["basis"] == basis_name]
        basis_scores.sort(key=lambda score: float(score["coefficient_of_variation"]))
        labels = [str(score["group_label"]) for score in basis_scores]
        values = [float(score["coefficient_of_variation"]) for score in basis_scores]
        axis.bar(labels, values, color="tab:blue")
        axis.set_ylabel("Collapse spread")
        axis.set_title(f"{basis_name} normalization quality")
        axis.tick_params(axis="x", rotation=30)
        axis.grid(True, axis="y", alpha=0.3)

    crossing_points = [
        point
        for point in geometry_points
        if point["sweep"] == "angle"
        and point["basis"] in PRIMARY_BASES
    ]
    for basis_name, color in (("f_bpre", "tab:green"), ("epre", "tab:orange")):
        series = sorted(
            (point for point in crossing_points if point["basis"] == basis_name),
            key=lambda point: float(point["incidence_angle_deg"]),
        )
        axes[1, 0].plot(
            [point["incidence_angle_deg"] for point in series],
            [point["best_lambda"] for point in series],
            marker="o",
            color=color,
            label=basis_name,
        )
        axes[1, 1].plot(
            [point["incidence_angle_deg"] for point in series],
            [point["magnification"] for point in series],
            marker="o",
            color=color,
            label=basis_name,
        )
    axes[1, 0].axvline(38.15, color="black", linestyle=":", linewidth=1.0)
    axes[1, 0].set_xlabel("Incidence angle (deg)")
    axes[1, 0].set_ylabel("Best lambda")
    axes[1, 0].set_title("Angle sweep through M = 1 crossing")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].axhline(1.0, color="black", linestyle=":", linewidth=1.0)
    axes[1, 1].axvline(38.15, color="black", linestyle=":", linewidth=1.0)
    axes[1, 1].set_xlabel("Incidence angle (deg)")
    axes[1, 1].set_ylabel("M = cos(td)/cos(ti)")
    axes[1, 1].set_title("Grating magnification across angle sweep")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.suptitle("Boundary-Basis Collapse and M-Crossing Diagnostics")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "physics" / "boundary_probe_sweeps",
    )
    args = parser.parse_args()
    for path in generate_artifacts(args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
