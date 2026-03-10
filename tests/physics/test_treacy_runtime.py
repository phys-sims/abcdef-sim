from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim import (
    AbcdefCfg,
    BeamSpec,
    FreeSpaceCfg,
    GratingCfg,
    InputRayCfg,
    PulseSpec,
    StandaloneLaserSpec,
    run_abcdef,
    treacy_compressor_preset,
)
from abcdef_sim.analytics.treacy_benchmark import (
    run_treacy_mirror_heatmap,
    run_treacy_radius_convergence,
)
from abcdef_sim.physics.abcdef.phase_terms import combine_phi_total_rad
from abcdef_sim.physics.abcdef.treacy import compute_treacy_analytic_metrics

pytestmark = pytest.mark.physics


def _weighted_xprime_rms(result: object) -> float:
    from abcdef_sim.data_models.results import AbcdefRunResult

    if not isinstance(result, AbcdefRunResult):
        raise TypeError(f"Expected AbcdefRunResult, got {type(result).__name__}")

    rays = np.asarray(result.pipeline_result.final_state.rays, dtype=np.float64)
    weights = np.asarray(result.fit.weights, dtype=np.float64)
    weights = weights / np.sum(weights)
    return float(np.sqrt(np.sum(weights * rays[:, 1, 0] ** 2)))


def _default_laser() -> StandaloneLaserSpec:
    return StandaloneLaserSpec(
        pulse=PulseSpec(
            width_fs=100.0,
            center_wavelength_nm=1030.0,
            n_samples=256,
            time_window_fs=3000.0,
        ),
        beam=BeamSpec(radius_mm=1.0, m2=1.0),
    )


def test_run_abcdef_treacy_produces_nonzero_phi1_for_finite_beam() -> None:
    result = run_abcdef(
        treacy_compressor_preset(length_to_mirror_um=50_000.0),
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=100.0,
                center_wavelength_nm=1030.0,
                n_samples=256,
                time_window_fs=3000.0,
            ),
            beam=BeamSpec(radius_mm=0.2, m2=1.0),
        ),
    )

    assert result.pipeline_result.phi1_rad is not None
    assert np.max(np.abs(np.asarray(result.pipeline_result.phi1_rad, dtype=np.float64))) > 0.0


def test_run_abcdef_treacy_preserves_phi2_for_noncentered_input_ray() -> None:
    cfg = treacy_compressor_preset(length_to_mirror_um=50_000.0).model_copy(
        update={"input_ray": InputRayCfg(x=50.0, x_prime=0.02)}
    )
    result = run_abcdef(
        cfg,
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=100.0,
                center_wavelength_nm=1030.0,
                n_samples=256,
                time_window_fs=3000.0,
            ),
            beam=BeamSpec(radius_mm=0.2, m2=1.0),
        ),
    )

    assert result.pipeline_result.phi2_rad is not None
    assert np.max(np.abs(np.asarray(result.pipeline_result.phi2_rad, dtype=np.float64))) > 0.0

    expected_phi_total = combine_phi_total_rad(
        np.asarray(
            result.final_state.meta["abcdef"]["treacy_analytic_phase_rad"], dtype=np.float64
        ),
        result.pipeline_result.phi1_rad,
        result.pipeline_result.phi2_rad,
        result.pipeline_result.phi4_rad,
    )
    np.testing.assert_allclose(result.pipeline_result.phi_total_rad, expected_phi_total)


def test_treacy_runtime_gdd_error_decreases_with_beam_radius() -> None:
    points = run_treacy_radius_convergence(
        beam_radii_mm=(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0),
        length_to_mirror_um=0.0,
        n_samples=256,
    )

    gdd_errors = np.array([point.without_phi2_gdd_rel_error for point in points], dtype=np.float64)
    assert np.all(np.isfinite(gdd_errors))
    assert gdd_errors[-1] < gdd_errors[0]
    assert gdd_errors[-1] * 4.0 < gdd_errors[0]
    assert points[-1].full_gdd_rel_error < 1e-3

    for point in points:
        assert np.isfinite(point.full_gdd_fs2)
        assert np.isfinite(point.full_gdd_rel_error)
        assert np.isfinite(point.without_phi2_gdd_fs2)
        assert np.isfinite(point.without_phi2_gdd_rel_error)
        assert np.isfinite(point.full_tod_fs3)
        assert np.isfinite(point.full_tod_rel_error)
        assert np.isfinite(point.without_phi2_tod_fs3)
        assert np.isfinite(point.without_phi2_tod_rel_error)
        assert np.isfinite(point.x_centroid_span_um)
        assert np.isfinite(point.x_centroid_slope_um_per_rad_per_fs)
        assert np.isfinite(point.x_prime_span)
        assert np.isfinite(point.x_prime_slope_per_rad_per_fs)
        assert np.isfinite(point.weighted_x_centroid_rms_um)
        assert np.isfinite(point.weighted_x_prime_rms)
        assert np.isfinite(point.weighted_mean_w_out_um)
        assert np.isfinite(point.weighted_mean_diffraction_angle_rad)
        assert np.isfinite(point.normalized_spatial_chirp_rms)
        assert np.isfinite(point.normalized_angular_dispersion_rms)
        assert np.isfinite(point.mode_overlap_with_center)
        assert np.isfinite(point.pulse_front_tilt_fs_per_um)
        assert 0.0 <= point.mode_overlap_with_center <= 1.0
    assert points[-1].full_gdd_rel_error > points[-1].without_phi2_gdd_rel_error
    assert points[-1].mode_overlap_with_center > points[0].mode_overlap_with_center
    assert points[-1].normalized_spatial_chirp_rms < points[0].normalized_spatial_chirp_rms
    for point in points[-4:]:
        assert np.sign(point.without_phi2_gdd_fs2) == np.sign(point.analytic_gdd_fs2)
        assert np.sign(point.without_phi2_tod_fs3) == np.sign(point.analytic_tod_fs3)


def test_treacy_double_pass_fold_reduces_weighted_output_angular_dispersion() -> None:
    laser = _default_laser()

    single_pass = run_abcdef(treacy_compressor_preset(n_passes=1), laser)
    double_pass = run_abcdef(treacy_compressor_preset(length_to_mirror_um=0.0), laser)

    assert _weighted_xprime_rms(double_pass) < _weighted_xprime_rms(single_pass)
    assert _weighted_xprime_rms(double_pass) < 1e-4


def test_treacy_single_pass_pair_cancels_weighted_output_angular_dispersion() -> None:
    laser = _default_laser()
    single_grating = run_abcdef(
        AbcdefCfg(
            name="single_grating_pair_probe",
            optics=(
                GratingCfg(
                    instance_name="g1",
                    line_density_lpmm=1200.0,
                    incidence_angle_deg=35.0,
                    diffraction_order=-1,
                ),
                FreeSpaceCfg(instance_name="gap_12", length=100_000.0),
            ),
        ),
        laser,
    )
    single_pass_pair = run_abcdef(treacy_compressor_preset(n_passes=1), laser)

    assert _weighted_xprime_rms(single_pass_pair) < (_weighted_xprime_rms(single_grating) / 50.0)
    assert _weighted_xprime_rms(single_pass_pair) < 5e-4


def test_treacy_runtime_records_resolved_diffraction_geometry() -> None:
    result = run_abcdef(treacy_compressor_preset(n_passes=1), _default_laser())

    meta = result.final_state.meta["abcdef"]
    assert meta["treacy_resolved_diffraction_angle_deg"] == pytest.approx(
        41.484970499511796,
        rel=0.0,
        abs=1e-12,
    )
    resolved_optics = meta["treacy_resolved_optics"]
    g2 = next(optic for optic in resolved_optics if optic["instance_name"] == "g2")
    assert g2["incidence_angle_deg"] == pytest.approx(meta["treacy_resolved_diffraction_angle_deg"])


def test_treacy_runtime_mirror_leg_changes_abcdef_result_while_analytic_stays_fixed() -> None:
    points = run_treacy_mirror_heatmap(
        beam_radii_mm=(0.2,),
        mirror_lengths_um=(0.0, 50_000.0, 100_000.0),
        n_samples=256,
    )

    analytic_gdd = {round(point.analytic_gdd_fs2, 6) for point in points}
    full_gdd = np.array([point.full_gdd_fs2 for point in points], dtype=np.float64)
    without_phi2_gdd = np.array([point.without_phi2_gdd_fs2 for point in points], dtype=np.float64)
    gdd_errors = np.array([point.without_phi2_gdd_rel_error for point in points], dtype=np.float64)
    x_spans = np.array([point.x_centroid_span_um for point in points], dtype=np.float64)
    x_prime_spans = np.array([point.x_prime_span for point in points], dtype=np.float64)
    normalized_spatial = np.array(
        [point.normalized_spatial_chirp_rms for point in points], dtype=np.float64
    )
    mode_overlap = np.array([point.mode_overlap_with_center for point in points], dtype=np.float64)

    assert len(analytic_gdd) == 1
    assert np.max(np.abs(full_gdd - full_gdd[0])) > 1e2
    assert np.max(np.abs(without_phi2_gdd - without_phi2_gdd[0])) > 1e2
    assert np.max(gdd_errors) > np.min(gdd_errors)
    np.testing.assert_allclose(x_spans, x_spans[0], rtol=0.0, atol=1e-9)
    assert np.all(np.isfinite(x_prime_spans))
    np.testing.assert_allclose(x_prime_spans, x_prime_spans[0], rtol=0.0, atol=1e-12)
    assert np.max(normalized_spatial) > np.min(normalized_spatial)
    assert np.max(mode_overlap) > np.min(mode_overlap)


def test_treacy_runtime_tracks_local_analytic_baseline() -> None:
    analytic = compute_treacy_analytic_metrics(
        line_density_lpmm=1200.0,
        incidence_angle_deg=35.0,
        separation_um=100_000.0,
        wavelength_nm=1030.0,
        diffraction_order=-1,
        n_passes=2,
    )
    point = run_treacy_radius_convergence(
        beam_radii_mm=(10.0,),
        length_to_mirror_um=0.0,
        n_samples=256,
    )[0]

    assert point.analytic_gdd_fs2 == pytest.approx(analytic.gdd_fs2, rel=1e-12, abs=1e-12)
    assert point.analytic_tod_fs3 == pytest.approx(analytic.tod_fs3, rel=1e-12, abs=1e-12)
