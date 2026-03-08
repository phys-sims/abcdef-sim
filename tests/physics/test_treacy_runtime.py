from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim import (
    BeamSpec,
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

    gdd_errors = np.array([point.gdd_rel_error for point in points], dtype=np.float64)
    assert np.all(np.isfinite(gdd_errors))
    assert gdd_errors[-1] < gdd_errors[0]
    assert gdd_errors[-1] * 4.0 < gdd_errors[0]

    for point in points:
        assert np.isfinite(point.comparison_gdd_fs2)
        assert np.isfinite(point.comparison_tod_fs3)
        assert np.isfinite(point.runner_gdd_fs2)
        assert np.isfinite(point.runner_tod_fs3)
    for point in points[-4:]:
        assert np.sign(point.comparison_gdd_fs2) == np.sign(point.analytic_gdd_fs2)
        assert np.sign(point.comparison_tod_fs3) == np.sign(point.analytic_tod_fs3)


def test_treacy_runtime_mirror_leg_changes_abcdef_result_while_analytic_stays_fixed() -> None:
    points = run_treacy_mirror_heatmap(
        beam_radii_mm=(0.2,),
        mirror_lengths_um=(0.0, 50_000.0, 100_000.0),
        n_samples=256,
    )

    analytic_gdd = {round(point.analytic_gdd_fs2, 6) for point in points}
    comparison_gdd = np.array([point.comparison_gdd_fs2 for point in points], dtype=np.float64)
    gdd_errors = np.array([point.gdd_rel_error for point in points], dtype=np.float64)

    assert len(analytic_gdd) == 1
    assert np.max(np.abs(comparison_gdd - comparison_gdd[0])) > 1e2
    assert np.max(gdd_errors) > np.min(gdd_errors)


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
