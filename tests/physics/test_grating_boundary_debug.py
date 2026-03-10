from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim import BeamSpec, PulseSpec, StandaloneLaserSpec, run_abcdef, treacy_compressor_preset
from abcdef_sim.analytics import (
    run_treacy_alignment_audit,
    run_treacy_actual_local_matrix_audit,
    run_treacy_actual_local_matrix_series,
    run_treacy_fitted_surface_map_audit,
    run_treacy_fitted_surface_map_series,
    run_treacy_adjacent_pair_subsystem_audit,
    run_treacy_adjacent_pair_subsystem_series,
    run_section_iv_grating_boundary_audit,
    run_section_iv_grating_boundary_series,
    run_treacy_endpoint_decomposition_study,
    run_treacy_grating_boundary_audit,
    run_treacy_grating_boundary_series,
)
from abcdef_sim.physics.abcdef.phase_terms import martinez_k

pytestmark = pytest.mark.physics


def test_treacy_grating_boundary_audit_intersects_surface_planes_exactly() -> None:
    points = run_treacy_grating_boundary_audit(beam_radius_mm=1000.0)

    assert points
    assert max(point.max_abs_surface_hit_plane_residual_um for point in points) < 1e-9


def test_treacy_grating_boundary_series_exposes_runtime_surface_state_gap() -> None:
    series = run_treacy_grating_boundary_series(beam_radius_mm=1000.0)

    assert series
    for point in series:
        assert point.omega_rad_per_fs.shape == point.input_plane_x_um.shape
        assert point.omega_rad_per_fs.shape == point.output_plane_x_um.shape
        assert point.omega_rad_per_fs.shape == point.surface_coordinate_input_frame_um.shape
        assert point.omega_rad_per_fs.shape == point.surface_coordinate_output_frame_um.shape
        assert point.omega_rad_per_fs.shape == point.tangential_phase_match_residual_per_um.shape

    g1 = next(point for point in series if point.instance_name == "g1")
    assert np.max(np.abs(g1.tangential_phase_match_residual_per_um)) < 1e-9
    assert np.max(np.abs(np.degrees(g1.incidence_angle_rad) - g1.configured_incidence_angle_deg)) < 1e-9

    points = run_treacy_grating_boundary_audit(beam_radius_mm=1000.0)
    assert max(point.surface_intersection_distance_rms_um for point in points) > 1.0
    assert max(point.surface_coordinate_minus_input_x_rms_um for point in points) > 1.0
    assert max(point.surface_coordinate_minus_output_x_rms_um for point in points) > 1.0
    assert max(point.max_abs_tangential_phase_match_residual_per_um for point in points) > 1e-3
    assert max(point.max_abs_incidence_angle_mismatch_deg for point in points) > 1.0


def test_treacy_adjacent_pair_subsystem_series_tracks_consecutive_grating_hits() -> None:
    series = run_treacy_adjacent_pair_subsystem_series(beam_radius_mm=1000.0)

    assert [point.pair_name for point in series] == [
        "g1->g2",
        "g2->g2_return",
        "g2_return->g1_return",
    ]
    for point in series:
        assert point.omega_rad_per_fs.shape == point.first_surface_output_x_um.shape
        assert point.omega_rad_per_fs.shape == point.second_surface_input_x_um.shape
        assert point.omega_rad_per_fs.shape == point.predicted_second_surface_x_um.shape
        assert point.omega_rad_per_fs.shape == point.x_residual_um.shape
        assert point.omega_rad_per_fs.shape == point.xprime_residual.shape
        assert point.omega_rad_per_fs.shape == point.anchored_predicted_second_surface_x_um.shape
        assert point.omega_rad_per_fs.shape == point.anchored_x_residual_um.shape
        assert point.omega_rad_per_fs.shape == point.anchored_xprime_residual.shape


def test_treacy_adjacent_pair_subsystem_audit_exposes_surface_to_surface_map_gap() -> None:
    points = run_treacy_adjacent_pair_subsystem_audit(beam_radius_mm=1000.0)

    assert len(points) == 3
    assert max(point.x_relative_residual_rms for point in points) > 0.1
    assert max(point.xprime_relative_residual_rms for point in points) > 0.1
    assert max(point.canonical_kernel_sample_k_rms_rad for point in points) > 1.0
    assert max(point.anchored_x_relative_residual_rms for point in points) < max(
        point.x_relative_residual_rms for point in points
    )
    assert max(point.anchored_xprime_relative_residual_rms for point in points) < max(
        point.xprime_relative_residual_rms for point in points
    )
    for point in points:
        assert point.anchored_x_relative_residual_rms < point.x_relative_residual_rms
        assert point.anchored_xprime_relative_residual_rms < point.xprime_relative_residual_rms


def test_treacy_alignment_audit_confirms_fold_and_return_plane_geometry() -> None:
    audit = run_treacy_alignment_audit()

    assert abs(audit.g2_return_plane_normal_offset_um) < 1e-9
    assert audit.g2_return_plane_angle_diff_mod_pi_rad < 1e-12
    assert abs(audit.g1_return_plane_normal_offset_um) < 1e-9
    assert audit.g1_return_plane_angle_diff_mod_pi_rad < 1e-12
    assert audit.fold_retro_axis_error_rad < 1e-12
    assert audit.fold_point_roundtrip_error_um < 1e-9


def test_treacy_actual_local_matrix_series_exposes_fixed_incidence_gap_on_later_hits() -> None:
    series = run_treacy_actual_local_matrix_series(beam_radius_mm=1000.0)

    g1 = next(point for point in series if point.instance_name == "g1")
    assert np.max(np.abs(g1.output_surface_x_um - g1.predicted_output_x_actual_um)) < 1e-12
    assert np.max(np.abs(g1.output_surface_xprime - g1.predicted_output_xprime_actual)) < 1e-12

    points = {point.instance_name: point for point in run_treacy_actual_local_matrix_audit(beam_radius_mm=1000.0)}
    g2 = points["g2"]
    assert g2.fixed_model_x_relative_residual_rms > 0.1
    assert g2.fixed_model_xprime_relative_residual_rms > 0.1
    assert g2.actual_local_x_residual_rms_um > g2.fixed_model_x_residual_rms_um
    assert g2.actual_local_xprime_residual_rms < 0.25 * g2.fixed_model_xprime_residual_rms

    for name in ("g2_return", "g1_return"):
        point = points[name]
        assert point.actual_local_x_residual_rms_um > point.fixed_model_x_residual_rms_um
        assert point.actual_local_xprime_residual_rms < 1e-12


def test_treacy_fitted_surface_map_series_reconstructs_surface_state_in_configured_frame() -> None:
    series = run_treacy_fitted_surface_map_series(beam_radius_mm=1000.0)
    points = {
        point.instance_name: point for point in run_treacy_fitted_surface_map_audit(beam_radius_mm=1000.0)
    }
    actual_local_points = {
        point.instance_name: point for point in run_treacy_actual_local_matrix_audit(beam_radius_mm=1000.0)
    }

    g1 = next(point for point in series if point.instance_name == "g1")
    assert np.max(np.abs(g1.output_surface_x_um - g1.predicted_output_surface_x_um)) < 1e-12
    assert np.max(np.abs(g1.output_surface_xprime - g1.predicted_output_surface_xprime)) < 1e-12

    for name in ("g2", "g2_return", "g1_return"):
        point = points[name]
        actual_local = actual_local_points[name]
        assert point.fitted_b_rms_um < 1e-12
        assert point.fitted_e_rms_um < 1e-12
        assert point.fitted_x_residual_rms_um < 1e-4
        assert point.fitted_xprime_residual_rms < actual_local.fixed_model_xprime_residual_rms
        assert point.fitted_generator_center_k_rms_rad > 1.0


def test_section_iv_boundary_series_matches_paired_shorthand_on_first_grating() -> None:
    series = run_section_iv_grating_boundary_series()

    g1 = next(point for point in series if point.stage_name == "g1")
    assert np.max(np.abs(g1.output_plane_x_um - g1.paired_output_x_um)) < 1e-12
    assert np.max(np.abs(g1.f_exact - g1.f_paired)) < 1e-12
    assert np.max(np.abs(g1.phi3_exact_sample_k_rad)) < 1e-12
    assert np.max(np.abs(g1.phi3_paired_sample_k_rad)) < 1e-12


def test_section_iv_boundary_audit_exposes_second_grating_pairing_gap() -> None:
    points = run_section_iv_grating_boundary_audit()

    g2 = next(point for point in points if point.stage_name == "g2")
    assert g2.output_x_pair_relative_rms > 0.1
    assert g2.f_pair_relative_rms > 0.01
    assert g2.phi3_pair_mismatch_rms_rad > 1.0
    assert g2.phi3_oracle_vs_paired_mismatch_rms_rad > 1.0
    assert g2.global_endpoint_exact_sample_k_rms_rad > 1.0
    assert g2.phi3_oracle_vs_exact_plus_global_endpoint_rms_rad < 1e-9


def test_section_iv_boundary_series_reconstructs_quadratic_paired_reference() -> None:
    series = run_section_iv_grating_boundary_series()

    g1 = next(point for point in series if point.stage_name == "g1")
    g2 = next(point for point in series if point.stage_name == "g2")
    expected = 0.5 * martinez_k(g2.omega_rad_per_fs) * g2.expected_b_um * (g2.magnification * g1.f_exact) ** 2

    assert np.max(np.abs(g1.phi3_oracle_sample_k_rad + g2.phi3_oracle_sample_k_rad - expected)) < 1e-12


def test_section_iv_boundary_series_exact_phi3_plus_global_endpoint_closes_oracle() -> None:
    series = run_section_iv_grating_boundary_series()

    g2 = next(point for point in series if point.stage_name == "g2")
    assert np.max(
        np.abs(
            g2.phi3_oracle_sample_k_rad
            - (g2.phi3_exact_sample_k_rad + g2.global_endpoint_exact_sample_k_rad)
        )
    ) < 1e-11
    assert np.max(
        np.abs(g2.phi3_exact_sample_k_rad + g2.local_endpoint_exact_sample_k_rad)
    ) < 1e-12


def test_section_iv_exact_phi3_matches_symmetrized_type2_generator_image() -> None:
    series = run_section_iv_grating_boundary_series()

    g2 = next(point for point in series if point.stage_name == "g2")
    a_exact = g2.output_plane_x_um / g2.input_plane_x_um
    generator = (a_exact * g2.input_plane_x_um * g2.output_plane_xprime) - (
        a_exact * g2.f_exact * g2.input_plane_x_um
    )
    symmetrized = martinez_k(g2.omega_rad_per_fs) * (
        0.5
        * (
            (g2.input_plane_x_um * g2.input_plane_xprime)
            + (g2.output_plane_x_um * g2.output_plane_xprime)
        )
        - generator
    )

    assert np.max(np.abs(symmetrized - g2.phi3_exact_sample_k_rad)) < 1e-11


def test_treacy_endpoint_decomposition_current_matches_runtime_result() -> None:
    points = run_treacy_endpoint_decomposition_study(
        beam_radii_mm=(1000.0,),
        mirror_lengths_um=(0.0,),
    )
    current = next(point for point in points if point.variant == "current")

    result = run_abcdef(
        treacy_compressor_preset(length_to_mirror_um=0.0),
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=100.0,
                center_wavelength_nm=1030.0,
                n_samples=256,
                time_window_fs=3000.0,
            ),
            beam=BeamSpec(radius_mm=1000.0, m2=1.0),
        ),
    )

    assert current.raw_abcdef_gdd_fs2 == pytest.approx(
        float(result.final_state.metrics["abcdef.gdd_fs2"])
    )
    assert current.raw_abcdef_tod_fs3 == pytest.approx(
        float(result.final_state.metrics["abcdef.tod_fs3"])
    )


def test_treacy_endpoint_decomposition_variants_are_distinct() -> None:
    points = run_treacy_endpoint_decomposition_study(
        beam_radii_mm=(1000.0,),
        mirror_lengths_um=(0.0,),
    )
    by_variant = {point.variant: point for point in points}
    current = by_variant["current"]
    actual_local = by_variant["actual_local_f_stage_center_global_center"]
    configured_surface_slope = by_variant["configured_surface_slope_stage_center_global_center"]
    configured_surface_slope_anchored = by_variant[
        "configured_surface_slope_anchored_center_global_center"
    ]
    fitted_surface_generator = by_variant["fitted_surface_generator_center_global_center"]
    fitted_surface_generator_pairs = by_variant[
        "fitted_surface_generator_plus_anchored_pairs_plus_global_center"
    ]

    assert by_variant["exact_local_center_global_center"].raw_abcdef_tod_fs3 != pytest.approx(
        current.raw_abcdef_tod_fs3
    )
    assert by_variant["exact_local_sample_global_sample"].raw_abcdef_gdd_fs2 != pytest.approx(
        current.raw_abcdef_gdd_fs2
    )
    assert by_variant["surface_hit_center_global_center"].raw_abcdef_gdd_fs2 != pytest.approx(
        by_variant["exact_local_center_global_center"].raw_abcdef_gdd_fs2
    )
    assert by_variant[
        "canonical_surface_sym_center_global_center"
    ].raw_abcdef_gdd_fs2 != pytest.approx(current.raw_abcdef_gdd_fs2)
    assert by_variant[
        "canonical_surface_raw_center_no_global"
    ].raw_abcdef_tod_fs3 != pytest.approx(current.raw_abcdef_tod_fs3)
    assert by_variant[
        "surface_anchored_runtime_center_global_center"
    ].raw_abcdef_tod_fs3 != pytest.approx(current.raw_abcdef_tod_fs3)
    assert by_variant[
        "surface_anchored_transport_center_global_center"
    ].raw_abcdef_tod_fs3 != pytest.approx(by_variant["exact_local_center_global_center"].raw_abcdef_tod_fs3)
    assert configured_surface_slope.raw_abcdef_gdd_fs2 != pytest.approx(current.raw_abcdef_gdd_fs2)
    assert configured_surface_slope.raw_abcdef_gdd_rel_error > 1.0
    assert configured_surface_slope.raw_abcdef_tod_rel_error > 1.0
    assert configured_surface_slope_anchored.raw_abcdef_gdd_rel_error > 1.0
    assert configured_surface_slope_anchored.raw_abcdef_tod_rel_error > 1.0
    assert configured_surface_slope_anchored.weighted_rms_rad > configured_surface_slope.weighted_rms_rad
    assert fitted_surface_generator.raw_abcdef_gdd_rel_error > 1.0
    assert fitted_surface_generator.raw_abcdef_tod_rel_error > 1.0
    assert fitted_surface_generator_pairs.raw_abcdef_gdd_rel_error > 1.0
    assert fitted_surface_generator_pairs.raw_abcdef_tod_rel_error > 0.5
    assert actual_local.raw_abcdef_gdd_fs2 != pytest.approx(current.raw_abcdef_gdd_fs2)
    assert actual_local.raw_abcdef_gdd_rel_error > 1.0
    assert actual_local.raw_abcdef_tod_rel_error > 1.0
    assert actual_local.weighted_rms_rad > current.weighted_rms_rad
    assert by_variant[
        "incoming_state_g2_surface_center_global_center"
    ].raw_abcdef_gdd_fs2 != pytest.approx(
        by_variant["exact_local_center_global_center"].raw_abcdef_gdd_fs2
    )
    assert by_variant["conjugate_pair_f2sq_center"].raw_abcdef_gdd_fs2 != pytest.approx(
        current.raw_abcdef_gdd_fs2
    )
    assert by_variant["conjugate_pair_mf1sq_center"].raw_abcdef_tod_fs3 != pytest.approx(
        current.raw_abcdef_tod_fs3
    )
