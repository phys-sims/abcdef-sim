from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.analytics.martinez_debug import (
    run_martinez_section_iv_coefficient_audit,
    run_martinez_section_iv_phase_study,
    run_treacy_partition_span_study,
)

pytestmark = pytest.mark.physics


def test_section_iv_matrix_audit_matches_equation_34_structure_on_narrow_span() -> None:
    points = run_martinez_section_iv_coefficient_audit(span_scale=0.125)

    actual_a = np.array([point.system_a for point in points], dtype=np.float64)
    actual_d = np.array([point.system_d for point in points], dtype=np.float64)
    actual_f = np.array([point.system_f for point in points], dtype=np.float64)
    expected_b = np.array([point.expected_b_um for point in points], dtype=np.float64)
    actual_b = np.array([point.system_b_um for point in points], dtype=np.float64)

    np.testing.assert_allclose(actual_a, -1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_d, -1.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_f, 0.0, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(actual_b, expected_b, rtol=0.0, atol=1e-9)


def test_section_iv_phase_study_shows_sample_k_improves_series2_tod_error() -> None:
    points = run_martinez_section_iv_phase_study(span_scales=(1.0,))
    point_map = {point.variant: point for point in points}

    assert point_map["sample_k_series2"].tod_abs_error_fs3 < point_map[
        "center_k_series2"
    ].tod_abs_error_fs3
    assert point_map["sample_k_series2"].weighted_rms_rad > 0.0


def test_treacy_partition_span_study_geometry_term_improves_current_full_span_error() -> None:
    points = run_treacy_partition_span_study(span_scales=(1.0,))
    point_map = {point.variant: point for point in points}

    assert point_map["geom_current"].raw_abcdef_gdd_rel_error < point_map[
        "axial_current"
    ].raw_abcdef_gdd_rel_error
    assert point_map["geom_current"].raw_abcdef_tod_rel_error < point_map[
        "axial_current"
    ].raw_abcdef_tod_rel_error


def test_treacy_partition_span_study_narrower_bandwidth_reduces_geom_series2_error() -> None:
    points = run_treacy_partition_span_study(span_scales=(1.0, 0.125))
    geom_series2 = sorted(
        (point for point in points if point.variant == "geom_series2"),
        key=lambda point: point.span_scale,
        reverse=True,
    )
    wide, narrow = geom_series2

    assert abs(narrow.raw_abcdef_gdd_rel_error - wide.raw_abcdef_gdd_rel_error) < 1e-3
    assert narrow.raw_abcdef_tod_rel_error < wide.raw_abcdef_tod_rel_error
