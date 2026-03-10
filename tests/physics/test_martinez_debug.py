from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.analytics.martinez_debug import (
    MartinezSectionIVPhi3VariantPoint,
    run_martinez_section_iv_coefficient_audit,
    run_martinez_section_iv_phi3_variant_study,
    run_treacy_grating_frame_audit,
    run_treacy_phi3_per_grating_budget,
    run_treacy_phi3_sign_audit,
    run_treacy_phi3_variant_comparison,
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


def test_section_iv_phi3_variants_have_candidate_better_than_runtime_exact() -> None:
    points = run_martinez_section_iv_phi3_variant_study(span_scales=(1.0, 0.125))
    grouped: dict[str, list[float]] = {}
    for point in points:
        grouped.setdefault(point.variant, []).append(
            point.gdd_abs_error_fs2 + point.tod_abs_error_fs3
        )

    current_error = float(np.mean(grouped["runtime_exact_centerk"]))
    best_variant, best_error = min(
        (
            (variant, float(np.mean(errors)))
            for variant, errors in grouped.items()
            if variant != "runtime_exact_centerk"
        ),
        key=lambda item: item[1],
    )

    assert best_variant != "runtime_exact_centerk"
    assert best_error < current_error


def test_section_iv_quadratic_reference_favors_phasead_samplek_variant() -> None:
    legacy_points = run_martinez_section_iv_phi3_variant_study(
        span_scales=(1.0, 0.125),
        reference_mode="legacy_linear",
    )
    quadratic_points = run_martinez_section_iv_phi3_variant_study(
        span_scales=(1.0, 0.125),
        reference_mode="quadratic_bfg2",
    )

    def _group_mean(points: tuple[MartinezSectionIVPhi3VariantPoint, ...]) -> dict[str, float]:
        grouped: dict[str, list[float]] = {}
        for point in points:
            grouped.setdefault(point.variant, []).append(
                point.gdd_abs_error_fs2 + point.tod_abs_error_fs3
            )
        return {variant: float(np.mean(values)) for variant, values in grouped.items()}

    legacy_mean = _group_mean(legacy_points)
    quadratic_mean = _group_mean(quadratic_points)

    assert quadratic_mean["martinez_series2_phasead_samplek"] < (
        legacy_mean["martinez_series2_phasead_samplek"] / 3.0
    )
    assert (
        quadratic_mean["martinez_series2_phasead_samplek"] < quadratic_mean["runtime_exact_centerk"]
    )


def test_treacy_phi3_variant_comparison_has_candidate_better_than_runtime_exact_large_beam() -> (
    None
):
    points = run_treacy_phi3_variant_comparison(beam_radii_mm=(10.0, 100.0, 1000.0))
    grouped: dict[str, list[float]] = {}
    for point in points:
        grouped.setdefault(point.variant, []).append(
            point.raw_abcdef_gdd_rel_error + point.raw_abcdef_tod_rel_error
        )

    current_error = float(np.mean(grouped["runtime_exact_centerk"]))
    best_variant, best_error = min(
        (
            (variant, float(np.mean(errors)))
            for variant, errors in grouped.items()
            if variant != "runtime_exact_centerk"
        ),
        key=lambda item: item[1],
    )

    assert best_variant != "runtime_exact_centerk"
    assert best_error < current_error


def test_treacy_terminal_bpre_probe_nearly_closes_large_beam_zero_mirror_gap() -> None:
    points = run_treacy_phi3_variant_comparison(beam_radii_mm=(10.0, 100.0, 1000.0))
    probe_points = [
        point for point in points if point.variant == "martinez_series2_terminal_bpre_probe"
    ]

    assert len(probe_points) == 3
    assert max(point.raw_abcdef_gdd_rel_error for point in probe_points) < 5e-3
    assert max(point.raw_abcdef_tod_rel_error for point in probe_points) < 5e-3


def test_treacy_phi3_per_grating_budget_concentrates_required_residual_on_later_passes() -> None:
    comparison = run_treacy_phi3_variant_comparison(beam_radii_mm=(10.0, 100.0, 1000.0))
    grouped: dict[str, list[float]] = {}
    for point in comparison:
        grouped.setdefault(point.variant, []).append(
            point.raw_abcdef_gdd_rel_error + point.raw_abcdef_tod_rel_error
        )
    best_variant = min(
        ((variant, float(np.mean(errors))) for variant, errors in grouped.items()),
        key=lambda item: item[1],
    )[0]

    points = run_treacy_phi3_per_grating_budget(beam_radius_mm=1000.0, variants=(best_variant,))
    return_pass_budget = sum(
        abs(point.required_residual_gdd_fs2) for point in points if point.is_return_pass
    )
    forward_budget = sum(
        abs(point.required_residual_gdd_fs2) for point in points if not point.is_return_pass
    )

    assert return_pass_budget > forward_budget


def test_treacy_phi3_sign_audit_shows_return_flip_changes_the_residual() -> None:
    stage_points, variant_points = run_treacy_phi3_sign_audit(beam_radius_mm=1000.0)
    assert any(point.return_flip_hypothesis for point in stage_points)

    point_map = {point.sign_case: point for point in variant_points}
    current = point_map["current"]
    flipped = point_map["flip_return_pass"]

    assert (
        abs(flipped.raw_abcdef_gdd_rel_error - current.raw_abcdef_gdd_rel_error) > 1e-4
        or abs(flipped.raw_abcdef_tod_rel_error - current.raw_abcdef_tod_rel_error) > 1e-4
    )


def test_treacy_post_grating_frame_transforms_preserve_x_and_flip_xprime() -> None:
    points = run_treacy_grating_frame_audit(beam_radius_mm=1000.0)

    assert points
    assert max(point.max_x_delta_um for point in points) < 1e-12
    assert max(point.max_xprime_flip_residual for point in points) < 1e-12
