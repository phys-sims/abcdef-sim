from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim import (
    AbcdefCfg,
    BeamSpec,
    FreeSpaceCfg,
    GratingCfg,
    PulseSpec,
    StandaloneLaserSpec,
    run_abcdef,
    treacy_compressor_preset,
)
from abcdef_sim.analytics import (
    build_output_plane_field_1d,
    summarize_output_plane_field,
    summarize_output_plane_geometry,
)
from abcdef_sim.physics.abcdef.phase_terms import martinez_k_center

pytestmark = pytest.mark.physics


def test_output_plane_field_preserves_spectral_power_and_centroid_for_treacy() -> None:
    result = run_abcdef(
        treacy_compressor_preset(length_to_mirror_um=0.0),
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
    field = build_output_plane_field_1d(result)
    spectral_power = np.trapezoid(field.intensity_x_omega, x=field.x_um, axis=0)

    np.testing.assert_allclose(
        spectral_power,
        field.spectral_power_au,
        rtol=2e-2,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        field.x_centroids_omega_um,
        field.x_out_um,
        rtol=0.0,
        atol=float(2.0 * np.max(np.diff(field.x_um))),
    )
    np.testing.assert_allclose(
        2.0 * field.x_rms_omega_um,
        field.w_out_um,
        rtol=2e-2,
        atol=float(2.0 * np.max(np.diff(field.x_um))),
    )


def test_output_plane_field_phase_curvature_matches_phi4_sign_convention() -> None:
    cfg = AbcdefCfg(optics=(FreeSpaceCfg(instance_name="fs", length=100_000.0),))
    result = run_abcdef(
        cfg,
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=100.0,
                center_wavelength_nm=1030.0,
                n_samples=128,
                time_window_fs=2000.0,
            ),
            beam=BeamSpec(radius_mm=0.5, m2=1.0),
        ),
    )
    x_grid = np.linspace(-2_500.0, 2_500.0, 401, dtype=np.float64)
    field = build_output_plane_field_1d(result, x_grid_um=x_grid)

    center_freq_idx = int(np.argmin(np.abs(field.delta_omega_rad_per_fs)))
    center_x_idx = int(np.argmin(np.abs(field.x_um - field.x_out_um[center_freq_idx])))
    probe_x_idx = center_x_idx + 40
    ratio = (
        field.field_x_omega[probe_x_idx, center_freq_idx]
        / field.field_x_omega[
            center_x_idx,
            center_freq_idx,
        ]
    )
    dx = field.x_um[probe_x_idx] - field.x_out_um[center_freq_idx]
    expected_phase = np.real(
        martinez_k_center(field.omega_rad_per_fs) * dx**2 / (2.0 * field.q_out_um[center_freq_idx])
    )

    assert np.angle(ratio) == pytest.approx(expected_phase, abs=1e-9)


def test_output_plane_geometry_summary_free_space_has_no_spatial_chirp() -> None:
    cfg = AbcdefCfg(optics=(FreeSpaceCfg(instance_name="fs", length=100_000.0),))
    result = run_abcdef(
        cfg,
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=100.0,
                center_wavelength_nm=1030.0,
                n_samples=128,
                time_window_fs=2000.0,
            ),
            beam=BeamSpec(radius_mm=0.5, m2=1.0),
        ),
    )
    metrics = summarize_output_plane_geometry(result)

    assert metrics.x_centroid_span_um == pytest.approx(0.0, abs=1e-12)
    assert metrics.x_centroid_slope_um_per_rad_per_fs == pytest.approx(0.0, abs=1e-12)
    assert metrics.x_prime_span == pytest.approx(0.0, abs=1e-12)
    assert metrics.x_prime_slope_per_rad_per_fs == pytest.approx(0.0, abs=1e-12)


def test_single_grating_gap_has_monotonic_spatial_chirp_and_angular_dispersion() -> None:
    cfg = AbcdefCfg(
        optics=(
            GratingCfg(
                instance_name="g1",
                line_density_lpmm=1200.0,
                incidence_angle_deg=35.0,
                diffraction_order=-1,
            ),
            FreeSpaceCfg(instance_name="gap", length=50_000.0),
        )
    )
    result = run_abcdef(
        cfg,
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=100.0,
                center_wavelength_nm=1030.0,
                n_samples=128,
                time_window_fs=2000.0,
            ),
            beam=BeamSpec(radius_mm=0.5, m2=1.0),
        ),
    )
    field = build_output_plane_field_1d(result)
    metrics = summarize_output_plane_geometry(result)

    x_diff = np.diff(field.x_out_um)
    x_prime_diff = np.diff(field.x_prime_out)
    assert metrics.x_centroid_span_um > 0.0
    assert metrics.x_prime_span > 0.0
    assert np.all(x_diff <= 0.0) or np.all(x_diff >= 0.0)
    assert np.all(x_prime_diff <= 0.0) or np.all(x_prime_diff >= 0.0)


def test_treacy_output_plane_summary_exposes_residual_spatial_terms() -> None:
    result = run_abcdef(
        treacy_compressor_preset(length_to_mirror_um=0.0),
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
    full_field = build_output_plane_field_1d(result, phase_variant="full")
    without_phi2_field = build_output_plane_field_1d(result, phase_variant="without_phi2")
    full_summary = summarize_output_plane_field(full_field)
    center_idx = int(np.argmin(np.abs(full_field.delta_omega_rad_per_fs)))

    assert full_summary.x_centroid_span_um > 0.0
    assert full_summary.x_prime_span > 0.0
    assert full_summary.mean_mode_overlap_with_center < 1.0
    assert full_field.x_out_um[center_idx] == pytest.approx(0.0, abs=1e-12)
    assert full_field.x_prime_out[center_idx] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(
        full_field.field_x_omega[:, center_idx],
        without_phi2_field.field_x_omega[:, center_idx],
        rtol=1e-9,
        atol=1e-9,
    )
