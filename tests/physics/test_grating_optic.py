from __future__ import annotations

import math

import numpy as np
import pytest

from abcdef_sim.optics.grating import (
    Grating,
    _grating_f_exact_local,
)

pytestmark = pytest.mark.physics

_C_UM_PER_FS = 0.299792458


def test_grating_matrix_matches_first_order_grating_oracle_at_omega0_slope() -> None:
    omega = np.array([1.65, 1.75, 1.85], dtype=float)
    omega0 = 1.75
    grating = Grating(
        line_density_lpmm=600.0,
        incidence_angle_deg=25.0,
        diffraction_order=-1,
        immersion_refractive_index=1.3,
    )

    matrix = grating.matrix(omega, omega0=omega0)
    d_um = 1000.0 / 600.0
    theta_i_rad = math.radians(25.0)
    lambda_um = (2.0 * np.pi * _C_UM_PER_FS) / omega
    theta_d_rad = np.arcsin(lambda_um / d_um - np.sin(theta_i_rad))
    expected_a = np.cos(theta_d_rad) / np.cos(theta_i_rad)
    expected_d = np.cos(theta_i_rad) / np.cos(theta_d_rad)
    expected_slope = (2.0 * np.pi * _C_UM_PER_FS * 1.3**2) / (
        omega0**2 * d_um * math.cos(float(theta_d_rad[1]))
    )
    observed_slope = (matrix[2, 1, 2] - matrix[0, 1, 2]) / (omega[2] - omega[0])

    np.testing.assert_allclose(matrix[:, 0, 0], expected_a, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(matrix[:, 1, 1], expected_d, rtol=1e-12, atol=1e-12)
    assert observed_slope == pytest.approx(expected_slope, rel=5e-3, abs=0.0)


def test_grating_f_matches_exact_local_shift() -> None:
    omega0 = 1.827
    omega = omega0 + np.linspace(-0.18, 0.18, 13, dtype=float)
    period_um = 1000.0 / 1200.0
    incidence_angle_rad = math.radians(35.0)
    grating = Grating(
        line_density_lpmm=1200.0,
        incidence_angle_deg=35.0,
        diffraction_order=-1,
        immersion_refractive_index=1.0,
    )

    matrix = grating.matrix(omega, omega0=omega0)
    exact_f = _grating_f_exact_local(
        omega,
        omega_ref=omega0,
        period_um=period_um,
        incidence_angle_rad=incidence_angle_rad,
        diffraction_order=-1.0,
        immersion_refractive_index=1.0,
    )
    theta_d = np.arcsin(
        ((2.0 * np.pi * _C_UM_PER_FS) / omega) / period_um - math.sin(incidence_angle_rad)
    )
    first_order_f = (
        (2.0 * np.pi * _C_UM_PER_FS) / (omega0**2 * period_um * math.cos(float(theta_d[6])))
    ) * (omega - omega0)

    exact_error = np.sqrt(np.mean((matrix[:, 1, 2] - exact_f) ** 2))
    first_order_error = np.sqrt(np.mean((first_order_f - exact_f) ** 2))

    assert exact_error < 1e-12
    assert first_order_error > 1e-2


def test_grating_abcd_block_preserves_unit_determinant() -> None:
    omega = np.linspace(1.6, 1.9, 11, dtype=float)
    matrix = Grating(line_density_lpmm=900.0, incidence_angle_deg=20.0).matrix(omega)

    np.testing.assert_allclose(matrix[:, 0, 0] * matrix[:, 1, 1], 1.0, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(matrix[:, 0, 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(matrix[:, 1, 0], 0.0, atol=1e-12)


def test_grating_frequency_shift_is_zero_at_omega0_and_changes_sign() -> None:
    omega = np.array([1.8, 1.85, 1.9], dtype=float)
    matrix = Grating().matrix(omega, omega0=1.85)

    assert matrix[1, 1, 2] == pytest.approx(0.0, abs=1e-12)
    assert matrix[0, 1, 2] < 0.0
    assert matrix[2, 1, 2] > 0.0


def test_grating_defaults_omega0_to_mean_frequency() -> None:
    omega = np.array([1.71, 1.79, 1.88, 1.93], dtype=float)
    grating = Grating(line_density_lpmm=750.0, incidence_angle_deg=18.0)

    implicit = grating.matrix(omega)
    explicit = grating.matrix(omega, omega0=float(np.mean(omega)))

    np.testing.assert_allclose(implicit, explicit, rtol=1e-12, atol=1e-12)


def test_grating_immersion_index_scales_only_the_f_term() -> None:
    omega = np.array([1.72, 1.79, 1.86], dtype=float)
    matrix_air = Grating(immersion_refractive_index=1.0).matrix(omega, omega0=1.79)
    matrix_medium = Grating(immersion_refractive_index=1.5).matrix(omega, omega0=1.79)

    np.testing.assert_allclose(matrix_air[:, :2, :2], matrix_medium[:, :2, :2], atol=1e-12)
    np.testing.assert_allclose(
        matrix_medium[:, 1, 2],
        matrix_air[:, 1, 2] * 1.5**2,
        rtol=1e-12,
        atol=1e-12,
    )


def test_grating_invalid_geometry_raises() -> None:
    wavelength_um = 1.03
    omega = np.array([(2.0 * np.pi * _C_UM_PER_FS) / wavelength_um], dtype=float)

    with pytest.raises(ValueError, match="invalid diffraction angle"):
        Grating(line_density_lpmm=1200.0, incidence_angle_deg=0.0).matrix(omega)


def test_grating_grazing_solution_raises_singularity_error() -> None:
    wavelength_um = 1.03
    omega = np.array([(2.0 * np.pi * _C_UM_PER_FS) / wavelength_um], dtype=float)
    incidence_angle_deg = math.degrees(math.asin(0.03))

    with pytest.raises(ValueError, match="singular"):
        Grating(line_density_lpmm=1000.0, incidence_angle_deg=incidence_angle_deg).matrix(omega)
