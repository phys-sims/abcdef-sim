from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.standalone import BeamSpec, PulseSpec, StandaloneLaserSpec
from abcdef_sim.optics.grating import Grating
from abcdef_sim.optics.thick_lens import ThickLens
from abcdef_sim.physics.abcd.matrices import free_space, thick_lens
from abcdef_sim.physics.abcdef.dispersion import evaluate_phase_polynomial, fit_phase_taylor
from abcdef_sim.physics.abcdef.pulse import (
    build_standalone_laser_state,
    update_beam_state_from_abcd,
)

pytestmark = pytest.mark.physics


def test_grating_matrix_frequency_shift_is_centered_at_omega0() -> None:
    omega0 = 1.85
    omega = np.array([1.8, 1.85, 1.9], dtype=float)
    matrix = Grating().matrix(omega, omega0=omega0)

    np.testing.assert_allclose(matrix[:, 0, 1], 0.0)
    np.testing.assert_allclose(matrix[:, 1, 0], 0.0)
    assert matrix[1, 1, 2] == pytest.approx(0.0, abs=1e-12)
    assert matrix[0, 1, 2] < 0.0
    assert matrix[2, 1, 2] > 0.0


def test_thick_lens_runtime_respects_xprime_conversion() -> None:
    omega = np.array([1.82], dtype=float)
    lens = ThickLens(
        name="ThickLens",
        instance_name="lens-1",
        _length=5.0,
        R1=80.0,
        R2=-60.0,
        n_in=1.0,
        n_out=1.33,
        refractive_index_model=1.5,
    )

    theta_matrix = thick_lens(
        n_lens=1.5,
        R1=80.0,
        R2=-60.0,
        thickness=5.0,
        n_in=1.0,
        n_out=1.33,
    )
    expected = np.eye(3, dtype=float)
    expected[0, 0] = theta_matrix[0, 0]
    expected[0, 1] = theta_matrix[0, 1]
    expected[1, 0] = 1.33 * theta_matrix[1, 0]
    expected[1, 1] = 1.33 * theta_matrix[1, 1]

    np.testing.assert_allclose(lens.matrix(omega)[0], expected)


def test_weighted_taylor_fit_recovers_known_coefficients() -> None:
    delta_omega = np.linspace(-0.3, 0.3, 9, dtype=float)
    coefficients = np.array([0.2, -0.1, 500.0, -1200.0, 3500.0], dtype=float)
    phi = evaluate_phase_polynomial(delta_omega, coefficients)
    weights = np.array([1.0, 2.0, 4.0, 8.0, 10.0, 8.0, 4.0, 2.0, 1.0], dtype=float)

    fit = fit_phase_taylor(
        delta_omega,
        phi,
        omega0_rad_per_fs=1.83,
        weights=weights,
        order=4,
    )

    np.testing.assert_allclose(fit.coefficients_rad[:5], coefficients, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit.phi_fit_rad, phi, rtol=1e-10, atol=1e-10)
    assert fit.weighted_rms_rad == pytest.approx(0.0, abs=1e-12)


def test_beam_update_preserves_m2_under_waist_assumption() -> None:
    state = build_standalone_laser_state(
        StandaloneLaserSpec(
            pulse=PulseSpec(width_fs=80.0, n_samples=128, time_window_fs=800.0),
            beam=BeamSpec(radius_mm=1.0, m2=1.3),
        )
    )

    out = update_beam_state_from_abcd(state, free_space(50_000.0))

    assert out.beam.m2 == pytest.approx(1.3)
    assert out.beam.radius_mm > state.beam.radius_mm
