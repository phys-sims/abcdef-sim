from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim import BeamSpec, PulseSpec, StandaloneLaserSpec, run_abcdef, treacy_compressor_preset
from abcdef_sim.physics.abcdef.dispersion import (
    evaluate_phase_polynomial,
    fit_phase_taylor,
    gdd_from_phase_coeffs,
    tod_from_phase_coeffs,
)

pytestmark = pytest.mark.physics


def test_fit_phase_taylor_recovers_gdd_tod_with_large_constant_and_linear_terms() -> None:
    delta_omega = np.linspace(-0.15, 0.15, 513, dtype=np.float64)
    coefficients = np.array(
        [
            7.5e8,
            -3.2e9,
            -1.3255990136e6,
            5.3471183062e6,
            2.5e7,
        ],
        dtype=np.float64,
    )
    phase = evaluate_phase_polynomial(delta_omega, coefficients)
    weights = np.exp(-0.5 * (delta_omega / 0.045) ** 2)

    fit = fit_phase_taylor(
        delta_omega,
        phase,
        omega0_rad_per_fs=1.0,
        weights=weights,
        order=4,
    )

    np.testing.assert_allclose(fit.coefficients_rad[:5], coefficients, rtol=5e-9, atol=0.1)
    assert gdd_from_phase_coeffs(fit.coefficients_rad) == pytest.approx(coefficients[2], abs=1e-3)
    assert tod_from_phase_coeffs(fit.coefficients_rad) == pytest.approx(coefficients[3], abs=1e-2)


def test_treacy_gdd_tod_extraction_is_invariant_to_large_affine_phase_offsets() -> None:
    result = run_abcdef(
        treacy_compressor_preset(length_to_mirror_um=0.0),
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=100.0,
                center_wavelength_nm=1030.0,
                n_samples=256,
                time_window_fs=3000.0,
            ),
            beam=BeamSpec(radius_mm=1.0, m2=1.0),
        ),
    )
    delta_omega = np.asarray(result.pipeline_result.delta_omega_rad_per_fs, dtype=np.float64)
    phase = np.asarray(result.pipeline_result.phi_total_rad, dtype=np.float64)
    affine_offset = 8.2e8 - 2.4e9 * delta_omega

    shifted_fit = fit_phase_taylor(
        delta_omega,
        phase + affine_offset,
        omega0_rad_per_fs=result.pipeline_result.omega0_rad_per_fs,
        weights=result.fit.weights,
        order=4,
    )

    assert gdd_from_phase_coeffs(shifted_fit.coefficients_rad) == pytest.approx(
        result.final_state.metrics["abcdef.gdd_fs2"],
        rel=0.0,
        abs=1e-3,
    )
    assert tod_from_phase_coeffs(shifted_fit.coefficients_rad) == pytest.approx(
        result.final_state.metrics["abcdef.tod_fs3"],
        rel=0.0,
        abs=2e-2,
    )
