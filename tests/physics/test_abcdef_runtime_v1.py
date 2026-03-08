from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.standalone import BeamSpec, PulseSpec, StandaloneLaserSpec
from abcdef_sim.data_models.states import RayState
from abcdef_sim.optics.grating import Grating
from abcdef_sim.optics.thick_lens import ThickLens
from abcdef_sim.physics.abcd.lenses import (
    SellmeierMaterial,
    ThickLensSpec,
    thick_lens_matrix_for_spec,
)
from abcdef_sim.physics.abcd.matrices import free_space, thick_lens
from abcdef_sim.physics.abcd.raytracing_validation import (
    abcdef_runtime_doublet_effective_focal_length_comparison,
    abcdef_runtime_doublet_ray_output_comparisons,
    abcdef_runtime_single_lens_effective_focal_length_comparison,
    run_abcdef_runtime_wavelength_tracking_benchmarks,
    single_lens_batched_ray_output_comparisons,
    single_lens_wavelength_grid_um,
)
from abcdef_sim.physics.abcdef.conventions import theta_abcd_to_xprime_abcdef
from abcdef_sim.physics.abcdef.dispersion import evaluate_phase_polynomial, fit_phase_taylor
from abcdef_sim.physics.abcdef.propagation import propagate_step
from abcdef_sim.physics.abcdef.pulse import (
    build_standalone_laser_state,
    update_beam_state_from_abcd,
)

pytestmark = pytest.mark.physics

_C_UM_PER_FS = 0.299792458


def _omega_to_wavelength_um(omega: float) -> float:
    return (2.0 * np.pi * _C_UM_PER_FS) / float(omega)


def _theta_ray_to_martinez_column(*, x: float, theta: float, n: float) -> np.ndarray:
    return np.array([[[x], [n * theta], [1.0]]], dtype=float)


def test_grating_matrix_frequency_shift_is_centered_at_omega0() -> None:
    omega0 = 1.85
    omega = np.array([1.8, 1.85, 1.9], dtype=float)
    matrix = Grating().matrix(omega, omega0=omega0)

    np.testing.assert_allclose(matrix[:, 0, 1], 0.0)
    np.testing.assert_allclose(matrix[:, 1, 0], 0.0)
    assert matrix[1, 1, 2] == pytest.approx(0.0, abs=1e-12)
    assert matrix[0, 1, 2] < 0.0
    assert matrix[2, 1, 2] > 0.0


def test_thick_lens_runtime_matches_abcd_thick_lens_over_frequency_grid() -> None:
    omega = np.linspace(1.76, 1.94, 7, dtype=float)
    material = SellmeierMaterial(
        name="demo",
        b_terms=(1.0, 0.2, 0.1),
        c_terms=(0.01, 0.04, 120.0),
    )
    lens = ThickLens(
        name="ThickLens",
        instance_name="lens-1",
        _length=6.0,
        R1=85.0,
        R2=-55.0,
        n_in=1.0,
        n_out=1.33,
        refractive_index_model=material,
    )
    spec = ThickLensSpec(
        refractive_index=material,
        R1=85.0,
        R2=-55.0,
        thickness=6.0,
        n_in=1.0,
        n_out=1.33,
    )

    expected = np.stack(
        [
            theta_abcd_to_xprime_abcdef(
                thick_lens_matrix_for_spec(spec, wavelength=_omega_to_wavelength_um(omega_i)),
                n_in=spec.n_in,
                n_out=spec.n_out,
            )
            for omega_i in omega
        ],
        axis=0,
    )

    np.testing.assert_allclose(lens.matrix(omega), expected, rtol=1e-12, atol=1e-12)


def test_thick_lens_runtime_matches_abcd_thick_lens_ray_propagation() -> None:
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
    state_in = RayState(
        rays=_theta_ray_to_martinez_column(x=0.7, theta=0.015, n=1.0),
        system=np.eye(3, dtype=float)[None, ...],
        meta={},
    )

    state_out = propagate_step(state_in, lens.matrix(omega))
    expected_theta_ray = theta_matrix @ np.array([0.7, 0.015], dtype=float)

    assert state_out.rays[0, 2, 0] == pytest.approx(1.0, abs=1e-12)
    assert state_out.rays[0, 0, 0] == pytest.approx(expected_theta_ray[0], rel=1e-12, abs=1e-12)
    assert state_out.rays[0, 1, 0] / 1.33 == pytest.approx(
        expected_theta_ray[1],
        rel=1e-12,
        abs=1e-12,
    )
    np.testing.assert_allclose(
        state_out.system[0],
        theta_abcd_to_xprime_abcdef(theta_matrix, n_in=1.0, n_out=1.33),
        rtol=1e-12,
        atol=1e-12,
    )


def test_thick_lens_runtime_effective_focal_length_matches_raytracing() -> None:
    pytest.importorskip("raytracing")

    comparison = abcdef_runtime_single_lens_effective_focal_length_comparison(
        single_lens_wavelength_grid_um(25)
    )

    np.testing.assert_allclose(comparison.local, comparison.reference, rtol=1e-10, atol=1e-10)
    assert comparison.max_abs_error() <= 1e-10


def test_thick_lens_runtime_batched_ray_outputs_match_raytracing() -> None:
    pytest.importorskip("raytracing")

    x_out, theta_out = single_lens_batched_ray_output_comparisons(
        single_lens_wavelength_grid_um(19)
    )

    np.testing.assert_allclose(x_out.local, x_out.reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(theta_out.local, theta_out.reference, rtol=1e-12, atol=1e-12)


def test_doublet_runtime_effective_focal_length_matches_raytracing() -> None:
    pytest.importorskip("raytracing")

    comparison = abcdef_runtime_doublet_effective_focal_length_comparison(
        single_lens_wavelength_grid_um(25)
    )

    np.testing.assert_allclose(comparison.local, comparison.reference, rtol=1e-10, atol=1e-10)
    assert comparison.max_abs_error() <= 1e-10


def test_doublet_runtime_batched_ray_outputs_match_raytracing() -> None:
    pytest.importorskip("raytracing")

    x_out, theta_out = abcdef_runtime_doublet_ray_output_comparisons(
        single_lens_wavelength_grid_um(19)
    )

    np.testing.assert_allclose(x_out.local, x_out.reference, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(theta_out.local, theta_out.reference, rtol=1e-12, atol=1e-12)


def test_runtime_benchmark_runner_returns_runtime_scenarios() -> None:
    pytest.importorskip("raytracing")

    comparisons = run_abcdef_runtime_wavelength_tracking_benchmarks(
        [1],
        warmup_runs=0,
        measured_runs=1,
    )

    assert len(comparisons) == 2
    assert all(comparison.scenario_name.startswith("ABCDEF runtime") for comparison in comparisons)
    assert all(comparison.local_seconds > 0.0 for comparison in comparisons)
    assert all(comparison.reference_seconds > 0.0 for comparison in comparisons)


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
