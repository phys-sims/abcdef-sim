from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from abcdef_sim.physics.abcdef.treacy import (
    compute_treacy_analytic_metrics,
    phase_from_treacy_dispersion,
)

pytestmark = pytest.mark.physics

RTOL_GDD_TOD = 5e-3
ATOL_GDD = 5e3
ATOL_TOD = 1e4
ATOL_ANGLE_DEG = 0.05
_C_UM_PER_FS = 0.299792458


def _separation_um(case: dict[str, float | int | str | None]) -> float:
    if case.get("separation_um") is not None:
        return float(case["separation_um"])
    if case.get("separation_mm") is not None:
        return float(case["separation_mm"]) * 1e3
    if case.get("separation_cm") is not None:
        return float(case["separation_cm"]) * 1e4
    if case.get("separation_m") is not None:
        return float(case["separation_m"]) * 1e6
    raise KeyError("Golden case must specify separation_um/mm/cm/m.")


def _expected_with_units(
    case: dict[str, float | int | str | None],
    *,
    ps_key: str,
    conversion: float,
) -> float | None:
    if case.get(ps_key) is None:
        return None

    expected = float(case[ps_key]) * conversion
    expected_assumes_n_passes = float(case.get("expect_assumes_n_passes", 2))
    case_n_passes = float(case["n_passes"])
    return expected * (case_n_passes / expected_assumes_n_passes)


def _finite_d2_d3_at_zero_from_phase(*, gdd_fs2: float, tod_fs3: float) -> tuple[float, float]:
    step_rad_per_fs = 1e-4
    delta_omega = step_rad_per_fs * np.array([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float64)
    phase = phase_from_treacy_dispersion(delta_omega, gdd_fs2=gdd_fs2, tod_fs3=tod_fs3)

    d2phi = (-phase[4] + 16.0 * phase[3] - 30.0 * phase[2] + 16.0 * phase[1] - phase[0]) / (
        12.0 * step_rad_per_fs**2
    )
    d3phi = (-phase[0] + 2.0 * phase[1] - 2.0 * phase[3] + phase[4]) / (2.0 * step_rad_per_fs**3)
    return float(d2phi), float(d3phi)


def _wavelength_nm_from_omega(omega_rad_per_fs: float) -> float:
    return 1e3 * ((2.0 * np.pi * _C_UM_PER_FS) / float(omega_rad_per_fs))


def _finite_tod_from_gdd_frequency_derivative(
    *,
    line_density_lpmm: float,
    incidence_angle_deg: float,
    separation_um: float,
    wavelength_nm: float,
    diffraction_order: int,
    n_passes: int,
    step_rad_per_fs: float = 1e-5,
) -> float:
    center = compute_treacy_analytic_metrics(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        wavelength_nm=wavelength_nm,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    omega0 = center.omega0_rad_per_fs
    lower = compute_treacy_analytic_metrics(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        wavelength_nm=_wavelength_nm_from_omega(omega0 - step_rad_per_fs),
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    upper = compute_treacy_analytic_metrics(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        wavelength_nm=_wavelength_nm_from_omega(omega0 + step_rad_per_fs),
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    return float((upper.gdd_fs2 - lower.gdd_fs2) / (2.0 * step_rad_per_fs))


@pytest.mark.parametrize(
    ("gdd_fs2", "tod_fs3"),
    [
        (2_500.0, -1_000.0),
        (-4_200.0, 8_000.0),
    ],
)
def test_phase_polynomial_derivatives_match_coefficients(gdd_fs2: float, tod_fs3: float) -> None:
    d2phi, d3phi = _finite_d2_d3_at_zero_from_phase(gdd_fs2=gdd_fs2, tod_fs3=tod_fs3)
    assert d2phi == pytest.approx(gdd_fs2, rel=1e-9, abs=1e-6)
    assert d3phi == pytest.approx(tod_fs3, rel=1e-9, abs=1e-6)


def test_treacy_metric_coefficients_match_phase_derivatives_near_zero() -> None:
    metrics = compute_treacy_analytic_metrics()
    d2phi, d3phi = _finite_d2_d3_at_zero_from_phase(
        gdd_fs2=metrics.gdd_fs2,
        tod_fs3=metrics.tod_fs3,
    )
    assert d2phi == pytest.approx(metrics.gdd_fs2, rel=1e-9, abs=1e-3)
    assert d3phi == pytest.approx(metrics.tod_fs3, rel=1e-9, abs=1e-3)


def test_treacy_canonical_geometry_matches_expected_gdd_tod() -> None:
    metrics = compute_treacy_analytic_metrics(
        line_density_lpmm=1200.0,
        incidence_angle_deg=35.0,
        separation_um=100_000.0,
        wavelength_nm=1030.0,
        diffraction_order=-1,
        n_passes=2,
    )

    assert metrics.gdd_fs2 == pytest.approx(-1.33e6, abs=5e3)
    assert metrics.tod_fs3 == pytest.approx(5.35e6, abs=1e4)
    assert metrics.gdd_fs2 < 0.0
    assert metrics.tod_fs3 > 0.0


def test_treacy_invalid_geometry_raises() -> None:
    with pytest.raises(ValueError, match="asin domain"):
        compute_treacy_analytic_metrics(
            line_density_lpmm=1200.0,
            incidence_angle_deg=35.0,
            separation_um=100_000.0,
            wavelength_nm=1030.0,
            diffraction_order=-3,
        )


def test_treacy_tod_matches_gdd_frequency_derivative_for_nondefault_order() -> None:
    metrics = compute_treacy_analytic_metrics(
        line_density_lpmm=300.0,
        incidence_angle_deg=10.0,
        separation_um=100_000.0,
        wavelength_nm=800.0,
        diffraction_order=1,
        n_passes=2,
    )
    finite_tod = _finite_tod_from_gdd_frequency_derivative(
        line_density_lpmm=300.0,
        incidence_angle_deg=10.0,
        separation_um=100_000.0,
        wavelength_nm=800.0,
        diffraction_order=1,
        n_passes=2,
    )

    assert finite_tod == pytest.approx(metrics.tod_fs3, rel=5e-5, abs=50.0)


def test_treacy_matches_golden_fixture_when_expectations_present() -> None:
    fixture_path = Path("tests/fixtures/treacy_grating_pair_golden.json")
    cases = json.loads(fixture_path.read_text())

    for case in cases:
        metrics = compute_treacy_analytic_metrics(
            line_density_lpmm=float(case["line_density_lpmm"]),
            incidence_angle_deg=float(case["incidence_angle_deg"]),
            separation_um=_separation_um(case),
            wavelength_nm=float(case["wavelength_nm"]),
            diffraction_order=int(case["diffraction_order"]),
            n_passes=int(case["n_passes"]),
        )
        expected_gdd_fs2 = _expected_with_units(
            case,
            ps_key="expect_gdd_ps2",
            conversion=1e6,
        )
        if expected_gdd_fs2 is not None:
            assert metrics.gdd_fs2 == pytest.approx(
                expected_gdd_fs2,
                rel=RTOL_GDD_TOD,
                abs=ATOL_GDD,
            )
        expected_tod_fs3 = _expected_with_units(
            case,
            ps_key="expect_tod_ps3",
            conversion=1e9,
        )
        if expected_tod_fs3 is not None:
            assert metrics.tod_fs3 == pytest.approx(
                expected_tod_fs3,
                rel=RTOL_GDD_TOD,
                abs=ATOL_TOD,
            )
        if case["expect_diffraction_angle_deg"] is not None:
            assert abs(metrics.diffraction_angle_deg) == pytest.approx(
                abs(float(case["expect_diffraction_angle_deg"])),
                abs=ATOL_ANGLE_DEG,
            )
