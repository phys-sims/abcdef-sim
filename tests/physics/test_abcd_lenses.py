from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.physics.abcd.gaussian import (
    beam_radius_from_q,
    q_from_waist,
    rayleigh_range_from_waist,
    sample_beam_radius_profile,
    waist_position_from_plane,
    waist_radius_from_q,
)
from abcdef_sim.physics.abcd.lenses import (
    DoubletAssembly,
    SellmeierMaterial,
    ThickLensSpec,
    doublet_matrix,
    refractive_index_sellmeier,
    thick_lens_matrix_for_spec,
)
from abcdef_sim.physics.abcd.matrices import compose, free_space, interface, thick_lens

pytestmark = pytest.mark.physics


def test_interface_none_matches_planar_refraction() -> None:
    local = interface(n1=1.0, n2=1.5, R=None)
    expected = np.array([[1.0, 0.0], [0.0, 1.0 / 1.5]], dtype=float)

    np.testing.assert_allclose(local, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"n_lens": 0.0, "R1": 50.0, "R2": -50.0, "thickness": 10.0}, "n_lens"),
        ({"n_lens": 1.5, "R1": 0.0, "R2": -50.0, "thickness": 10.0}, "R1"),
        ({"n_lens": 1.5, "R1": 50.0, "R2": 0.0, "thickness": 10.0}, "R2"),
        ({"n_lens": 1.5, "R1": 50.0, "R2": -50.0, "thickness": -1.0}, "thickness"),
    ],
)
def test_thick_lens_rejects_invalid_geometry(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        thick_lens(**kwargs)


def test_sellmeier_refractive_index_matches_manual_formula_for_mm_inputs() -> None:
    material = SellmeierMaterial(
        name="N-SF6",
        b_terms=(1.77931763, 0.338149866, 2.08734474),
        c_terms=(
            0.0133714182e-6,
            0.0617533621e-6,
            174.01759e-6,
        ),
    )
    wavelength = 0.810

    local = refractive_index_sellmeier(material, wavelength)
    manual = np.sqrt(
        1.0
        + sum(
            b_term * wavelength**2 / (wavelength**2 - c_term)
            for b_term, c_term in zip(material.b_terms, material.c_terms, strict=True)
        )
    )

    assert local == pytest.approx(manual, rel=1e-12, abs=1e-12)


def test_thick_lens_spec_with_constant_index_matches_low_level_primitive() -> None:
    spec = ThickLensSpec(refractive_index=1.5, R1=50.0, R2=-50.0, thickness=10.0)

    local = thick_lens_matrix_for_spec(spec)
    expected = thick_lens(n_lens=1.5, R1=50.0, R2=-50.0, thickness=10.0)

    np.testing.assert_allclose(local, expected, rtol=1e-12, atol=1e-12)


def test_thick_lens_spec_requires_wavelength_for_sellmeier_material() -> None:
    spec = ThickLensSpec(
        refractive_index=SellmeierMaterial(
            name="demo",
            b_terms=(1.0, 0.2, 0.3),
            c_terms=(1.0e-6, 2.0e-6, 3.0e-6),
        ),
        R1=50.0,
        R2=-50.0,
        thickness=10.0,
    )

    with pytest.raises(ValueError, match="wavelength is required"):
        thick_lens_matrix_for_spec(spec)


def test_doublet_matrix_matches_manual_composition_with_gap() -> None:
    spec = DoubletAssembly(
        first=ThickLensSpec(refractive_index=1.45, R1=60.0, R2=-40.0, thickness=5.0),
        second=ThickLensSpec(refractive_index=1.62, R1=40.0, R2=-100.0, thickness=3.0),
        gap=2.0,
    )

    local = doublet_matrix(spec)
    expected = compose(
        thick_lens(n_lens=1.45, R1=60.0, R2=-40.0, thickness=5.0),
        free_space(2.0),
        thick_lens(n_lens=1.62, R1=40.0, R2=-100.0, thickness=3.0),
    )

    np.testing.assert_allclose(local, expected, rtol=1e-12, atol=1e-12)


def test_doublet_rejects_negative_gap() -> None:
    with pytest.raises(ValueError, match="gap"):
        DoubletAssembly(
            first=ThickLensSpec(refractive_index=1.45, R1=60.0, R2=-40.0, thickness=5.0),
            second=ThickLensSpec(refractive_index=1.62, R1=40.0, R2=-100.0, thickness=3.0),
            gap=-0.1,
        )


def test_q_helpers_round_trip_waist_and_position() -> None:
    wavelength = 0.00103
    waist_radius = 0.25
    distance = -30.0

    q = q_from_waist(waist_radius=waist_radius, wavelength=wavelength, distance_from_waist=distance)

    assert waist_radius_from_q(q, wavelength) == pytest.approx(waist_radius, rel=1e-12, abs=1e-12)
    assert waist_position_from_plane(q) == pytest.approx(30.0, rel=1e-12, abs=1e-12)


def test_beam_radius_from_q_matches_analytic_profile() -> None:
    wavelength = 0.00081
    waist_radius = 0.12
    distance = 25.0
    q = q_from_waist(waist_radius=waist_radius, wavelength=wavelength, distance_from_waist=distance)
    z_r = rayleigh_range_from_waist(waist_radius, wavelength)
    expected = waist_radius * np.sqrt(1.0 + (distance / z_r) ** 2)

    assert beam_radius_from_q(q, wavelength) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_sample_beam_radius_profile_matches_closed_form() -> None:
    wavelength = 0.00155
    waist_radius = 0.18
    q_at_plane = q_from_waist(
        waist_radius=waist_radius,
        wavelength=wavelength,
        distance_from_waist=-10.0,
    )
    z_samples = np.array([0.0, 25.0, 50.0], dtype=float)

    local = sample_beam_radius_profile(q_at_plane, wavelength, z_samples)
    q_samples = z_samples + q_at_plane
    z_r = np.imag(q_at_plane)
    expected = waist_radius * np.sqrt(1.0 + (np.real(q_samples) / z_r) ** 2)

    np.testing.assert_allclose(local, expected, rtol=1e-12, atol=1e-12)
