from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.physics.abcd.gaussian import q_from_waist
from abcdef_sim.physics.abcd.lenses import (
    DoubletAssembly,
    SellmeierMaterial,
    ThickLensSpec,
    doublet_matrix,
    sample_doublet_beam_radius_profile,
)
from abcdef_sim.physics.abcd.matrices import compose, free_space, interface, thick_lens, thin_lens
from abcdef_sim.physics.abcd.ray import Ray, propagate_ray
from abcdef_sim.physics.abcd.raytracing_ref import (
    from_raytracing_matrix,
    propagate_gaussian_beam_raytracing,
    raytracing_compose,
    raytracing_interface,
    raytracing_space,
    raytracing_thick_lens,
    sample_gaussian_beam_radii_raytracing,
)
from abcdef_sim.physics.validation import ObservableComparison

rt = pytest.importorskip("raytracing")

pytestmark = [pytest.mark.physics, pytest.mark.integration]


def _n_lak22() -> SellmeierMaterial:
    return SellmeierMaterial(
        name="N-LAK22",
        b_terms=(1.14229781, 0.535138441, 1.04088385),
        c_terms=(0.00585778594e-6, 0.0198546147e-6, 100.834017e-6),
    )


def _n_sf6() -> SellmeierMaterial:
    return SellmeierMaterial(
        name="N-SF6",
        b_terms=(1.77931763, 0.338149866, 2.08734474),
        c_terms=(0.0133714182e-6, 0.0617533621e-6, 174.01759e-6),
    )


def _reference_doublet() -> DoubletAssembly:
    return DoubletAssembly(
        first=ThickLensSpec(
            refractive_index=_n_lak22(),
            R1=112.88,
            R2=-112.88,
            thickness=6.0,
            n_in=1.0,
            n_out=1.0,
        ),
        second=ThickLensSpec(
            refractive_index=_n_sf6(),
            R1=-112.88,
            R2=-1415.62,
            thickness=4.0,
            n_in=1.0,
            n_out=1.0,
        ),
        gap=0.0,
    )


def _raytracing_doublet_elements(spec: DoubletAssembly, wavelength: float) -> list[object]:
    first_n = spec.first.refractive_index.refractive_index(wavelength)
    second_n = spec.second.refractive_index.refractive_index(wavelength)
    elements: list[object] = [
        raytracing_thick_lens(
            n=float(first_n),
            R1=spec.first.R1,
            R2=spec.first.R2,
            thickness=spec.first.thickness,
            n_in=spec.first.n_in,
            n_out=spec.first.n_out,
        )
    ]
    if spec.gap > 0.0:
        elements.append(raytracing_space(d=spec.gap))
    elements.append(
        raytracing_thick_lens(
            n=float(second_n),
            R1=spec.second.R1,
            R2=spec.second.R2,
            thickness=spec.second.thickness,
            n_in=spec.second.n_in,
            n_out=spec.second.n_out,
        )
    )
    return elements


def test_free_space_matches_raytracing_space() -> None:
    d = 100.0
    local = free_space(d)
    oracle = from_raytracing_matrix(rt.Space(d=d))

    np.testing.assert_allclose(local, oracle, rtol=1e-12, atol=1e-12)


def test_thin_lens_matches_raytracing_lens() -> None:
    f = 250.0
    local = thin_lens(f)
    oracle = from_raytracing_matrix(rt.Lens(f=f))

    np.testing.assert_allclose(local, oracle, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize(
    ("n1", "n2", "radius", "rt_radius"),
    [
        (1.0, 1.5, 80.0, 80.0),
        (1.5, 1.0, None, np.inf),
    ],
)
def test_interface_matches_raytracing_dielectric_interface(
    n1: float,
    n2: float,
    radius: float | None,
    rt_radius: float,
) -> None:
    local = interface(n1=n1, n2=n2, R=radius)
    oracle = from_raytracing_matrix(rt.DielectricInterface(n1=n1, n2=n2, R=rt_radius))

    np.testing.assert_allclose(local, oracle, rtol=1e-12, atol=1e-12)


def test_thick_lens_matches_raytracing_thick_lens_deterministic_case() -> None:
    n = 1.5
    R1 = 4.0
    R2 = 6.0
    thickness = 3.0

    local = thick_lens(n_lens=n, R1=R1, R2=R2, thickness=thickness)
    oracle = from_raytracing_matrix(rt.ThickLens(n=n, R1=R1, R2=R2, thickness=thickness))
    expected = np.array(
        [
            [0.75, 2.0],
            [-0.0625, 1.1666666666666667],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(local, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(local, oracle, rtol=1e-12, atol=1e-12)


def test_thick_lens_matches_raytracing_biconvex_sign_convention_case() -> None:
    n = 1.5
    R1 = 50.0
    R2 = -50.0
    thickness = 10.0

    local = thick_lens(n_lens=n, R1=R1, R2=R2, thickness=thickness)
    oracle = from_raytracing_matrix(rt.ThickLens(n=n, R1=R1, R2=R2, thickness=thickness))

    np.testing.assert_allclose(local, oracle, rtol=1e-12, atol=1e-12)


def test_thick_lens_with_planar_face_matches_raytracing_primitive_composition() -> None:
    local = thick_lens(n_lens=1.62, R1=None, R2=-75.0, thickness=5.0)
    oracle = from_raytracing_matrix(raytracing_thick_lens(n=1.62, R1=None, R2=-75.0, thickness=5.0))

    np.testing.assert_allclose(local, oracle, rtol=1e-12, atol=1e-12)


@pytest.mark.slow
def test_thick_lens_ray_propagation_matches_raytracing() -> None:
    n = 1.5
    R1 = 50.0
    R2 = -50.0
    thickness = 10.0

    matrix = thick_lens(n_lens=n, R1=R1, R2=R2, thickness=thickness)
    oracle_lens = rt.ThickLens(n=n, R1=R1, R2=R2, thickness=thickness)

    rays = [
        Ray(y=0.0, theta=0.01),
        Ray(y=1.2, theta=-0.02),
        Ray(y=-0.3, theta=0.005),
    ]
    for ray in rays:
        local = propagate_ray(ray, matrix)
        oracle = oracle_lens * rt.Ray(y=ray.y, theta=ray.theta)

        np.testing.assert_allclose(
            [local.y, local.theta],
            [oracle.y, oracle.theta],
            rtol=1e-12,
            atol=1e-12,
        )


def test_compose_order_matches_raytracing_matrix_multiplication() -> None:
    m1 = free_space(50.0)
    m2 = thin_lens(125.0)
    m3 = interface(1.0, 1.5, 100.0)

    local = compose(m1, m2, m3)
    rt_total = raytracing_compose(
        raytracing_space(d=50.0),
        rt.Lens(f=125.0),
        raytracing_interface(n1=1.0, n2=1.5, R=100.0),
    )

    np.testing.assert_allclose(local, from_raytracing_matrix(rt_total), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("wavelength", [0.000810, 0.001550])
def test_doublet_matrix_matches_raytracing_primitive_composition(wavelength: float) -> None:
    spec = _reference_doublet()

    local = doublet_matrix(spec, wavelength=wavelength)
    oracle = from_raytracing_matrix(
        raytracing_compose(*_raytracing_doublet_elements(spec, wavelength))
    )

    np.testing.assert_allclose(local, oracle, rtol=1e-12, atol=1e-12)


@pytest.mark.slow
@pytest.mark.parametrize("wavelength", [0.000810, 0.001550])
def test_doublet_gaussian_beam_profile_matches_raytracing(wavelength: float) -> None:
    spec = _reference_doublet()
    q_in = q_from_waist(waist_radius=1.024, wavelength=wavelength, distance_from_waist=0.0)
    z_samples = np.linspace(0.0, 250.0, 33, dtype=float)

    local = sample_doublet_beam_radius_profile(q_in, spec, wavelength, z_samples)
    oracle_beam = propagate_gaussian_beam_raytracing(
        q_in,
        wavelength,
        _raytracing_doublet_elements(spec, wavelength),
    )
    oracle = sample_gaussian_beam_radii_raytracing(oracle_beam, z_samples)
    comparison = ObservableComparison(
        name=f"Doublet beam profile ({wavelength:.3f})",
        observable_label="beam radius",
        coordinates=z_samples,
        coordinate_label="z",
        local=local,
        reference=oracle,
    )

    np.testing.assert_allclose(comparison.local, comparison.reference, rtol=1e-9, atol=1e-12)
    assert comparison.max_abs_error() <= 1e-9
