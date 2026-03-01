from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.adapters.raytracing_ref import from_raytracing_matrix
from abcdef_sim.physics.abcd.matrices import compose, free_space, interface, thick_lens, thin_lens
from abcdef_sim.physics.abcd.ray import Ray, propagate_ray

rt = pytest.importorskip("raytracing")


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
    rt_total = rt.DielectricInterface(n1=1.0, n2=1.5, R=100.0) * rt.Lens(f=125.0) * rt.Space(d=50.0)

    np.testing.assert_allclose(local, from_raytracing_matrix(rt_total), rtol=1e-12, atol=1e-12)
