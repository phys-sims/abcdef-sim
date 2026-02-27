from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.adapters.raytracing_ref import from_raytracing_matrix, trace_ray_raytracing
from abcdef_sim.physics.abcd.matrices import compose, free_space, thin_lens
from abcdef_sim.physics.abcd.ray import Ray, propagate_ray


def _rt() -> object:
    return pytest.importorskip("raytracing")


def test_matrix_equivalence_free_space_and_lens() -> None:
    rt = _rt()
    m_free = free_space(100.0)
    m_lens = thin_lens(250.0)

    local = compose(m_free, m_lens)
    rt_total = rt.Matrix(A=1.0, B=0.0, C=0.0, D=1.0)
    rt_total = rt.Matrix(A=m_lens[0, 0], B=m_lens[0, 1], C=m_lens[1, 0], D=m_lens[1, 1]) * rt_total
    rt_total = rt.Matrix(A=m_free[0, 0], B=m_free[0, 1], C=m_free[1, 0], D=m_free[1, 1]) * rt_total

    np.testing.assert_allclose(local, from_raytracing_matrix(rt_total), rtol=1e-10, atol=1e-12)


def test_ray_propagation_equivalence() -> None:
    _rt()
    elements = [free_space(50.0), thin_lens(125.0), free_space(30.0)]
    total = compose(*elements)

    rays = [Ray(y=0.0, theta=0.01), Ray(y=1.2, theta=-0.02), Ray(y=-0.3, theta=0.005)]
    for ray in rays:
        local = propagate_ray(ray, total)
        ref = trace_ray_raytracing(ray, elements)
        np.testing.assert_allclose(
            [local.y, local.theta],
            [ref.y, ref.theta],
            rtol=1e-9,
            atol=1e-12,
        )
