from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.physics.abcd.gaussian import q_propagate
from abcdef_sim.physics.abcd.matrices import compose, free_space, thin_lens
from abcdef_sim.physics.abcd.ray import Ray, propagate_ray


def test_compose_respects_traversal_order_for_ray_propagation() -> None:
    ray = Ray(y=1.0, theta=0.01)

    system = compose(free_space(100.0), thin_lens(250.0))
    propagated = propagate_ray(ray, system)

    # y' = y + L*theta = 2.0; theta' = theta - y'/f = 0.002
    np.testing.assert_allclose([propagated.y, propagated.theta], [2.0, 0.002], atol=1e-12, rtol=1e-12)


def test_q_parameter_free_space_adds_distance() -> None:
    q_in = 1.0 + 2.0j

    q_out = q_propagate(q_in, free_space(30.0))

    assert q_out == pytest.approx(31.0 + 2.0j)


def test_thin_lens_rejects_zero_focal_length() -> None:
    with pytest.raises(ValueError, match="focal_length must be non-zero"):
        thin_lens(0.0)
