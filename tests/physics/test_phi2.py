from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.physics.abcdef.phase_terms import phi2_rad

pytestmark = pytest.mark.physics


def test_phi2_rad_matches_martinez_equation_26() -> None:
    k = np.array([2.0, 4.0, 6.0], dtype=float)
    ray_in = np.array(
        [
            [[1.0], [0.2], [1.0]],
            [[-0.5], [0.4], [1.0]],
            [[0.25], [-0.6], [1.0]],
        ],
        dtype=float,
    )
    ray_out = np.array(
        [
            [[0.7], [0.1], [1.0]],
            [[-0.3], [0.2], [1.0]],
            [[0.5], [-0.1], [1.0]],
        ],
        dtype=float,
    )

    expected = 0.5 * k * (ray_in[:, 0, 0] * ray_in[:, 1, 0] - ray_out[:, 0, 0] * ray_out[:, 1, 0])

    np.testing.assert_allclose(phi2_rad(k, ray_in, ray_out), expected)
