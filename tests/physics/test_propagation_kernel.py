from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.states import RayState
from abcdef_sim.physics.abcdef.propagation import propagate_step

pytestmark = pytest.mark.physics


def test_propagate_step_updates_rays_and_system_in_traversal_order() -> None:
    ray_in = np.array(
        [
            [[1.5], [0.25], [1.0]],
            [[-0.8], [0.4], [1.0]],
        ],
        dtype=float,
    )
    state = RayState(
        rays=ray_in,
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
        meta={"label": "seed"},
    )
    M1 = np.array(
        [
            [
                [1.0, 2.0, 0.5],
                [0.0, 1.0, -0.25],
                [0.0, 0.0, 1.0],
            ],
            [
                [1.0, -1.5, -0.2],
                [0.3, 1.0, 0.75],
                [0.0, 0.0, 1.0],
            ],
        ],
        dtype=float,
    )
    M2 = np.array(
        [
            [
                [0.8, 0.0, 1.2],
                [-0.1, 1.1, 0.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [1.2, 0.4, -1.1],
                [0.0, 0.9, 0.5],
                [0.0, 0.0, 1.0],
            ],
        ],
        dtype=float,
    )

    final = propagate_step(propagate_step(state, M1), M2)
    expected_system = M2 @ M1
    expected_rays = expected_system @ ray_in

    np.testing.assert_allclose(final.system, expected_system)
    np.testing.assert_allclose(final.rays, expected_rays)
    assert not np.allclose(final.system, M1 @ M2)
    assert final.meta == state.meta
    assert final.meta is not state.meta


def test_propagate_step_rejects_matrix_batch_count_mismatch() -> None:
    state = RayState(
        rays=np.array(
            [
                [[1.0], [0.1], [1.0]],
                [[-2.0], [0.3], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
    )
    M_elem = np.repeat(np.eye(3, dtype=float)[None, ...], 3, axis=0)

    with pytest.raises(ValueError) as excinfo:
        propagate_step(state, M_elem)

    message = str(excinfo.value)
    assert "matching batch sizes" in message
    assert "(3, 3, 3)" in message
    assert "(2, 3, 1)" in message


def test_propagate_step_rejects_unbatched_matrix_input() -> None:
    state = RayState(
        rays=np.array(
            [
                [[1.0], [0.1], [1.0]],
                [[-2.0], [0.3], [1.0]],
            ],
            dtype=float,
        ),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], 2, axis=0),
    )

    with pytest.raises(ValueError) as excinfo:
        propagate_step(state, np.eye(3, dtype=float))

    assert "M_elem must have shape (N,3,3)" in str(excinfo.value)
