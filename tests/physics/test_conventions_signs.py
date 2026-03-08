from __future__ import annotations

import numpy as np
import pytest


pytestmark = pytest.mark.physics

from abcdef_sim.physics.abcdef.conventions import (
    compose_system,
    extract_E,
    extract_F,
    validate_matrix_shape,
    validate_ray_shape,
)


def test_extract_e_and_f_read_the_inhomogeneous_terms() -> None:
    matrices = np.array(
        [
            [
                [1.0, 2.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            [
                [1.0, 0.0, -3.0],
                [4.0, 1.0, 2.0],
                [0.0, 0.0, 1.0],
            ],
        ],
        dtype=float,
    )

    np.testing.assert_allclose(extract_E(matrices), np.array([0.0, -3.0], dtype=float))
    np.testing.assert_allclose(extract_F(matrices), np.array([1.0, 2.0], dtype=float))


def test_compose_system_uses_reverse_order_multiplication() -> None:
    system = np.array(
        [
            [1.0, 3.0, 0.5],
            [0.0, 1.0, -0.25],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    M_elem = np.array(
        [
            [1.0, 0.0, 2.0],
            [-0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    expected = M_elem @ system
    wrong_order = system @ M_elem

    composed = compose_system(system, M_elem)

    np.testing.assert_allclose(composed, expected)
    assert not np.allclose(composed, wrong_order)


def test_validate_matrix_shape_rejects_non_abcdef_shapes() -> None:
    with pytest.raises(ValueError) as excinfo:
        validate_matrix_shape(np.eye(2, dtype=float))

    message = str(excinfo.value)
    assert "shape (3, 3) or (N, 3, 3)" in message
    assert "(2, 2)" in message


def test_validate_ray_shape_normalizes_n_by_3_input() -> None:
    rays = np.array(
        [
            [1.0, 0.1, 1.0],
            [-2.0, 0.3, 1.0],
        ],
        dtype=float,
    )

    normalized = validate_ray_shape(rays)

    assert normalized.shape == (2, 3, 1)
    np.testing.assert_allclose(normalized[..., 0], rays)
