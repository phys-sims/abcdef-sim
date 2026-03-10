from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.physics.abcdef.phase_terms import (
    combine_phi_total_rad,
    martinez_k,
    phi0_rad_i,
    phi1_rad,
    phi3_rad_i,
    phi4_rad,
)

pytestmark = pytest.mark.physics


def test_phi0_rad_i_matches_plane_wave_optical_path_phase() -> None:
    omega = np.linspace(3.0, 4.2, 5, dtype=float)
    n = np.linspace(1.1, 1.5, 5, dtype=float)
    length = 3.5

    expected = martinez_k(omega) * length * n

    np.testing.assert_allclose(phi0_rad_i(omega, length, n), expected)


def test_phi3_rad_i_uses_post_element_displacement_with_martinez_sign() -> None:
    k = 3.0
    F_i = np.array([0.0, 0.1, -0.2, 0.3, -0.4], dtype=float)
    x_after = np.array([1.5, -2.0, 0.5, -1.0, 2.5], dtype=float)

    expected = 0.5 * k * F_i * x_after

    np.testing.assert_allclose(phi3_rad_i(k, F_i, x_after), expected)


def test_phi1_rad_matches_martinez_equation_25_phase_factor() -> None:
    matrices = np.array(
        [
            [[1.0, 0.4, 0.0], [0.2, 1.2, 0.0], [0.0, 0.0, 1.0]],
            [[0.9, -0.1, 0.0], [0.3, 1.1, 0.0], [0.0, 0.0, 1.0]],
            [[1.1, 0.3, 0.0], [-0.2, 0.8, 0.0], [0.0, 0.0, 1.0]],
        ],
        dtype=float,
    )
    q_in = np.array([2.0 + 3.0j, 1.5 + 2.5j, 3.0 + 1.0j], dtype=np.complex128)

    lhs = 1.0 / np.sqrt(matrices[:, 0, 0].astype(np.complex128) + matrices[:, 0, 1] / q_in)
    w_in = np.array([1.0, 1.3, 0.8], dtype=float)
    w_out = w_in / np.abs(lhs) ** 2
    expected = -np.angle(lhs)

    # Martinez eq. 25: 1/sqrt(A + B/q_in) = sqrt(w_in / w_out) * exp(-j phi1).
    np.testing.assert_allclose(phi1_rad(matrices, q_in, w_in, w_out), expected)


def test_phi1_rad_accepts_unbatched_abcdef_matrix() -> None:
    matrix = np.array(
        [[1.0, 0.4, 0.0], [0.2, 1.2, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    q_in = 2.0 + 3.0j

    lhs = 1.0 / np.sqrt(matrix[0, 0].astype(np.complex128) + matrix[0, 1] / q_in)
    w_in = 1.2
    w_out = w_in / np.abs(lhs) ** 2
    expected = np.array([-np.angle(lhs)], dtype=float)

    np.testing.assert_allclose(phi1_rad(matrix, q_in, w_in, w_out), expected)


def test_phi4_rad_matches_martinez_equation_30_and_broadcasts_scalar_x() -> None:
    k = 3.5
    x = 0.5
    x_out = np.array([0.1, -0.2, 0.4], dtype=float)
    q_out = np.array([1.0 + 2.0j, 2.0 + 1.0j, 3.0 + 4.0j], dtype=np.complex128)

    expected = np.real(k * (x - x_out) ** 2 / (2.0 * q_out))

    np.testing.assert_allclose(phi4_rad(k, x, x_out, q_out), expected)


def test_combine_phi_total_rad_sums_present_terms_only() -> None:
    phi0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    phi1 = np.array([0.5, 0.4, 0.3, 0.2, 0.1], dtype=float)
    phi3 = np.array([-0.2, -0.1, 0.0, 0.1, 0.2], dtype=float)
    phi4 = np.array([0.05, 0.05, 0.05, 0.05, 0.05], dtype=float)

    expected_without_phi4 = phi0 + phi1 + phi3
    expected_with_phi4 = expected_without_phi4 + phi4

    np.testing.assert_allclose(combine_phi_total_rad(phi0, phi1, None, phi3), expected_without_phi4)
    np.testing.assert_allclose(
        combine_phi_total_rad(phi0, phi1, None, phi3, phi4),
        expected_with_phi4,
    )
