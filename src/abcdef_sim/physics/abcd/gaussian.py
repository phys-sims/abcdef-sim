from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]
EPS: Final[float] = 1e-12


def q_propagate(q: complex, M: NDArrayF) -> complex:
    """Propagate Gaussian-beam complex q-parameter through an ABCD matrix.

    Assumptions:
      - Scalar Gaussian beam in paraxial regime.
      - Matrix ``M`` uses the same axis/sign conventions as the ray model.

    Units/sign conventions:
      - ``q = z + i zR`` with propagation distance ``z`` in ECO-0001 length units.
      - Uses ``q_out = (A q + B) / (C q + D)``.

    Equation reference:
      - Standard Kogelnik first-order ABCD transform for Gaussian beams.
    """

    M_arr = np.asarray(M, dtype=float)
    if M_arr.shape != (2, 2):
        raise ValueError(f"M must have shape (2, 2); got {M_arr.shape}")

    A, B = float(M_arr[0, 0]), float(M_arr[0, 1])
    C, D = float(M_arr[1, 0]), float(M_arr[1, 1])
    denom = C * q + D
    if denom == 0:
        raise ZeroDivisionError("C*q + D is zero; q transform is singular")
    return (A * q + B) / denom


def rayleigh_range_from_waist(w0: float, wavelength: float, n: float = 1.0) -> float:
    """Compute Rayleigh range from beam waist.

    Assumptions:
      - Fundamental TEM00 Gaussian beam with scalar approximation.

    Units/sign conventions:
      - ``w0`` and ``wavelength`` are in the same internal length unit.
      - ``n`` is refractive index of the medium.
      - Uses ``zR = pi * n * w0^2 / wavelength``.
    """

    w0f = float(w0)
    wavelength_f = float(wavelength)
    nf = float(n)
    if w0f < 0.0:
        raise ValueError("w0 must be >= 0")
    if wavelength_f <= 0.0:
        raise ValueError("wavelength must be > 0")
    if nf <= 0.0:
        raise ValueError("n must be > 0")
    return float(np.pi * nf * (w0f**2) / wavelength_f)


def q_from_waist(
    waist_radius: float,
    wavelength: float,
    distance_from_waist: float = 0.0,
    n: float = 1.0,
) -> complex:
    """Build a Gaussian-beam q-parameter from waist data.

    Assumptions:
      - Uses the local repo convention ``q = z + i zR``.

    Units/sign conventions:
      - ``distance_from_waist`` is positive when the current plane lies
        downstream of the waist.
      - ``waist_radius`` and ``wavelength`` share one consistent length unit.
    """

    return complex(
        float(distance_from_waist),
        rayleigh_range_from_waist(waist_radius, wavelength, n=n),
    )


def distance_from_waist(q: complex) -> float:
    """Return the current-plane offset relative to the beam waist."""

    return float(np.real(complex(q)))


def rayleigh_range_from_q(q: complex) -> float:
    """Return Rayleigh range from a q-parameter using ``q = z + i zR``."""

    z_r = float(np.imag(complex(q)))
    if z_r <= 0.0:
        raise ValueError("The imaginary part of q must be > 0")
    return z_r


def waist_radius_from_q(q: complex, wavelength: float, n: float = 1.0) -> float:
    """Recover the beam-waist radius from a q-parameter."""

    z_r = rayleigh_range_from_q(q)
    wavelength_f = float(wavelength)
    nf = float(n)
    if wavelength_f <= 0.0:
        raise ValueError("wavelength must be > 0")
    if nf <= 0.0:
        raise ValueError("n must be > 0")
    return float(np.sqrt(wavelength_f * z_r / (np.pi * nf)))


def waist_position_from_plane(q: complex) -> float:
    """Return waist position relative to the current plane.

    Positive values mean the waist lies downstream of the current plane.
    """

    return float(-distance_from_waist(q))


def beam_radius_from_q(q: complex, wavelength: float, n: float = 1.0) -> float:
    """Compute the current Gaussian beam radius from ``q = z + i zR``."""

    z = distance_from_waist(q)
    z_r = rayleigh_range_from_q(q)
    w0 = waist_radius_from_q(q, wavelength, n=n)
    return float(w0 * np.sqrt(1.0 + (z / z_r) ** 2))


def sample_beam_radius_profile(
    q_at_plane: complex,
    wavelength: float,
    z_samples: npt.ArrayLike,
    n: float = 1.0,
) -> NDArrayF:
    """Sample beam radius after additional free-space propagation distances."""

    z_arr = np.asarray(z_samples, dtype=float)
    q_samples = z_arr + complex(q_at_plane)
    z_rel = np.real(q_samples)
    z_r = rayleigh_range_from_q(q_at_plane)
    w0 = waist_radius_from_q(q_at_plane, wavelength, n=n)
    profile = w0 * np.sqrt(1.0 + (z_rel / z_r) ** 2)
    return np.asarray(profile, dtype=float)
