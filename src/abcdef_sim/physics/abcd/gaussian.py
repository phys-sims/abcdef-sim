from __future__ import annotations

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]


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

    if wavelength == 0:
        raise ValueError("wavelength must be non-zero")
    return float(np.pi * n * (w0**2) / wavelength)
