from __future__ import annotations

from typing import Final

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]
EPS: Final[float] = 1e-12


def free_space(length: float) -> NDArrayF:
    """Return the paraxial free-space ABCD matrix.

    Assumptions:
      - Paraxial optics with small ray angles.
      - Ray state is represented as the column vector ``[y, theta]^T``.

    Units/sign conventions:
      - ``length`` and ``y`` use the internal length unit adopted by ECO-0001.
      - ``theta`` is in radians, positive for counter-clockwise tilt.
      - Uses the standard transfer relation ``y2 = y1 + L * theta1``.

    Equation reference:
      - Derived from first-order propagation over distance ``L``:
        ``M = [[1, L], [0, 1]]``.
    """

    return np.array([[1.0, float(length)], [0.0, 1.0]], dtype=float)


def thin_lens(focal_length: float) -> NDArrayF:
    """Return the paraxial thin-lens ABCD matrix.

    Assumptions:
      - Paraxial optics with thin-element approximation.
      - Positive focal length corresponds to a converging lens.

    Units/sign conventions:
      - ``focal_length`` is in the ECO-0001 internal length unit.
      - Ray state follows ``[y, theta]^T`` and angle sign conventions above.

    Equation reference:
      - Lens kick relation ``theta2 = theta1 - y1 / f`` gives
        ``M = [[1, 0], [-1/f, 1]]``.
    """

    f = float(focal_length)
    if abs(f) < EPS:
        raise ValueError("focal_length must be non-zero")
    return np.array([[1.0, 0.0], [-1.0 / f, 1.0]], dtype=float)


def interface(n1: float, n2: float, R: float | None = None) -> NDArrayF:
    """Return the ABCD matrix for a refractive index interface.

    Assumptions:
      - Paraxial refraction at a spherical or planar interface.
      - State vector is ``[y, theta]^T`` using geometric angle ``theta``.

    Units/sign conventions:
      - ``n1`` and ``n2`` are refractive indices before/after the interface.
      - ``R`` is the interface radius in internal length units.
      - Radius sign follows Cartesian convention: center of curvature to the right
        of the surface gives ``R > 0``.
      - The matrix uses
        ``theta2 = (n1/n2) * theta1 + ((n1 - n2) / (n2 * R)) * y1``.
      - Conventions should be kept aligned with ECO-0001 adoption ADR.

    Equation reference:
      - First-order Snell-law expansion for spherical interfaces.
    """

    n1f = float(n1)
    n2f = float(n2)
    if abs(n2f) < EPS:
        raise ValueError("n2 must be non-zero")

    if R is None:
        c_term = 0.0
    else:
        radius = float(R)
        if abs(radius) < EPS:
            raise ValueError("R must be non-zero when provided")
        c_term = (n1f - n2f) / (n2f * radius)

    return np.array([[1.0, 0.0], [c_term, n1f / n2f]], dtype=float)


def thick_lens_elements(
    n_lens: float,
    R1: float | None,
    R2: float | None,
    thickness: float,
    n_in: float = 1.0,
    n_out: float = 1.0,
) -> tuple[NDArrayF, NDArrayF, NDArrayF]:
    """Return primitive ABCD matrices for a thick lens.

    The ray state convention is ``[y, theta]^T``, where ``theta`` is geometric
    angle. Radius signs use the Cartesian convention: center of curvature to the
    right of a surface gives ``R > 0``.

    The three returned matrices are:
      1) ``interface(n_in, n_lens, R1)``
      2) ``free_space(thickness)``
      3) ``interface(n_lens, n_out, R2)``

    Notes:
      - ``R1`` and/or ``R2`` may be ``None`` for planar faces.
      - For left-to-right propagation, a biconvex lens usually has ``R1 > 0``
        and ``R2 < 0``.

    Example:
      - For ``n_lens=1.5``, ``R1=4.0``, ``R2=6.0``, ``thickness=3.0`` in air,
        ``compose(*thick_lens_elements(...))`` is approximately
        ``[[0.75, 2.0], [-0.0625, 1.1666666667]]``.
    """

    n_lens_f = float(n_lens)
    n_in_f = float(n_in)
    n_out_f = float(n_out)
    thickness_f = float(thickness)

    if n_lens_f <= 0.0:
        raise ValueError("n_lens must be > 0")
    if n_in_f <= 0.0:
        raise ValueError("n_in must be > 0")
    if n_out_f <= 0.0:
        raise ValueError("n_out must be > 0")
    if thickness_f < 0.0:
        raise ValueError("thickness must be >= 0")

    if R1 is not None and abs(float(R1)) < EPS:
        raise ValueError("R1 must be non-zero when provided")
    if R2 is not None and abs(float(R2)) < EPS:
        raise ValueError("R2 must be non-zero when provided")

    M1 = interface(n_in_f, n_lens_f, R1)
    Mp = free_space(thickness_f)
    M2 = interface(n_lens_f, n_out_f, R2)
    return M1, Mp, M2


def thick_lens(
    n_lens: float,
    R1: float | None,
    R2: float | None,
    thickness: float,
    n_in: float = 1.0,
    n_out: float = 1.0,
) -> NDArrayF:
    """Return the composite ABCD matrix of a thick lens.

    State convention is ``[y, theta]^T`` with geometric ``theta``. Radius signs
    follow Cartesian convention: center of curvature to the right gives
    ``R > 0``. For left-to-right propagation, a biconvex lens usually has
    ``R1 > 0`` and ``R2 < 0``.

    This is equivalent to composing, in traversal order,
    ``interface(n_in, n_lens, R1)``, ``free_space(thickness)``, and
    ``interface(n_lens, n_out, R2)``.
    """

    return compose(
        *thick_lens_elements(
            n_lens=n_lens,
            R1=R1,
            R2=R2,
            thickness=thickness,
            n_in=n_in,
            n_out=n_out,
        )
    )


def compose(*matrices: NDArrayF) -> NDArrayF:
    """Compose ABCD matrices in optical traversal order.

    Assumptions:
      - Each matrix is 2x2 and operates on column vectors ``[y, theta]^T``.

    Units/sign conventions:
      - Inputs must already share one consistent unit/sign system (ECO-0001).

    Equation reference:
      - For elements ``M1, M2, ...`` traversed in that order,
        ``v_out = Mn @ ... @ M2 @ M1 @ v_in``.
      - Therefore ``compose(M1, M2, ..., Mn)`` returns ``Mn @ ... @ M1``.
    """

    M_total = np.eye(2, dtype=float)
    for M in matrices:
        M_arr = np.asarray(M, dtype=float)
        if M_arr.shape != (2, 2):
            raise ValueError(f"Each matrix must have shape (2, 2); got {M_arr.shape}")
        M_total = M_arr @ M_total
    return M_total


def det(M: NDArrayF) -> float:
    """Return the determinant of a 2x2 ABCD matrix."""

    M_arr = np.asarray(M, dtype=float)
    if M_arr.shape != (2, 2):
        raise ValueError(f"M must have shape (2, 2); got {M_arr.shape}")
    return float(np.linalg.det(M_arr))


def is_symplectic(M: NDArrayF, atol: float = 1e-9) -> bool:
    """Check whether an ABCD matrix is symplectic in the 2x2 paraxial sense.

    Assumptions:
      - For ``[y, theta]`` parameterization in homogeneous media, ideal paraxial
        systems satisfy ``det(M) = 1``.
    """

    return abs(det(M) - 1.0) <= float(atol)
