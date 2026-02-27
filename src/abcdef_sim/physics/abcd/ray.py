from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class Ray:
    """Paraxial ray represented by height and angle.

    Assumptions:
      - First-order paraxial optics with small ``theta``.

    Units/sign conventions:
      - ``y`` follows ECO-0001 internal length conventions.
      - ``theta`` is in radians and positive for counter-clockwise tilt.
    """

    y: float
    theta: float


def propagate_ray(ray: Ray, M: NDArrayF) -> Ray:
    """Propagate a paraxial ray through an ABCD matrix.

    Assumptions:
      - ``M`` is a 2x2 transfer matrix acting on ``[y, theta]^T``.

    Units/sign conventions:
      - Matrix and ray must use one consistent convention set (ECO-0001).

    Equation reference:
      - ``[y_out, theta_out]^T = M @ [y_in, theta_in]^T``.
    """

    M_arr = np.asarray(M, dtype=float)
    if M_arr.shape != (2, 2):
        raise ValueError(f"M must have shape (2, 2); got {M_arr.shape}")

    v_out = M_arr @ np.array([ray.y, ray.theta], dtype=float)
    return Ray(y=float(v_out[0]), theta=float(v_out[1]))
