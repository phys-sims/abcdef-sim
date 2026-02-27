from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from abcdef_sim.physics.abcd.ray import Ray

NDArrayF = npt.NDArray[np.float64]


def _import_raytracing() -> Any:
    try:
        import raytracing as rt
    except ImportError as exc:
        raise ImportError(
            "raytracing is required for reference validation helpers. "
            "Install with `pip install abcdef-sim[validation]`."
        ) from exc
    return rt


def to_raytracing_matrix(M: NDArrayF) -> Any:
    """Convert a local 2x2 ABCD matrix into a raytracing.Matrix instance."""

    rt = _import_raytracing()
    M_arr = np.asarray(M, dtype=float)
    if M_arr.shape != (2, 2):
        raise ValueError(f"M must have shape (2, 2); got {M_arr.shape}")
    return rt.Matrix(A=M_arr[0, 0], B=M_arr[0, 1], C=M_arr[1, 0], D=M_arr[1, 1])


def from_raytracing_matrix(rt_matrix: Any) -> NDArrayF:
    """Convert a raytracing.Matrix-like object into a local numpy 2x2 matrix."""

    return np.array(
        [
            [float(rt_matrix.A), float(rt_matrix.B)],
            [float(rt_matrix.C), float(rt_matrix.D)],
        ],
        dtype=float,
    )


def trace_ray_raytracing(ray: Ray, elements: Sequence[NDArrayF]) -> Ray:
    """Trace a ray using the external raytracing package for validation.

    The ``elements`` sequence must contain local 2x2 matrices in traversal order.
    """

    rt = _import_raytracing()
    rt_ray = rt.Ray(y=ray.y, theta=ray.theta)

    total = rt.Matrix(A=1.0, B=0.0, C=0.0, D=1.0)
    for element in elements:
        total = to_raytracing_matrix(element) * total

    out = total * rt_ray
    return Ray(y=float(out.y), theta=float(out.theta))
