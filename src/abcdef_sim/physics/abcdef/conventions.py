"""Martinez-aligned ABCDEF conventions.

This module is the single source of truth for the ABCDEF ray and matrix
conventions used by ``abcdef-sim``.

Martinez constraints adopted here:
  - Rays are column vectors ``[x, x_prime, 1]^T``.
  - ``x_prime`` means ``n dx/dz = n theta``.
  - In a 3x3 ABCDEF matrix, ``E`` is the inhomogeneous shift in ``x`` and
    ``F`` is the inhomogeneous shift in ``x_prime``.
  - System composition is reverse-order: the first traversed element is on the
    right, and cumulative updates follow ``system <- M_elem @ system``.

Actual phase terms are intentionally out of scope for this module.
"""

from __future__ import annotations

from typing import Final, Literal

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]
SlopeConvention = Literal["n_theta"]

CONVENTIONS_VERSION: Final[str] = "eco-0001+martinez-v1"
RAY_VECTOR_DOC: Final[str] = "[x, x_prime, 1]^T"

__all__ = [
    "CONVENTIONS_VERSION",
    "RAY_VECTOR_DOC",
    "SlopeConvention",
    "compose_system",
    "extract_E",
    "extract_F",
    "validate_matrix_shape",
    "validate_ray_shape",
]


def validate_matrix_shape(M: object) -> NDArrayF:
    """Validate an ABCDEF matrix array shape."""

    M_arr = np.asarray(M, dtype=float)
    if M_arr.shape == (3, 3):
        return M_arr
    if M_arr.ndim == 3 and M_arr.shape[1:] == (3, 3):
        return M_arr
    raise ValueError(f"ABCDEF matrix must have shape (3, 3) or (N, 3, 3); got {M_arr.shape}")


def validate_ray_shape(rays: object) -> NDArrayF:
    """Validate and normalize ray vectors to batched ``(N, 3, 1)`` form."""

    rays_arr = np.asarray(rays, dtype=float)
    if rays_arr.shape == (3, 1):
        return rays_arr[None, ...]
    if rays_arr.ndim == 2 and rays_arr.shape[1] == 3:
        return rays_arr[..., None]
    if rays_arr.ndim == 3 and rays_arr.shape[1:] == (3, 1):
        return rays_arr
    raise ValueError(
        f"Ray vectors must have shape (3, 1), (N, 3, 1), or (N, 3); got {rays_arr.shape}"
    )


def extract_E(M: object) -> NDArrayF:
    """Return the inhomogeneous ``E`` term batch from ABCDEF matrices."""

    matrices, _ = _as_matrix_batch(M)
    return matrices[:, 0, 2]


def extract_F(M: object) -> NDArrayF:
    """Return the inhomogeneous ``F`` term batch from ABCDEF matrices."""

    matrices, _ = _as_matrix_batch(M)
    return matrices[:, 1, 2]


def compose_system(system: object, M_elem: object) -> NDArrayF:
    """Compose cumulative systems in Martinez reverse-order form."""

    system_batch, system_was_unbatched = _as_matrix_batch(system)
    elem_batch, elem_was_unbatched = _as_matrix_batch(M_elem)

    system_count = system_batch.shape[0]
    elem_count = elem_batch.shape[0]
    if system_count != elem_count:
        if system_count == 1:
            system_batch = np.broadcast_to(system_batch, (elem_count, 3, 3))
        elif elem_count == 1:
            elem_batch = np.broadcast_to(elem_batch, (system_count, 3, 3))
        else:
            raise ValueError(
                "system and M_elem must have matching batch sizes or singleton "
                f"batches; got {system_count} and {elem_count}"
            )

    composed = elem_batch @ system_batch
    if system_was_unbatched and elem_was_unbatched:
        return composed[0]
    return composed


def _as_matrix_batch(M: object) -> tuple[NDArrayF, bool]:
    M_arr = validate_matrix_shape(M)
    if M_arr.ndim == 2:
        return M_arr[None, ...], True
    return M_arr, False
