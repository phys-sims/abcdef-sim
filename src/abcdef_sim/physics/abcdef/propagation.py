"""Pure batched ABCDEF propagation kernels."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from abcdef_sim.data_models.states import RayState
from abcdef_sim.physics.abcdef.conventions import validate_matrix_shape, validate_ray_shape

NDArrayF = npt.NDArray[np.float64]

__all__ = ["propagate_step"]


def propagate_step(state: RayState, M_elem: NDArrayF) -> RayState:
    """Propagate a batched ABCDEF ray state by one element.

    Assumptions:
      - ``state.rays`` follows the Martinez convention ``[x, x_prime, 1]^T``.
      - ``M_elem`` is a per-omega ABCDEF batch with one matrix per ray batch entry.

    Equation reference:
      - ``rays_out = M_elem @ rays_in``
      - ``system_out = M_elem @ system_in``

    Notes:
      - System accumulation follows the reverse-order Martinez composition
        convention ``system <- M_elem @ system``.
      - ``meta`` is shallow-copied onto the returned ``RayState``.
    """

    rays = validate_ray_shape(state.rays)
    matrices = validate_matrix_shape(M_elem)
    system = np.asarray(state.system, dtype=float)

    if matrices.ndim != 3:
        raise ValueError(f"M_elem must have shape (N,3,3); got {matrices.shape}")
    if system.ndim != 3 or system.shape[1:] != (3, 3):
        raise ValueError(f"state.system must have shape (N,3,3); got {system.shape}")

    batch_size = rays.shape[0]
    if matrices.shape[0] != batch_size or system.shape[0] != batch_size:
        raise ValueError(
            "M_elem, state.rays, and state.system must have matching batch sizes: "
            f"M_elem.shape={matrices.shape}, rays.shape={rays.shape}, system.shape={system.shape}"
        )

    rays_out = matrices @ rays
    system_out = matrices @ system
    return RayState(rays=rays_out, system=system_out, meta=dict(state.meta))
