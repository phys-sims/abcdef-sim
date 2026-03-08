from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim.data_models.states import RayState


def test_raystate_rejects_mismatched_ray_and_system_batch_sizes() -> None:
    rays = np.array(
        [
            [1.0, 0.1, 1.0],
            [-2.0, 0.3, 1.0],
        ],
        dtype=float,
    )
    system = np.repeat(np.eye(3, dtype=float)[None, ...], 3, axis=0)

    with pytest.raises(ValueError) as excinfo:
        RayState(rays=rays, system=system)

    message = str(excinfo.value)
    assert "system batch dimension must match rays" in message
    assert "(2, 3, 1)" in message
    assert "(3, 3, 3)" in message
