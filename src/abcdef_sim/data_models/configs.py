from typing import Callable, Optional

from pydantic import Field, model_validator, ConfigDict

from phys_pipeline.v1.types import StageConfig
from abcdef_sim.data_models.optics import Optic

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]


class OpticStageCfg(StageConfig):
    """
    Immutable config for a single optic stage evaluated on a frequency grid.

    Invariant:
      omega[i] <-> abcdef[i] <-> refractive_index[i]
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    optic_name: str
    instance_name: str
    length: float

    omega: NDArrayF                 # (N,)
    abcdef: NDArrayF                # (N,3,3)
    refractive_index: NDArrayF      # (N,)

    @model_validator(mode="after")
    def _check_shapes(self) -> "OpticStageCfg":
        w = np.asarray(self.omega, dtype=np.float64).reshape(-1)
        m = np.asarray(self.abcdef, dtype=np.float64)
        n = np.asarray(self.refractive_index, dtype=np.float64).reshape(-1)

        if m.shape != (w.size, 3, 3):
             raise ValueError(f"abcdef must be (N,3,3); got {m.shape} for N={w.size}")
        if n.shape != (w.size,):
            raise ValueError(f"refractive_index must be (N,); got {n.shape} for N={w.size}")

        return self
