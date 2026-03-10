from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import ConfigDict, field_validator, model_validator

from abcdef_sim._phys_pipeline import StageConfig

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
    phase_model: Literal["physical", "none"] = "physical"

    omega: NDArrayF  # (N,)
    delta_omega_rad_per_fs: NDArrayF | None = None  # (N,)
    omega0_rad_per_fs: float = 0.0
    abcdef: NDArrayF  # (N,3,3)
    refractive_index: NDArrayF  # (N,)

    @field_validator("omega", "delta_omega_rad_per_fs", "abcdef", "refractive_index", mode="before")
    @classmethod
    def _coerce_arrays(cls, value: object) -> NDArrayF | None:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64)

    @model_validator(mode="after")
    def _check_shapes(self) -> "OpticStageCfg":
        w = np.asarray(self.omega, dtype=np.float64).reshape(-1)
        dw = self.delta_omega_rad_per_fs
        m = np.asarray(self.abcdef, dtype=np.float64)
        n = np.asarray(self.refractive_index, dtype=np.float64).reshape(-1)

        if m.shape != (w.size, 3, 3):
            raise ValueError(f"abcdef must be (N,3,3); got {m.shape} for N={w.size}")
        if n.shape != (w.size,):
            raise ValueError(f"refractive_index must be (N,); got {n.shape} for N={w.size}")
        if dw is None:
            object.__setattr__(self, "delta_omega_rad_per_fs", w - float(np.mean(w)))
        else:
            dw_arr = np.asarray(dw, dtype=np.float64).reshape(-1)
            if dw_arr.shape != (w.size,):
                raise ValueError(
                    f"delta_omega_rad_per_fs must be (N,); got {dw_arr.shape} for N={w.size}"
                )
            if not np.allclose(w, float(self.omega0_rad_per_fs) + dw_arr, atol=1e-12, rtol=0.0):
                raise ValueError("omega must equal omega0_rad_per_fs + delta_omega_rad_per_fs")

        return self
