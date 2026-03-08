from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from abcdef_sim.data_models.states import RayState

NDArrayF = npt.NDArray[np.float64]


def _coerce_vector(value: object) -> NDArrayF:
    return np.asarray(value, dtype=np.float64).reshape(-1)


class PhaseContribution(BaseModel):
    """Per-optic Martinez phase bookkeeping on a shared omega grid."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    optic_name: str
    instance_name: str
    backend_id: str | None = None

    omega: NDArrayF
    phi0_rad: NDArrayF
    phi3_rad: NDArrayF

    filter_amp: NDArrayF | None = None
    filter_phase_rad: NDArrayF | None = None

    @field_validator(
        "omega",
        "phi0_rad",
        "phi3_rad",
        "filter_amp",
        "filter_phase_rad",
        mode="before",
    )
    @classmethod
    def _normalize_vector_fields(cls, value: object) -> NDArrayF | None:
        if value is None:
            return None
        return _coerce_vector(value)

    @model_validator(mode="after")
    def _check_shapes(self) -> "PhaseContribution":
        omega = self.omega
        expected_shape = (omega.size,)
        for field_name in ("phi0_rad", "phi3_rad", "filter_amp", "filter_phase_rad"):
            field_value = getattr(self, field_name)
            if field_value is not None and field_value.shape != expected_shape:
                raise ValueError(
                    f"{field_name} must have shape {expected_shape}; got {field_value.shape}"
                )
        return self


class PipelineResult(BaseModel):
    """Final ray state plus phase bookkeeping across the full optics chain."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    final_state: RayState
    omega: NDArrayF
    contributions: tuple[PhaseContribution, ...]
    phi1_rad: NDArrayF | None = None
    phi2_rad: NDArrayF | None = None
    phi4_rad: NDArrayF | None = None
    phi_total_rad: NDArrayF

    @field_validator("omega", "phi1_rad", "phi2_rad", "phi4_rad", "phi_total_rad", mode="before")
    @classmethod
    def _normalize_vector_fields(cls, value: object) -> NDArrayF | None:
        if value is None:
            return None
        return _coerce_vector(value)

    @model_validator(mode="after")
    def _check_shapes(self) -> "PipelineResult":
        omega = self.omega
        expected_shape = (omega.size,)

        for field_name in ("phi1_rad", "phi2_rad", "phi4_rad", "phi_total_rad"):
            field_value = getattr(self, field_name)
            if field_value is not None and field_value.shape != expected_shape:
                raise ValueError(
                    f"{field_name} must have shape {expected_shape}; got {field_value.shape}"
                )

        state_batch = np.asarray(self.final_state.rays, dtype=float).shape[0]
        if state_batch != omega.size:
            raise ValueError(
                "final_state batch dimension must match omega: "
                f"omega.shape={omega.shape}, final_state.rays.shape={self.final_state.rays.shape}"
            )

        for contribution in self.contributions:
            if contribution.omega.shape != expected_shape:
                raise ValueError(
                    "contribution omega must match pipeline omega shape: "
                    f"expected {expected_shape}, got {contribution.omega.shape}"
                )
            if not np.array_equal(contribution.omega, omega):
                raise ValueError(
                    "each contribution omega must exactly match pipeline omega values"
                )

        return self
