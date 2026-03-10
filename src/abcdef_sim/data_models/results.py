from __future__ import annotations

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from abcdef_sim.data_models.standalone import LaserState as StandaloneLaserState
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
    delta_omega_rad_per_fs: NDArrayF | None = None
    omega0_rad_per_fs: float = 0.0
    phi0_rad: NDArrayF
    phi_geom_rad: NDArrayF | None = None
    phi3_transport_like_rad: NDArrayF | None = None
    phi3_phase_rad: NDArrayF | None = None
    phi3_rad: NDArrayF
    path_length_um: NDArrayF | None = None
    group_delay_fs: NDArrayF | None = None

    filter_amp: NDArrayF | None = None
    filter_phase_rad: NDArrayF | None = None

    @field_validator(
        "omega",
        "delta_omega_rad_per_fs",
        "phi0_rad",
        "phi_geom_rad",
        "phi3_transport_like_rad",
        "phi3_phase_rad",
        "phi3_rad",
        "path_length_um",
        "group_delay_fs",
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
    def _check_shapes(self) -> PhaseContribution:
        omega = self.omega
        expected_shape = (omega.size,)
        if self.delta_omega_rad_per_fs is None:
            object.__setattr__(self, "delta_omega_rad_per_fs", omega - float(np.mean(omega)))
        elif self.delta_omega_rad_per_fs.shape != expected_shape:
            raise ValueError(
                "delta_omega_rad_per_fs must have shape "
                f"{expected_shape}; got {self.delta_omega_rad_per_fs.shape}"
            )
        for field_name in (
            "phi0_rad",
            "phi_geom_rad",
            "phi3_transport_like_rad",
            "phi3_phase_rad",
            "phi3_rad",
            "path_length_um",
            "group_delay_fs",
            "filter_amp",
            "filter_phase_rad",
        ):
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
    delta_omega_rad_per_fs: NDArrayF | None = None
    omega0_rad_per_fs: float = 0.0
    contributions: tuple[PhaseContribution, ...]
    phi0_axial_total_rad: NDArrayF | None = None
    phi_geom_total_rad: NDArrayF | None = None
    phi3_transport_like_total_rad: NDArrayF | None = None
    phi3_phase_total_rad: NDArrayF | None = None
    phi3_total_rad: NDArrayF | None = None
    phi1_rad: NDArrayF | None = None
    phi2_rad: NDArrayF | None = None
    phi4_rad: NDArrayF | None = None
    phi_total_rad: NDArrayF

    @field_validator(
        "omega",
        "delta_omega_rad_per_fs",
        "phi0_axial_total_rad",
        "phi_geom_total_rad",
        "phi3_transport_like_total_rad",
        "phi3_phase_total_rad",
        "phi3_total_rad",
        "phi1_rad",
        "phi2_rad",
        "phi4_rad",
        "phi_total_rad",
        mode="before",
    )
    @classmethod
    def _normalize_vector_fields(cls, value: object) -> NDArrayF | None:
        if value is None:
            return None
        return _coerce_vector(value)

    @model_validator(mode="after")
    def _check_shapes(self) -> PipelineResult:
        omega = self.omega
        expected_shape = (omega.size,)
        delta_omega = self.delta_omega_rad_per_fs

        for field_name in (
            "phi0_axial_total_rad",
            "phi_geom_total_rad",
            "phi3_transport_like_total_rad",
            "phi3_phase_total_rad",
            "phi3_total_rad",
            "phi1_rad",
            "phi2_rad",
            "phi4_rad",
            "phi_total_rad",
        ):
            field_value = getattr(self, field_name)
            if field_value is not None and field_value.shape != expected_shape:
                raise ValueError(
                    f"{field_name} must have shape {expected_shape}; got {field_value.shape}"
                )
        if delta_omega is None:
            object.__setattr__(self, "delta_omega_rad_per_fs", omega - float(np.mean(omega)))
        elif delta_omega.shape != expected_shape:
            raise ValueError(
                f"delta_omega_rad_per_fs must have shape {expected_shape}; got {delta_omega.shape}"
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
                raise ValueError("each contribution omega must exactly match pipeline omega values")
            if not np.array_equal(
                np.asarray(contribution.delta_omega_rad_per_fs, dtype=np.float64).reshape(-1),
                np.asarray(self.delta_omega_rad_per_fs, dtype=np.float64).reshape(-1),
            ):
                raise ValueError(
                    "each contribution delta_omega_rad_per_fs must exactly match pipeline values"
                )
            if float(contribution.omega0_rad_per_fs) != float(self.omega0_rad_per_fs):
                raise ValueError(
                    "each contribution omega0_rad_per_fs must match the pipeline value"
                )

        return self


class TaylorPhaseFit(BaseModel):
    """Weighted Taylor fit of total spectral phase on a centered offset-frequency grid."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    omega0_rad_per_fs: float
    delta_omega_rad_per_fs: NDArrayF
    coefficients_rad: NDArrayF
    phi_fit_rad: NDArrayF
    residual_rad: NDArrayF
    weights: NDArrayF
    weighted_rms_rad: float
    max_abs_residual_rad: float

    @field_validator(
        "delta_omega_rad_per_fs",
        "coefficients_rad",
        "phi_fit_rad",
        "residual_rad",
        "weights",
        mode="before",
    )
    @classmethod
    def _normalize_fields(cls, value: object) -> NDArrayF:
        return _coerce_vector(value)

    @model_validator(mode="after")
    def _check_shapes(self) -> TaylorPhaseFit:
        expected = self.delta_omega_rad_per_fs.shape
        for field_name in ("phi_fit_rad", "residual_rad", "weights"):
            field_value = getattr(self, field_name)
            if field_value.shape != expected:
                raise ValueError(
                    f"{field_name} must have shape {expected}; got {field_value.shape}"
                )
        if self.coefficients_rad.size < 5:
            raise ValueError("TaylorPhaseFit.coefficients_rad must contain at least 5 terms.")
        return self


class AbcdefRunResult(BaseModel):
    """Standalone run output: mutated laser state plus full optics-chain bookkeeping."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    initial_state: StandaloneLaserState
    final_state: StandaloneLaserState
    pipeline_result: PipelineResult
    fit: TaylorPhaseFit
