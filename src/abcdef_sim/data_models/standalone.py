from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class PulseGrid(BaseModel):
    model_config = ConfigDict(frozen=True)

    t: list[float]
    w: list[float]
    dt: float
    dw: float
    center_wavelength_nm: float


class PulseSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    shape: Literal["gaussian", "sech2"] = "gaussian"
    peak_power_w: float | None = None
    avg_power_w: float | None = None
    pulse_energy_j: float | None = None
    width_fs: float = 100.0
    center_wavelength_nm: float = 1030.0
    rep_rate_mhz: float = 1.0
    n_samples: int = 256
    time_window_fs: float = 2000.0

    @field_validator("peak_power_w", "avg_power_w", "pulse_energy_j")
    @classmethod
    def _validate_nonnegative_scalars(cls, value: float | None) -> float | None:
        if value is None:
            return value
        if value < 0.0:
            raise ValueError("PulseSpec power and energy values must be >= 0.")
        return value

    @field_validator("width_fs", "center_wavelength_nm", "rep_rate_mhz", "time_window_fs")
    @classmethod
    def _validate_positive_scalars(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("PulseSpec positive scalar fields must be > 0.")
        return value

    @field_validator("n_samples")
    @classmethod
    def _validate_sample_count(cls, value: int) -> int:
        if value < 2:
            raise ValueError("PulseSpec.n_samples must be >= 2.")
        return value

    @model_validator(mode="after")
    def _validate_normalization_inputs(self) -> PulseSpec:
        explicit = [
            self.peak_power_w is not None,
            self.avg_power_w is not None,
            self.pulse_energy_j is not None,
        ]
        if sum(explicit) > 1:
            raise ValueError(
                "Exactly one of peak_power_w, avg_power_w, or pulse_energy_j may be provided."
            )
        return self


class BeamSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    radius_mm: float = 1.0
    m2: float = 1.0

    @field_validator("radius_mm", "m2")
    @classmethod
    def _validate_positive_scalars(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("BeamSpec.radius_mm and BeamSpec.m2 must be > 0.")
        return value


class StandaloneLaserSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    pulse: PulseSpec = Field(default_factory=PulseSpec)
    beam: BeamSpec = Field(default_factory=BeamSpec)


@dataclass(slots=True)
class PulseState:
    grid: PulseGrid
    field_t: np.ndarray
    field_w: np.ndarray
    intensity_t: np.ndarray
    spectrum_w: np.ndarray


@dataclass(slots=True)
class BeamState:
    radius_mm: float
    m2: float


@dataclass(slots=True)
class LaserState:
    pulse: PulseState
    beam: BeamState
    meta: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)

    def deepcopy(self) -> LaserState:
        return copy.deepcopy(self)
