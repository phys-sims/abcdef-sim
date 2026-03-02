from __future__ import annotations

from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

CanonicalOpticKind = Literal["FreeSpace", "Grating"]
OpticKind = Literal["FreeSpace", "Grating", "free_space", "grating"]
NDArrayF = npt.NDArray[np.float64]

_KIND_ALIASES: dict[str, CanonicalOpticKind] = {
    "FreeSpace": "FreeSpace",
    "free_space": "FreeSpace",
    "Grating": "Grating",
    "grating": "Grating",
}


class OpticSpec(BaseModel):
    """
    Pure-data description of an optic instance in a system.

    - kind: selects which Optic subclass to instantiate
    - instance_name: unique identifier within a preset (also used for caching identity)
    - params: numeric parameters needed to construct the optic
    - tags: metadata (e.g., "expensive", "requires_gpu", etc.)

    Canonical internal kind values are PascalCase (e.g., "FreeSpace").
    """

    model_config = ConfigDict(frozen=True)

    kind: OpticKind
    instance_name: str
    params: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)

    @field_validator("kind", mode="before")
    @classmethod
    def _normalize_kind(cls, value: Any) -> CanonicalOpticKind:
        if not isinstance(value, str):
            raise TypeError("OpticSpec.kind must be a string.")
        try:
            return _KIND_ALIASES[value]
        except KeyError as e:
            allowed = ", ".join(repr(k) for k in _KIND_ALIASES)
            raise ValueError(f"Unknown OpticSpec.kind {value!r}. Allowed values: {allowed}.") from e


class SystemPreset(BaseModel):
    """
    Pure-data optics chain.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    optics: tuple[OpticSpec, ...]
    tags: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_instance_names(self) -> SystemPreset:
        names = [o.instance_name for o in self.optics]
        if len(names) != len(set(names)):
            raise ValueError("SystemPreset.optics must have unique instance_name values.")
        return self


class LaserSpec(BaseModel):
    """
    Pure-data laser definition.

    Keep immutable and hashable - part of initial state.
    """

    model_config = ConfigDict(frozen=True)

    w0: float
    span: float
    N: int

    pulse: dict[str, float] = Field(default_factory=dict)
    beam: dict[str, float] = Field(default_factory=dict)

    def omega(self) -> NDArrayF:
        return np.linspace(self.w0 - self.span, self.w0 + self.span, self.N, dtype=float)
