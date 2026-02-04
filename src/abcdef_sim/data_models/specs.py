from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, Field, ConfigDict, model_validator


OpticKind = Literal["FreeSpace", "Grating"]  # extend as optics added


class OpticSpec(BaseModel):
    """
    Pure-data description of an optic instance in a system.

    - kind: selects which Optic subclass to instantiate
    - instance_name: unique identifier within a preset (also used for caching identity)
    - params: numeric parameters needed to construct the optic
    - tags: metadata (e.g., "expensive", "requires_gpu", etc.)
    """
    model_config = ConfigDict(frozen=True)

    kind: OpticKind
    instance_name: str
    params: dict[str, float] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)


class SystemPreset(BaseModel):
    """
    Pure-data optics chain.
    """
    model_config = ConfigDict(frozen=True)

    name: str
    optics: tuple[OpticSpec, ...]
    tags: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_instance_names(self) -> "SystemPreset":
        names = [o.instance_name for o in self.optics]
        if len(names) != len(set(names)):
            raise ValueError("SystemPreset.optics must have unique instance_name values.")
        return self


class LaserSpec(BaseModel):
    """
    Pure-data laser definition.

    Keep immutable and hashable - part of intial state.
    """
    model_config = ConfigDict(frozen=True)

    w0: float
    span: float
    N: int

    pulse: dict[str, float] = Field(default_factory=dict)
    beam: dict[str, float] = Field(default_factory=dict)

    def omega(self):
        import numpy as np
        return np.linspace(self.w0 - self.span, self.w0 + self.span, self.N, dtype=float)
