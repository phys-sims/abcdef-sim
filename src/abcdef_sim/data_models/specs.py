from __future__ import annotations

from typing import Annotated, Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

OpticKind = Literal["FreeSpace", "Grating", "ThickLens"]
NDArrayF = npt.NDArray[np.float64]


class OpticSpec(BaseModel):
    """
    Pure-data description of a normalized runtime optic instance.

    ``params`` is intentionally permissive so the public typed config layer can
    normalize into one immutable cache-friendly representation without losing
    structured optic data such as Sellmeier coefficients.
    """

    model_config = ConfigDict(frozen=True)

    kind: OpticKind
    instance_name: str
    params: dict[str, Any] = Field(default_factory=dict)
    tags: dict[str, Any] = Field(default_factory=dict)


class SystemPreset(BaseModel):
    """Pure-data optics chain."""

    model_config = ConfigDict(frozen=True)

    name: str
    optics: tuple[OpticSpec, ...]
    tags: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _unique_instance_names(self) -> SystemPreset:
        names = [optic.instance_name for optic in self.optics]
        if len(names) != len(set(names)):
            raise ValueError("SystemPreset.optics must have unique instance_name values.")
        return self


class LaserSpec(BaseModel):
    """
    Internal spectral-grid definition used by assembly and cfg generation.

    Conventions:
      - ``w0`` is the optical center angular frequency ``omega0``.
      - ``span`` is the half-width of the centered offset grid ``Δω``.
      - ``omega()`` returns absolute optical frequencies ``omega0 + Δω``.
      - ``delta_omega()`` returns a centered offset-frequency grid.
    """

    model_config = ConfigDict(frozen=True)

    w0: float
    span: float
    N: int
    delta_omega_rad_per_fs: tuple[float, ...] | None = None

    pulse: dict[str, float] = Field(default_factory=dict)
    beam: dict[str, float] = Field(default_factory=dict)

    @field_validator("N")
    @classmethod
    def _validate_sample_count(cls, value: int) -> int:
        if value < 1:
            raise ValueError("LaserSpec.N must be >= 1.")
        return value

    @field_validator("span")
    @classmethod
    def _validate_span(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("LaserSpec.span must be >= 0.")
        return value

    @field_validator("delta_omega_rad_per_fs")
    @classmethod
    def _validate_delta_grid(
        cls,
        value: tuple[float, ...] | None,
    ) -> tuple[float, ...] | None:
        if value is None:
            return value
        if len(value) == 0:
            raise ValueError("LaserSpec.delta_omega_rad_per_fs must be non-empty when provided.")
        return tuple(float(item) for item in value)

    @model_validator(mode="after")
    def _validate_delta_grid_length(self) -> LaserSpec:
        if self.delta_omega_rad_per_fs is not None and len(self.delta_omega_rad_per_fs) != self.N:
            raise ValueError(
                "LaserSpec.delta_omega_rad_per_fs length must match N: "
                f"expected {self.N}, got {len(self.delta_omega_rad_per_fs)}"
            )
        return self

    @property
    def omega0_rad_per_fs(self) -> float:
        return float(self.w0)

    def delta_omega(self) -> NDArrayF:
        if self.delta_omega_rad_per_fs is not None:
            return np.asarray(self.delta_omega_rad_per_fs, dtype=np.float64)
        return np.linspace(-self.span, self.span, self.N, dtype=float)

    def omega(self) -> NDArrayF:
        return self.omega0_rad_per_fs + self.delta_omega()


class InputRayCfg(BaseModel):
    """Single fixed Martinez input ray for the full optics chain."""

    model_config = ConfigDict(frozen=True)

    x: float = 0.0
    x_prime: float = 0.0

    def to_column(self, *, batch_size: int) -> NDArrayF:
        ray = np.array([self.x, self.x_prime, 1.0], dtype=np.float64)
        return np.repeat(ray[None, :, None], batch_size, axis=0)


class SellmeierMaterialCfg(BaseModel):
    """Sellmeier coefficients with wavelength inputs expressed in micrometers."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["sellmeier"] = "sellmeier"
    b_terms: tuple[float, ...]
    c_terms_um2: tuple[float, ...]

    @model_validator(mode="after")
    def _validate_lengths(self) -> SellmeierMaterialCfg:
        if not self.b_terms:
            raise ValueError("SellmeierMaterialCfg.b_terms must be non-empty.")
        if len(self.b_terms) != len(self.c_terms_um2):
            raise ValueError("SellmeierMaterialCfg terms must have matching lengths.")
        return self

    def to_params(self) -> dict[str, Any]:
        return {
            "refractive_index_model": {
                "kind": self.kind,
                "b_terms": list(self.b_terms),
                "c_terms_um2": list(self.c_terms_um2),
            }
        }


class FreeSpaceCfg(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["free_space"] = "free_space"
    instance_name: str
    length: float
    medium_refractive_index: float = 1.0

    @field_validator("length")
    @classmethod
    def _validate_length(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("FreeSpaceCfg.length must be >= 0.")
        return value

    @field_validator("medium_refractive_index")
    @classmethod
    def _validate_medium_index(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("FreeSpaceCfg.medium_refractive_index must be > 0.")
        return value

    def to_spec(self) -> OpticSpec:
        return OpticSpec(
            kind="FreeSpace",
            instance_name=self.instance_name,
            params={
                "L": float(self.length),
                "medium_refractive_index": float(self.medium_refractive_index),
            },
        )


class GratingCfg(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["grating"] = "grating"
    instance_name: str
    line_density_lpmm: float
    incidence_angle_deg: float
    diffraction_order: int = -1
    immersion_refractive_index: float = 1.0

    @field_validator("line_density_lpmm")
    @classmethod
    def _validate_line_density(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("GratingCfg.line_density_lpmm must be > 0.")
        return value

    @field_validator("immersion_refractive_index")
    @classmethod
    def _validate_immersion_index(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("GratingCfg.immersion_refractive_index must be > 0.")
        return value

    def to_spec(self) -> OpticSpec:
        return OpticSpec(
            kind="Grating",
            instance_name=self.instance_name,
            params={
                "line_density_lpmm": float(self.line_density_lpmm),
                "incidence_angle_deg": float(self.incidence_angle_deg),
                "diffraction_order": int(self.diffraction_order),
                "immersion_refractive_index": float(self.immersion_refractive_index),
            },
        )


class ThickLensCfg(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["thick_lens"] = "thick_lens"
    instance_name: str
    R1: float | None
    R2: float | None
    thickness: float
    n_in: float = 1.0
    n_out: float = 1.0
    refractive_index: float | SellmeierMaterialCfg

    @field_validator("thickness")
    @classmethod
    def _validate_thickness(cls, value: float) -> float:
        if value < 0.0:
            raise ValueError("ThickLensCfg.thickness must be >= 0.")
        return value

    @field_validator("n_in", "n_out")
    @classmethod
    def _validate_positive_indices(cls, value: float) -> float:
        if value <= 0.0:
            raise ValueError("ThickLensCfg medium indices must be > 0.")
        return value

    def to_spec(self) -> OpticSpec:
        refractive_index_model: Any
        if isinstance(self.refractive_index, SellmeierMaterialCfg):
            refractive_index_model = {
                "kind": self.refractive_index.kind,
                "b_terms": list(self.refractive_index.b_terms),
                "c_terms_um2": list(self.refractive_index.c_terms_um2),
            }
        else:
            refractive_index_model = float(self.refractive_index)

        return OpticSpec(
            kind="ThickLens",
            instance_name=self.instance_name,
            params={
                "R1": None if self.R1 is None else float(self.R1),
                "R2": None if self.R2 is None else float(self.R2),
                "thickness": float(self.thickness),
                "n_in": float(self.n_in),
                "n_out": float(self.n_out),
                "refractive_index_model": refractive_index_model,
            },
        )


OpticCfg = Annotated[FreeSpaceCfg | GratingCfg | ThickLensCfg, Field(discriminator="kind")]


class AbcdefCfg(BaseModel):
    """Public typed config boundary for standalone runs and cpa-sim wrapping."""

    model_config = ConfigDict(frozen=True)

    name: str = "abcdef"
    optics: tuple[OpticCfg, ...]
    input_ray: InputRayCfg = Field(default_factory=InputRayCfg)
    tags: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_unique_names(self) -> AbcdefCfg:
        if not self.optics:
            raise ValueError("AbcdefCfg.optics must contain at least one optic.")
        names = [optic.instance_name for optic in self.optics]
        if len(names) != len(set(names)):
            raise ValueError("AbcdefCfg.optics must have unique instance_name values.")
        return self

    def to_preset(self) -> SystemPreset:
        return SystemPreset(
            name=self.name,
            optics=tuple(optic.to_spec() for optic in self.optics),
            tags=dict(self.tags),
        )
