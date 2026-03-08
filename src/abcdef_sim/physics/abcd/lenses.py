from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from typing import Final

import numpy as np
import numpy.typing as npt

from abcdef_sim.physics.abcd.gaussian import q_propagate, sample_beam_radius_profile
from abcdef_sim.physics.abcd.matrices import compose, free_space, thick_lens

NDArrayF = npt.NDArray[np.float64]
EPS: Final[float] = 1e-12


@dataclass(frozen=True, slots=True)
class SellmeierMaterial:
    """Sellmeier refractive-index coefficients in one explicit length system.

    Units/sign conventions:
      - ``wavelength`` passed to ``refractive_index_sellmeier`` must use the
        same length unit as ``sqrt(c_terms[i])``.
      - No implicit nm/mm conversion is performed in this module.
    """

    name: str
    b_terms: tuple[float, ...]
    c_terms: tuple[float, ...]

    def __post_init__(self) -> None:
        b_terms = tuple(float(value) for value in self.b_terms)
        c_terms = tuple(float(value) for value in self.c_terms)
        if not b_terms:
            raise ValueError("b_terms must be non-empty")
        if len(b_terms) != len(c_terms):
            raise ValueError("b_terms and c_terms must have the same length")
        if any(value <= 0.0 for value in c_terms):
            raise ValueError("c_terms must be > 0")

        object.__setattr__(self, "b_terms", b_terms)
        object.__setattr__(self, "c_terms", c_terms)

    def refractive_index(self, wavelength: float | npt.ArrayLike) -> float | NDArrayF:
        """Evaluate the refractive index for the provided wavelength(s)."""

        return refractive_index_sellmeier(self, wavelength)


def refractive_index_sellmeier(
    material: SellmeierMaterial,
    wavelength: float | npt.ArrayLike,
) -> float | NDArrayF:
    """Evaluate a Sellmeier refractive index model."""

    wavelength_arr = np.asarray(wavelength, dtype=float)
    scalar = wavelength_arr.ndim == 0
    wavelength_values = wavelength_arr.reshape(1) if scalar else wavelength_arr
    if np.any(wavelength_values <= 0.0):
        raise ValueError("wavelength must be > 0")

    lambda_sq = np.square(wavelength_values)
    n_sq = np.ones_like(wavelength_values, dtype=float)
    for b_term, c_term in zip(material.b_terms, material.c_terms, strict=True):
        denom = lambda_sq - c_term
        if np.any(np.abs(denom) <= EPS):
            raise ValueError("wavelength is singular for the provided Sellmeier coefficients")
        n_sq = n_sq + b_term * lambda_sq / denom

    result = np.sqrt(n_sq)
    return float(result[0]) if scalar else result


def _resolve_refractive_index(
    refractive_index: float | SellmeierMaterial,
    wavelength: float | None,
) -> float:
    if isinstance(refractive_index, SellmeierMaterial):
        if wavelength is None:
            raise ValueError("wavelength is required for Sellmeier materials")
        return float(refractive_index_sellmeier(refractive_index, wavelength))

    if not isinstance(refractive_index, Real):
        raise TypeError("refractive_index must be a real scalar or SellmeierMaterial")
    return float(refractive_index)


@dataclass(frozen=True, slots=True)
class ThickLensSpec:
    """Structured thick-lens specification for wavelength-aware assembly."""

    refractive_index: float | SellmeierMaterial
    R1: float | None
    R2: float | None
    thickness: float
    n_in: float = 1.0
    n_out: float = 1.0

    def __post_init__(self) -> None:
        thickness = float(self.thickness)
        n_in = float(self.n_in)
        n_out = float(self.n_out)
        if thickness < 0.0:
            raise ValueError("thickness must be >= 0")
        if n_in <= 0.0:
            raise ValueError("n_in must be > 0")
        if n_out <= 0.0:
            raise ValueError("n_out must be > 0")
        if self.R1 is not None and abs(float(self.R1)) <= EPS:
            raise ValueError("R1 must be non-zero when provided")
        if self.R2 is not None and abs(float(self.R2)) <= EPS:
            raise ValueError("R2 must be non-zero when provided")
        if isinstance(self.refractive_index, Real) and float(self.refractive_index) <= 0.0:
            raise ValueError("refractive_index must be > 0")

        object.__setattr__(self, "thickness", thickness)
        object.__setattr__(self, "n_in", n_in)
        object.__setattr__(self, "n_out", n_out)
        object.__setattr__(self, "R1", None if self.R1 is None else float(self.R1))
        object.__setattr__(self, "R2", None if self.R2 is None else float(self.R2))
        if isinstance(self.refractive_index, Real):
            object.__setattr__(self, "refractive_index", float(self.refractive_index))


@dataclass(frozen=True, slots=True)
class DoubletAssembly:
    """Composition of two complete thick-lens elements plus optional spacing.

    Notes:
      - This helper composes two standalone thick-lens elements. Boundary media
        are therefore explicit on each element via ``n_in`` and ``n_out``.
      - A zero gap does not collapse those boundary media into a single cemented
        internal interface; it models two element boundaries at the same plane.
    """

    first: ThickLensSpec
    second: ThickLensSpec
    gap: float = 0.0

    def __post_init__(self) -> None:
        gap = float(self.gap)
        if gap < 0.0:
            raise ValueError("gap must be >= 0")
        object.__setattr__(self, "gap", gap)


def thick_lens_matrix_for_spec(spec: ThickLensSpec, wavelength: float | None = None) -> NDArrayF:
    """Build a thick-lens ABCD matrix from a structured lens specification."""

    n_lens = _resolve_refractive_index(spec.refractive_index, wavelength)
    return thick_lens(
        n_lens=n_lens,
        R1=spec.R1,
        R2=spec.R2,
        thickness=spec.thickness,
        n_in=spec.n_in,
        n_out=spec.n_out,
    )


def doublet_matrix(spec: DoubletAssembly, wavelength: float | None = None) -> NDArrayF:
    """Build a two-element ABCD matrix in optical traversal order."""

    matrices = [thick_lens_matrix_for_spec(spec.first, wavelength)]
    if spec.gap > 0.0:
        matrices.append(free_space(spec.gap))
    matrices.append(thick_lens_matrix_for_spec(spec.second, wavelength))
    return compose(*matrices)


def propagate_q_through_thick_lens(
    q_in: complex,
    spec: ThickLensSpec,
    wavelength: float | None = None,
) -> complex:
    """Propagate a q-parameter through a structured thick lens."""

    return q_propagate(q_in, thick_lens_matrix_for_spec(spec, wavelength))


def propagate_q_through_doublet(
    q_in: complex,
    spec: DoubletAssembly,
    wavelength: float | None = None,
) -> complex:
    """Propagate a q-parameter through a two-element lens assembly."""

    return q_propagate(q_in, doublet_matrix(spec, wavelength))


def sample_thick_lens_beam_radius_profile(
    q_in: complex,
    spec: ThickLensSpec,
    wavelength: float,
    z_samples: npt.ArrayLike,
) -> NDArrayF:
    """Sample beam radius after a structured thick lens."""

    q_out = propagate_q_through_thick_lens(q_in, spec, wavelength)
    return sample_beam_radius_profile(q_out, wavelength, z_samples, n=spec.n_out)


def sample_doublet_beam_radius_profile(
    q_in: complex,
    spec: DoubletAssembly,
    wavelength: float,
    z_samples: npt.ArrayLike,
) -> NDArrayF:
    """Sample beam radius after a structured two-element lens assembly."""

    q_out = propagate_q_through_doublet(q_in, spec, wavelength)
    return sample_beam_radius_profile(q_out, wavelength, z_samples, n=spec.second.n_out)
