"""ABCD+dispersion physics kernels, adapters, and result helpers."""

from abcdef_sim.physics.abcdef.dispersion import (
    gdd_from_phase_coeffs,
    phase_polynomial,
    tod_from_phase_coeffs,
)

from . import adapters, phase_terms, postprocess, propagation
from .postprocess import compute_pipeline_result
from .propagation import propagate_step

__all__ = [
    "adapters",
    "compute_pipeline_result",
    "gdd_from_phase_coeffs",
    "phase_polynomial",
    "phase_terms",
    "postprocess",
    "propagate_step",
    "propagation",
    "tod_from_phase_coeffs",
]
