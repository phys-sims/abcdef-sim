"""ABCD+dispersion physics kernels, adapters, and result helpers."""

from abcdef_sim.physics.abcdef.dispersion import (
    evaluate_phase_polynomial,
    fit_phase_taylor,
    fod_from_phase_coeffs,
    gdd_from_phase_coeffs,
    phase_polynomial,
    tod_from_phase_coeffs,
)
from abcdef_sim.physics.abcdef.treacy import (
    TreacyAnalyticMetrics,
    compute_treacy_analytic_metrics,
    phase_from_treacy_dispersion,
)

from . import adapters, phase_terms, postprocess, propagation, pulse, treacy
from .postprocess import compute_pipeline_result
from .propagation import propagate_step

__all__ = [
    "adapters",
    "compute_pipeline_result",
    "compute_treacy_analytic_metrics",
    "evaluate_phase_polynomial",
    "fit_phase_taylor",
    "fod_from_phase_coeffs",
    "gdd_from_phase_coeffs",
    "phase_polynomial",
    "phase_from_treacy_dispersion",
    "phase_terms",
    "postprocess",
    "pulse",
    "propagate_step",
    "propagation",
    "TreacyAnalyticMetrics",
    "tod_from_phase_coeffs",
    "treacy",
]
