"""ABCD+dispersion physics scaffolding package."""

from abcdef_sim.physics.abcdef import adapters, phase_terms, propagation
from abcdef_sim.physics.abcdef.dispersion import (
    gdd_from_phase_coeffs,
    phase_polynomial,
    tod_from_phase_coeffs,
)
from abcdef_sim.physics.abcdef.propagate import propagate_state_abcd_dispersion
from abcdef_sim.physics.abcdef.state import DispersionRayState

__all__ = [
    "DispersionRayState",
    "adapters",
    "gdd_from_phase_coeffs",
    "phase_polynomial",
    "phase_terms",
    "propagate_state_abcd_dispersion",
    "propagation",
    "tod_from_phase_coeffs",
]
