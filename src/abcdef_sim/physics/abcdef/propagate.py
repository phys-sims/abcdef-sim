from __future__ import annotations

from abcdef_sim.physics.abcdef.state import DispersionRayState


def propagate_state_abcd_dispersion(
    state: DispersionRayState,
    *,
    matrix: object,
    phase_coefficients: tuple[float, ...] = (),
) -> DispersionRayState:
    """Placeholder for coupled geometry + dispersion propagation.

    Assumptions:
      - Intended to combine paraxial ABCD transport with frequency-dependent phase.

    Units/sign conventions:
      - Must follow ECO-0001 units for ``omega`` (rad/fs), delays (fs), and lengths.
      - Dispersion coefficient signs should follow the conventions documented in
        ADR-0007 and the ECO-0001 reference.

    Equation reference:
      - Planned implementation: geometric update via 2x2 matrix and phase update via
        Taylor expansion ``phi(omega)``.
    """

    raise NotImplementedError(
        "Dispersion-aware propagation is intentionally a documented placeholder."
    )
