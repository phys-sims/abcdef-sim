from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class DispersionRayState:
    """Minimal dispersion-aware ray state.

    Assumptions:
      - First-order paraxial geometry is tracked with ``y`` and ``theta``.
      - Spectral dependence is represented by angular frequency ``omega``.

    Units/sign conventions:
      - ``y`` uses ECO-0001 internal length units.
      - ``theta`` is in radians.
      - ``omega`` is angular frequency in rad/fs when using ECO-0001 defaults.
      - ``phase_rad`` accumulates spectral phase in radians.
      - ``time_delay_fs`` accumulates group delay in femtoseconds.
      - ``opl`` stores optical path length in internal length units.
    """

    y: float
    theta: float
    omega: float
    phase_rad: float = 0.0
    time_delay_fs: float = 0.0
    opl: float = 0.0
