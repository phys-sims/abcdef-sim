from __future__ import annotations

from collections.abc import Sequence


def phase_polynomial(omega: float, omega0: float, coefficients: Sequence[float]) -> float:
    """Evaluate a spectral phase polynomial around ``omega0``.

    Assumptions:
      - Spectral phase is represented as
        ``phi(omega) = sum_k coeff[k] * (omega - omega0)^k / k!``.

    Units/sign conventions:
      - ``omega`` and ``omega0`` are angular frequencies (rad/fs under ECO-0001).
      - ``coeff[2]`` corresponds to GDD and ``coeff[3]`` to TOD under the above
        Taylor-series convention.
      - Positive chirp corresponds to positive slope in instantaneous frequency vs.
        time, consistent with ECO-0001 ADR adoption.
    """

    x = float(omega - omega0)
    fact = 1.0
    power = 1.0
    total = 0.0
    for k, coeff in enumerate(coefficients):
        if k > 0:
            fact *= float(k)
            power *= x
        total += float(coeff) * power / fact
    return total


def gdd_from_phase_coeffs(coefficients: Sequence[float]) -> float:
    """Return GDD (second phase derivative at ``omega0``) from Taylor coefficients."""

    if len(coefficients) < 3:
        return 0.0
    return float(coefficients[2])


def tod_from_phase_coeffs(coefficients: Sequence[float]) -> float:
    """Return TOD (third phase derivative at ``omega0``) from Taylor coefficients."""

    if len(coefficients) < 4:
        return 0.0
    return float(coefficients[3])
