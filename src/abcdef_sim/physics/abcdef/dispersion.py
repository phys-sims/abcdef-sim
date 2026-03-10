from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from abcdef_sim.data_models.results import TaylorPhaseFit

NDArrayF = npt.NDArray[np.float64]


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


def evaluate_phase_polynomial(delta_omega: object, coefficients: Sequence[float]) -> NDArrayF:
    """Evaluate a Taylor-series spectral phase directly on a centered ``Δω`` grid."""

    delta_omega_arr = np.asarray(delta_omega, dtype=np.float64).reshape(-1)
    design = _design_matrix(delta_omega_arr, order=len(coefficients) - 1)
    coeff_arr = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    return design @ coeff_arr


def fit_phase_taylor(
    delta_omega: object,
    phi_total_rad: object,
    *,
    omega0_rad_per_fs: float,
    weights: object,
    order: int = 4,
) -> TaylorPhaseFit:
    """Fit a centered Taylor series to a spectral phase curve with sample weights."""

    if order < 0:
        raise ValueError("order must be >= 0")

    delta_omega_arr, phi_arr, weights_arr = _normalize_fit_inputs(
        delta_omega=delta_omega,
        phi_total_rad=phi_total_rad,
        weights=weights,
    )

    design = _design_matrix(delta_omega_arr, order=order)
    coefficients = _solve_weighted_least_squares(design, phi_arr, weights_arr)

    phi_fit = design @ coefficients
    residual = phi_arr - phi_fit
    weighted_rms = float(np.sqrt(np.sum(weights_arr * residual**2) / np.sum(weights_arr)))
    max_abs_residual = float(np.max(np.abs(residual)))

    return TaylorPhaseFit(
        omega0_rad_per_fs=float(omega0_rad_per_fs),
        delta_omega_rad_per_fs=delta_omega_arr,
        coefficients_rad=coefficients,
        phi_fit_rad=phi_fit,
        residual_rad=residual,
        weights=weights_arr,
        weighted_rms_rad=weighted_rms,
        max_abs_residual_rad=max_abs_residual,
    )


def fit_phase_taylor_affine_detrended(
    delta_omega: object,
    phi_total_rad: object,
    *,
    omega0_rad_per_fs: float,
    weights: object,
    order: int = 4,
) -> TaylorPhaseFit:
    """Fit higher-order dispersion after stripping the weighted affine phase component."""

    if order < 0:
        raise ValueError("order must be >= 0")
    if order < 2:
        return fit_phase_taylor(
            delta_omega,
            phi_total_rad,
            omega0_rad_per_fs=omega0_rad_per_fs,
            weights=weights,
            order=order,
        )

    delta_omega_arr, phi_arr, weights_arr = _normalize_fit_inputs(
        delta_omega=delta_omega,
        phi_total_rad=phi_total_rad,
        weights=weights,
    )

    full_design = _design_matrix(delta_omega_arr, order=order)
    affine_design = full_design[:, :2]
    higher_order_design = full_design[:, 2:]
    projected_target = _subtract_weighted_affine_projection(phi_arr, affine_design, weights_arr)
    projected_higher_order_design = np.column_stack(
        [
            _subtract_weighted_affine_projection(
                higher_order_design[:, idx],
                affine_design,
                weights_arr,
            )
            for idx in range(higher_order_design.shape[1])
        ]
    )
    higher_order_coefficients = _solve_weighted_least_squares(
        projected_higher_order_design,
        projected_target,
        weights_arr,
    )
    affine_coefficients = _solve_weighted_least_squares(
        affine_design,
        phi_arr - (higher_order_design @ higher_order_coefficients),
        weights_arr,
    )

    coefficients = np.zeros(order + 1, dtype=np.float64)
    coefficients[:2] = affine_coefficients
    coefficients[2:] = higher_order_coefficients

    phi_fit = full_design @ coefficients
    residual = phi_arr - phi_fit
    weighted_rms = float(np.sqrt(np.sum(weights_arr * residual**2) / np.sum(weights_arr)))
    max_abs_residual = float(np.max(np.abs(residual)))

    return TaylorPhaseFit(
        omega0_rad_per_fs=float(omega0_rad_per_fs),
        delta_omega_rad_per_fs=delta_omega_arr,
        coefficients_rad=coefficients,
        phi_fit_rad=phi_fit,
        residual_rad=residual,
        weights=weights_arr,
        weighted_rms_rad=weighted_rms,
        max_abs_residual_rad=max_abs_residual,
    )


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


def fod_from_phase_coeffs(coefficients: Sequence[float]) -> float:
    """Return FOD (fourth phase derivative at ``omega0``) from Taylor coefficients."""

    if len(coefficients) < 5:
        return 0.0
    return float(coefficients[4])


def _design_matrix(delta_omega: NDArrayF, *, order: int) -> NDArrayF:
    columns: list[NDArrayF] = []
    factorial = 1.0
    power = np.ones_like(delta_omega, dtype=np.float64)
    for k in range(order + 1):
        if k == 0:
            columns.append(np.ones_like(delta_omega, dtype=np.float64))
            continue
        factorial *= float(k)
        power = power * delta_omega
        columns.append(power / factorial)
    return np.column_stack(columns)


def _normalize_fit_inputs(
    *,
    delta_omega: object,
    phi_total_rad: object,
    weights: object,
) -> tuple[NDArrayF, NDArrayF, NDArrayF]:
    delta_omega_arr = np.asarray(delta_omega, dtype=np.float64).reshape(-1)
    phi_arr = np.asarray(phi_total_rad, dtype=np.float64).reshape(-1)
    weights_arr = np.asarray(weights, dtype=np.float64).reshape(-1)

    if delta_omega_arr.shape != phi_arr.shape or weights_arr.shape != phi_arr.shape:
        raise ValueError("delta_omega, phi_total_rad, and weights must share one shape")

    if not np.any(weights_arr > 0.0):
        weights_arr = np.ones_like(phi_arr, dtype=np.float64)
    else:
        weights_arr = np.maximum(weights_arr, 0.0)

    return delta_omega_arr, phi_arr, weights_arr


def _solve_weighted_least_squares(
    design: NDArrayF,
    rhs: NDArrayF,
    weights: NDArrayF,
) -> NDArrayF:
    weight_scale = np.sqrt(weights)
    lhs_weighted = design * weight_scale[:, None]
    rhs_weighted = rhs * weight_scale
    coefficients, *_ = np.linalg.lstsq(lhs_weighted, rhs_weighted, rcond=None)
    return np.asarray(coefficients, dtype=np.float64).reshape(-1)


def _subtract_weighted_affine_projection(
    vector: NDArrayF,
    affine_design: NDArrayF,
    weights: NDArrayF,
) -> NDArrayF:
    affine_coefficients = _solve_weighted_least_squares(affine_design, vector, weights)
    return np.asarray(vector, dtype=np.float64).reshape(-1) - (affine_design @ affine_coefficients)
