from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt

from abcdef_sim.data_models.results import AbcdefRunResult
from abcdef_sim.physics.abcd.gaussian import beam_radius_from_q, q_from_waist, q_propagate
from abcdef_sim.physics.abcdef.phase_terms import martinez_k_center

NDArrayF = npt.NDArray[np.float64]
NDArrayC = npt.NDArray[np.complex128]
PhaseVariant = Literal["full", "without_phi2"]
_C_UM_PER_FS = 0.299792458
_DEFAULT_X_MIN_SAMPLES = 512
_DEFAULT_X_MAX_SAMPLES = 8192
_DEFAULT_X_SAMPLES_PER_RADIUS = 6.0
_DEFAULT_X_CHUNK_SIZE = 256
_SPATIAL_WEIGHT_FLOOR = np.finfo(np.float64).tiny

__all__ = [
    "OutputPlaneField1D",
    "OutputPlaneFieldSummary",
    "OutputPlaneSpatialMetrics",
    "build_output_plane_field_1d",
    "summarize_output_plane_field",
    "summarize_output_plane_geometry",
]


@dataclass(frozen=True, slots=True)
class OutputPlaneSpatialMetrics:
    x_centroid_span_um: float
    x_centroid_slope_um_per_rad_per_fs: float
    x_prime_span: float
    x_prime_slope_per_rad_per_fs: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True, slots=True)
class OutputPlaneFieldSummary:
    phase_variant: PhaseVariant
    x_centroid_span_um: float
    x_centroid_slope_um_per_rad_per_fs: float
    x_prime_span: float
    x_prime_slope_per_rad_per_fs: float
    pulse_front_tilt_fs_per_um: float
    time_centroid_span_fs: float
    mean_mode_overlap_with_center: float
    center_frequency_x_um: float
    center_frequency_x_prime: float

    def to_dict(self) -> dict[str, float | str]:
        payload = asdict(self)
        payload["phase_variant"] = str(self.phase_variant)
        return payload


@dataclass(frozen=True, slots=True)
class OutputPlaneField1D:
    phase_variant: PhaseVariant
    x_um: NDArrayF
    omega_rad_per_fs: NDArrayF
    delta_omega_rad_per_fs: NDArrayF
    t_fs: NDArrayF
    field_x_omega: NDArrayC
    field_x_t: NDArrayC
    intensity_x_omega: NDArrayF
    intensity_x_t: NDArrayF
    spectral_power_au: NDArrayF
    x_out_um: NDArrayF
    x_prime_out: NDArrayF
    w_out_um: NDArrayF
    q_out_um: NDArrayC
    x_centroids_omega_um: NDArrayF
    x_rms_omega_um: NDArrayF
    time_centroids_x_fs: NDArrayF
    time_rms_x_fs: NDArrayF


def build_output_plane_field_1d(
    run_result: AbcdefRunResult,
    *,
    x_grid_um: npt.ArrayLike | None = None,
    phase_variant: PhaseVariant = "full",
) -> OutputPlaneField1D:
    omega = np.asarray(run_result.pipeline_result.omega, dtype=np.float64).reshape(-1)
    delta_omega = np.asarray(
        run_result.pipeline_result.delta_omega_rad_per_fs,
        dtype=np.float64,
    ).reshape(-1)
    t_fs = np.asarray(run_result.initial_state.pulse.grid.t, dtype=np.float64).reshape(-1)
    spectral_power = _spectral_power_au(run_result)
    spectral_field = _spectral_field_input(run_result)
    x_out_um, x_prime_out, w_out_um, q_out_um = _output_plane_geometry(run_result)

    if x_grid_um is None:
        x_um = _default_x_grid_um(x_out_um=x_out_um, w_out_um=w_out_um)
    else:
        x_um = np.asarray(x_grid_um, dtype=np.float64).reshape(-1)
    if x_um.size < 2:
        raise ValueError("x_grid_um must contain at least two samples.")

    selected_phase = _selected_phase_rad(run_result, phase_variant=phase_variant)
    phi4_on_axis = np.zeros_like(selected_phase)
    if run_result.pipeline_result.phi4_rad is not None:
        phi4_on_axis = np.asarray(run_result.pipeline_result.phi4_rad, dtype=np.float64).reshape(-1)
    base_phase = selected_phase - phi4_on_axis
    scalar_prefactor = spectral_field * np.exp(1j * base_phase)
    k_center = martinez_k_center(omega)
    normalization = (2.0 / (np.pi * w_out_um**2)) ** 0.25

    field_x_omega = np.empty((x_um.size, omega.size), dtype=np.complex128)
    for start in range(0, x_um.size, _DEFAULT_X_CHUNK_SIZE):
        stop = min(start + _DEFAULT_X_CHUNK_SIZE, x_um.size)
        x_chunk = x_um[start:stop, None]
        dx_chunk = x_chunk - x_out_um[None, :]
        amplitude = normalization[None, :] * np.exp(-(dx_chunk**2) / (w_out_um[None, :] ** 2))
        transverse_phase = np.real(k_center * dx_chunk**2 / (2.0 * q_out_um[None, :]))
        field_x_omega[start:stop] = amplitude * scalar_prefactor[None, :] * np.exp(
            1j * transverse_phase
        )

    intensity_x_omega = np.abs(field_x_omega) ** 2
    field_x_t = np.fft.fftshift(
        np.fft.ifft(np.fft.ifftshift(field_x_omega, axes=1), axis=1),
        axes=1,
    )
    intensity_x_t = np.abs(field_x_t) ** 2
    x_centroids = _weighted_centroid(samples=x_um, values=intensity_x_omega, axis=0)
    x_rms = _weighted_rms(
        samples=x_um,
        values=intensity_x_omega,
        centroids=x_centroids,
        axis=0,
    )
    time_centroids = _weighted_centroid(samples=t_fs, values=intensity_x_t, axis=1)
    time_rms = _weighted_rms(samples=t_fs, values=intensity_x_t, centroids=time_centroids, axis=1)

    return OutputPlaneField1D(
        phase_variant=phase_variant,
        x_um=x_um,
        omega_rad_per_fs=omega,
        delta_omega_rad_per_fs=delta_omega,
        t_fs=t_fs,
        field_x_omega=field_x_omega,
        field_x_t=field_x_t,
        intensity_x_omega=intensity_x_omega,
        intensity_x_t=intensity_x_t,
        spectral_power_au=spectral_power,
        x_out_um=x_out_um,
        x_prime_out=x_prime_out,
        w_out_um=w_out_um,
        q_out_um=q_out_um,
        x_centroids_omega_um=x_centroids,
        x_rms_omega_um=x_rms,
        time_centroids_x_fs=time_centroids,
        time_rms_x_fs=time_rms,
    )


def summarize_output_plane_geometry(run_result: AbcdefRunResult) -> OutputPlaneSpatialMetrics:
    x_out_um, x_prime_out, _, _ = _output_plane_geometry(run_result)
    delta_omega = np.asarray(
        run_result.pipeline_result.delta_omega_rad_per_fs,
        dtype=np.float64,
    ).reshape(-1)
    spectral_power = _spectral_power_au(run_result)
    return OutputPlaneSpatialMetrics(
        x_centroid_span_um=float(np.max(x_out_um) - np.min(x_out_um)),
        x_centroid_slope_um_per_rad_per_fs=_weighted_linear_slope(
            x=delta_omega,
            y=x_out_um,
            weights=spectral_power,
        ),
        x_prime_span=float(np.max(x_prime_out) - np.min(x_prime_out)),
        x_prime_slope_per_rad_per_fs=_weighted_linear_slope(
            x=delta_omega,
            y=x_prime_out,
            weights=spectral_power,
        ),
    )


def summarize_output_plane_field(field: OutputPlaneField1D) -> OutputPlaneFieldSummary:
    weights_x = np.sum(field.intensity_x_t, axis=1)
    spatial_weights = np.asarray(field.spectral_power_au, dtype=np.float64).reshape(-1)
    spatial_metrics = OutputPlaneSpatialMetrics(
        x_centroid_span_um=float(np.max(field.x_centroids_omega_um) - np.min(field.x_centroids_omega_um)),
        x_centroid_slope_um_per_rad_per_fs=_weighted_linear_slope(
            x=field.delta_omega_rad_per_fs,
            y=field.x_centroids_omega_um,
            weights=spatial_weights,
        ),
        x_prime_span=float(np.max(field.x_prime_out) - np.min(field.x_prime_out)),
        x_prime_slope_per_rad_per_fs=_weighted_linear_slope(
            x=field.delta_omega_rad_per_fs,
            y=field.x_prime_out,
            weights=spatial_weights,
        ),
    )
    center_idx = int(np.argmin(np.abs(field.delta_omega_rad_per_fs)))
    normalized_modes = _normalize_modes(field.field_x_omega, x_um=field.x_um)
    overlaps = np.abs(
        np.trapezoid(
            np.conjugate(normalized_modes[:, center_idx])[:, None] * normalized_modes,
            x=field.x_um,
            axis=0,
        )
    ) ** 2
    mean_mode_overlap = float(np.average(overlaps, weights=spatial_weights))

    return OutputPlaneFieldSummary(
        phase_variant=field.phase_variant,
        x_centroid_span_um=spatial_metrics.x_centroid_span_um,
        x_centroid_slope_um_per_rad_per_fs=spatial_metrics.x_centroid_slope_um_per_rad_per_fs,
        x_prime_span=spatial_metrics.x_prime_span,
        x_prime_slope_per_rad_per_fs=spatial_metrics.x_prime_slope_per_rad_per_fs,
        pulse_front_tilt_fs_per_um=_weighted_linear_slope(
            x=field.x_um,
            y=field.time_centroids_x_fs,
            weights=weights_x,
        ),
        time_centroid_span_fs=float(np.max(field.time_centroids_x_fs) - np.min(field.time_centroids_x_fs)),
        mean_mode_overlap_with_center=mean_mode_overlap,
        center_frequency_x_um=float(field.x_centroids_omega_um[center_idx]),
        center_frequency_x_prime=float(field.x_prime_out[center_idx]),
    )


def _default_x_grid_um(*, x_out_um: NDArrayF, w_out_um: NDArrayF) -> NDArrayF:
    sigma_max = max(float(np.max(w_out_um)) * 0.5, 1e-6)
    lower = float(np.min(x_out_um) - 6.0 * sigma_max)
    upper = float(np.max(x_out_um) + 6.0 * sigma_max)
    if upper <= lower:
        lower -= 6.0 * sigma_max
        upper += 6.0 * sigma_max
    window_um = upper - lower
    min_radius_um = max(float(np.min(w_out_um)), 1e-6)
    target_dx_um = max(min_radius_um / _DEFAULT_X_SAMPLES_PER_RADIUS, 1e-6)
    required_samples = int(np.ceil(window_um / target_dx_um)) + 1
    n_samples = min(
        _DEFAULT_X_MAX_SAMPLES,
        max(_DEFAULT_X_MIN_SAMPLES, required_samples),
    )
    return np.linspace(lower, upper, n_samples, dtype=np.float64)


def _output_plane_geometry(run_result: AbcdefRunResult) -> tuple[NDArrayF, NDArrayF, NDArrayF, NDArrayC]:
    meta = run_result.final_state.meta.get("abcdef", {})
    x_out = meta.get("x_out_um")
    x_prime = meta.get("x_prime_out")
    w_out = meta.get("w_out_um")
    q_out_real = meta.get("q_out_real_um")
    q_out_imag = meta.get("q_out_imag_um")
    if (
        isinstance(x_out, list)
        and isinstance(x_prime, list)
        and isinstance(w_out, list)
        and isinstance(q_out_real, list)
        and isinstance(q_out_imag, list)
    ):
        return (
            np.asarray(x_out, dtype=np.float64),
            np.asarray(x_prime, dtype=np.float64),
            np.asarray(w_out, dtype=np.float64),
            np.asarray(q_out_real, dtype=np.float64) + 1j * np.asarray(q_out_imag, dtype=np.float64),
        )

    omega = np.asarray(run_result.pipeline_result.omega, dtype=np.float64).reshape(-1)
    system = np.asarray(run_result.pipeline_result.final_state.system, dtype=np.float64)
    beam_radius_um = float(run_result.initial_state.beam.radius_mm) * 1e3
    m2 = float(run_result.initial_state.beam.m2)
    wavelength_um = (2.0 * np.pi * _C_UM_PER_FS) / omega
    effective_wavelength_um = wavelength_um * m2
    q_out = np.empty(omega.size, dtype=np.complex128)
    w_out_arr = np.empty(omega.size, dtype=np.float64)
    for idx, wavelength_i in enumerate(effective_wavelength_um):
        q_in_i = q_from_waist(beam_radius_um, float(wavelength_i))
        q_out_i = q_propagate(q_in_i, system[idx, :2, :2])
        q_out[idx] = q_out_i
        w_out_arr[idx] = beam_radius_from_q(q_out_i, float(wavelength_i))

    rays = np.asarray(run_result.pipeline_result.final_state.rays, dtype=np.float64)
    return (
        rays[:, 0, 0],
        rays[:, 1, 0],
        w_out_arr,
        q_out,
    )


def _selected_phase_rad(run_result: AbcdefRunResult, *, phase_variant: PhaseVariant) -> NDArrayF:
    phase = np.asarray(run_result.pipeline_result.phi_total_rad, dtype=np.float64).reshape(-1)
    if phase_variant == "full":
        return phase
    phi2 = run_result.pipeline_result.phi2_rad
    if phi2 is None:
        return phase
    return phase - np.asarray(phi2, dtype=np.float64).reshape(-1)


def _spectral_field_input(run_result: AbcdefRunResult) -> NDArrayC:
    field = np.asarray(run_result.initial_state.pulse.field_w, dtype=np.complex128).reshape(-1).copy()
    for contribution in run_result.pipeline_result.contributions:
        if contribution.filter_amp is not None:
            field = field * np.asarray(contribution.filter_amp, dtype=np.float64).reshape(-1)
    return field


def _spectral_power_au(run_result: AbcdefRunResult) -> NDArrayF:
    meta = run_result.final_state.meta.get("abcdef", {})
    stored_power = meta.get("spectral_power_au")
    if isinstance(stored_power, list):
        return np.asarray(stored_power, dtype=np.float64)

    power = np.abs(_spectral_field_input(run_result)) ** 2
    return np.asarray(power, dtype=np.float64)


def _weighted_centroid(samples: NDArrayF, values: NDArrayF, *, axis: int) -> NDArrayF:
    weights = np.asarray(values, dtype=np.float64)
    if axis == 0:
        numerator = np.sum(samples[:, None] * weights, axis=0)
        denominator = np.sum(weights, axis=0)
    else:
        numerator = np.sum(samples[None, :] * weights, axis=1)
        denominator = np.sum(weights, axis=1)
    denominator = np.maximum(denominator, _SPATIAL_WEIGHT_FLOOR)
    return numerator / denominator


def _weighted_rms(
    *,
    samples: NDArrayF,
    values: NDArrayF,
    centroids: NDArrayF,
    axis: int,
) -> NDArrayF:
    weights = np.asarray(values, dtype=np.float64)
    if axis == 0:
        variance = np.sum(((samples[:, None] - centroids[None, :]) ** 2) * weights, axis=0)
        denominator = np.sum(weights, axis=0)
    else:
        variance = np.sum(((samples[None, :] - centroids[:, None]) ** 2) * weights, axis=1)
        denominator = np.sum(weights, axis=1)
    denominator = np.maximum(denominator, _SPATIAL_WEIGHT_FLOOR)
    return np.sqrt(np.maximum(variance / denominator, 0.0))


def _weighted_linear_slope(*, x: NDArrayF, y: NDArrayF, weights: NDArrayF) -> float:
    x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
    y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
    w_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    if x_arr.size != y_arr.size or x_arr.size != w_arr.size:
        raise ValueError("x, y, and weights must have matching 1D shapes.")

    weight_sum = float(np.sum(w_arr))
    if weight_sum <= 0.0:
        return 0.0

    x_mean = float(np.sum(w_arr * x_arr) / weight_sum)
    y_mean = float(np.sum(w_arr * y_arr) / weight_sum)
    numerator = float(np.sum(w_arr * (x_arr - x_mean) * (y_arr - y_mean)))
    denominator = float(np.sum(w_arr * (x_arr - x_mean) ** 2))
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def _normalize_modes(field_x_omega: NDArrayC, *, x_um: NDArrayF) -> NDArrayC:
    power = np.trapezoid(np.abs(field_x_omega) ** 2, x=x_um, axis=0)
    denom = np.sqrt(np.maximum(power, _SPATIAL_WEIGHT_FLOOR))
    return field_x_omega / denom[None, :]
