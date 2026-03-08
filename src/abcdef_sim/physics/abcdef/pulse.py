from __future__ import annotations

import math

import numpy as np

from abcdef_sim.data_models.standalone import (
    BeamState,
    LaserState,
    PulseGrid,
    PulseState,
    StandaloneLaserSpec,
)
from abcdef_sim.physics.abcd.gaussian import beam_radius_from_q, q_from_waist, q_propagate

_C_UM_PER_FS = 0.299792458
_SECH_FWHM_FACTOR = 2.0 * np.arccosh(np.sqrt(2.0))


def omega0_from_wavelength_nm(center_wavelength_nm: float) -> float:
    wavelength_um = float(center_wavelength_nm) * 1e-3
    return float(2.0 * np.pi * _C_UM_PER_FS / wavelength_um)


def build_standalone_laser_state(spec: StandaloneLaserSpec) -> LaserState:
    pulse_spec = spec.pulse
    t = np.linspace(
        -0.5 * pulse_spec.time_window_fs,
        0.5 * pulse_spec.time_window_fs,
        pulse_spec.n_samples,
        dtype=np.float64,
    )
    dt = float(t[1] - t[0])

    width_fs = float(pulse_spec.width_fs)
    peak_power_w = _resolve_peak_power_w(spec)

    if pulse_spec.shape == "gaussian":
        intensity = peak_power_w * np.exp(-4.0 * np.log(2.0) * (t / width_fs) ** 2)
    else:
        t0_fs = width_fs / _SECH_FWHM_FACTOR
        intensity = peak_power_w / np.cosh(t / t0_fs) ** 2

    field_t = np.sqrt(intensity).astype(np.complex128)
    field_w = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(field_t)))
    w = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(t.size, d=dt))
    dw = float(w[1] - w[0])

    return LaserState(
        pulse=PulseState(
            grid=PulseGrid(
                t=t.tolist(),
                w=w.tolist(),
                dt=dt,
                dw=dw,
                center_wavelength_nm=float(pulse_spec.center_wavelength_nm),
            ),
            field_t=field_t,
            field_w=field_w,
            intensity_t=np.abs(field_t) ** 2,
            spectrum_w=np.abs(field_w) ** 2,
        ),
        beam=BeamState(
            radius_mm=float(spec.beam.radius_mm),
            m2=float(spec.beam.m2),
        ),
        meta={
            "laser.width_fs": float(width_fs),
            "laser.peak_power_w": float(peak_power_w),
            "laser.center_wavelength_nm": float(pulse_spec.center_wavelength_nm),
        },
        metrics={
            "laser.energy_au": float(np.sum(np.abs(field_t) ** 2) * dt),
            "laser.peak_power_w": float(peak_power_w),
        },
    )


def apply_phase_to_state(state: LaserState, phase_rad: np.ndarray) -> LaserState:
    out = state.deepcopy()
    phase = np.asarray(phase_rad, dtype=np.float64).reshape(-1)
    if phase.shape != np.asarray(out.pulse.field_w).shape:
        raise ValueError(
            "phase_rad must match pulse.field_w shape: "
            f"expected {np.asarray(out.pulse.field_w).shape}, got {phase.shape}"
        )

    out.pulse.field_w = out.pulse.field_w * np.exp(1j * phase)
    out.pulse.field_t = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(out.pulse.field_w)))
    out.pulse.intensity_t = np.abs(out.pulse.field_t) ** 2
    out.pulse.spectrum_w = np.abs(out.pulse.field_w) ** 2
    return out


def update_beam_state_from_abcd(
    state: LaserState,
    matrix_abcd: np.ndarray,
) -> LaserState:
    out = state.deepcopy()
    wavelength_um = float(out.pulse.grid.center_wavelength_nm) * 1e-3
    waist_radius_um = float(out.beam.radius_mm) * 1e3
    effective_wavelength_um = wavelength_um * float(out.beam.m2)
    q_in = q_from_waist(waist_radius_um, effective_wavelength_um)
    q_out = q_propagate(q_in, np.asarray(matrix_abcd, dtype=np.float64))
    radius_out_um = beam_radius_from_q(q_out, effective_wavelength_um)
    out.beam = BeamState(radius_mm=float(radius_out_um / 1e3), m2=float(out.beam.m2))
    return out


def _resolve_peak_power_w(spec: StandaloneLaserSpec) -> float:
    pulse = spec.pulse
    if pulse.peak_power_w is not None:
        return float(pulse.peak_power_w)

    width_fs = float(pulse.width_fs)
    if pulse.shape == "gaussian":
        duration_factor_fs = width_fs * math.sqrt(math.pi / (4.0 * math.log(2.0)))
    else:
        duration_factor_fs = 2.0 * (width_fs / _SECH_FWHM_FACTOR)

    if pulse.pulse_energy_j is not None:
        return float(pulse.pulse_energy_j) / (duration_factor_fs * 1e-15)
    if pulse.avg_power_w is not None:
        rep_rate_hz = float(pulse.rep_rate_mhz) * 1e6
        pulse_energy_j = float(pulse.avg_power_w) / rep_rate_hz
        return pulse_energy_j / (duration_factor_fs * 1e-15)
    return 1.0
