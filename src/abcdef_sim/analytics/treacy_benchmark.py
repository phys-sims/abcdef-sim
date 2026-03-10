from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from abcdef_sim import (
    BeamSpec,
    PulseSpec,
    StandaloneLaserSpec,
    run_abcdef,
    treacy_compressor_preset,
)
from abcdef_sim.analytics.spatiospectral import (
    build_output_plane_field_1d,
    summarize_output_plane_field,
    summarize_output_plane_geometry,
)
from abcdef_sim.physics.abcdef.treacy import compute_treacy_analytic_metrics

DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM: tuple[float, ...] = (
    0.1,
    0.2,
    0.5,
    1.0,
    2.0,
    5.0,
    10.0,
)
DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM: tuple[float, ...] = (
    0.0,
    50_000.0,
    100_000.0,
    200_000.0,
    400_000.0,
    800_000.0,
    1_600_000.0,
)

__all__ = [
    "DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM",
    "DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM",
    "TreacyBenchmarkPoint",
    "run_treacy_benchmark_point",
    "run_treacy_mirror_heatmap",
    "run_treacy_radius_convergence",
]


@dataclass(frozen=True, slots=True)
class TreacyBenchmarkPoint:
    beam_radius_mm: float
    length_to_mirror_um: float
    analytic_gdd_fs2: float
    raw_abcdef_gdd_fs2: float
    raw_abcdef_gdd_rel_error: float
    analytic_tod_fs3: float
    raw_abcdef_tod_fs3: float
    raw_abcdef_tod_rel_error: float
    x_centroid_span_um: float
    x_centroid_slope_um_per_rad_per_fs: float
    x_prime_span: float
    x_prime_slope_per_rad_per_fs: float
    weighted_x_centroid_rms_um: float
    weighted_x_prime_rms: float
    weighted_mean_w_out_um: float
    weighted_mean_diffraction_angle_rad: float
    normalized_spatial_chirp_rms: float
    normalized_angular_dispersion_rms: float
    mode_overlap_with_center: float
    pulse_front_tilt_fs_per_um: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def run_treacy_benchmark_point(
    *,
    beam_radius_mm: float,
    length_to_mirror_um: float,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    center_wavelength_nm: float = 1030.0,
    diffraction_order: int = -1,
    n_passes: Literal[2] = 2,
    pulse_width_fs: float = 100.0,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> TreacyBenchmarkPoint:
    result = run_abcdef(
        treacy_compressor_preset(
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            separation_um=separation_um,
            length_to_mirror_um=length_to_mirror_um,
            diffraction_order=diffraction_order,
            n_passes=n_passes,
        ),
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=pulse_width_fs,
                center_wavelength_nm=center_wavelength_nm,
                n_samples=n_samples,
                time_window_fs=time_window_fs,
            ),
            beam=BeamSpec(radius_mm=beam_radius_mm, m2=1.0),
        ),
    )
    analytic = compute_treacy_analytic_metrics(
        line_density_lpmm=line_density_lpmm,
        incidence_angle_deg=incidence_angle_deg,
        separation_um=separation_um,
        wavelength_nm=center_wavelength_nm,
        diffraction_order=diffraction_order,
        n_passes=n_passes,
    )
    raw_abcdef_gdd_fs2 = float(result.final_state.metrics["abcdef.gdd_fs2"])
    raw_abcdef_tod_fs3 = float(result.final_state.metrics["abcdef.tod_fs3"])
    spatial_metrics = summarize_output_plane_geometry(result)
    field_summary = summarize_output_plane_field(build_output_plane_field_1d(result))
    return TreacyBenchmarkPoint(
        beam_radius_mm=float(beam_radius_mm),
        length_to_mirror_um=float(length_to_mirror_um),
        analytic_gdd_fs2=float(analytic.gdd_fs2),
        raw_abcdef_gdd_fs2=raw_abcdef_gdd_fs2,
        raw_abcdef_gdd_rel_error=_relative_error(raw_abcdef_gdd_fs2, analytic.gdd_fs2),
        analytic_tod_fs3=float(analytic.tod_fs3),
        raw_abcdef_tod_fs3=raw_abcdef_tod_fs3,
        raw_abcdef_tod_rel_error=_relative_error(raw_abcdef_tod_fs3, analytic.tod_fs3),
        x_centroid_span_um=spatial_metrics.x_centroid_span_um,
        x_centroid_slope_um_per_rad_per_fs=spatial_metrics.x_centroid_slope_um_per_rad_per_fs,
        x_prime_span=spatial_metrics.x_prime_span,
        x_prime_slope_per_rad_per_fs=spatial_metrics.x_prime_slope_per_rad_per_fs,
        weighted_x_centroid_rms_um=spatial_metrics.weighted_x_centroid_rms_um,
        weighted_x_prime_rms=spatial_metrics.weighted_x_prime_rms,
        weighted_mean_w_out_um=spatial_metrics.weighted_mean_w_out_um,
        weighted_mean_diffraction_angle_rad=spatial_metrics.weighted_mean_diffraction_angle_rad,
        normalized_spatial_chirp_rms=spatial_metrics.normalized_spatial_chirp_rms,
        normalized_angular_dispersion_rms=spatial_metrics.normalized_angular_dispersion_rms,
        mode_overlap_with_center=field_summary.mean_mode_overlap_with_center,
        pulse_front_tilt_fs_per_um=field_summary.pulse_front_tilt_fs_per_um,
    )


def run_treacy_radius_convergence(
    *,
    beam_radii_mm: tuple[float, ...] = DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM,
    length_to_mirror_um: float = 0.0,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    center_wavelength_nm: float = 1030.0,
    diffraction_order: int = -1,
    pulse_width_fs: float = 100.0,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyBenchmarkPoint, ...]:
    return tuple(
        run_treacy_benchmark_point(
            beam_radius_mm=beam_radius_mm,
            length_to_mirror_um=length_to_mirror_um,
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            separation_um=separation_um,
            center_wavelength_nm=center_wavelength_nm,
            diffraction_order=diffraction_order,
            pulse_width_fs=pulse_width_fs,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        )
        for beam_radius_mm in beam_radii_mm
    )


def run_treacy_mirror_heatmap(
    *,
    beam_radii_mm: tuple[float, ...] = DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM,
    mirror_lengths_um: tuple[float, ...] = DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM,
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    center_wavelength_nm: float = 1030.0,
    diffraction_order: int = -1,
    pulse_width_fs: float = 100.0,
    n_samples: int = 256,
    time_window_fs: float = 3000.0,
) -> tuple[TreacyBenchmarkPoint, ...]:
    return tuple(
        run_treacy_benchmark_point(
            beam_radius_mm=beam_radius_mm,
            length_to_mirror_um=length_to_mirror_um,
            line_density_lpmm=line_density_lpmm,
            incidence_angle_deg=incidence_angle_deg,
            separation_um=separation_um,
            center_wavelength_nm=center_wavelength_nm,
            diffraction_order=diffraction_order,
            pulse_width_fs=pulse_width_fs,
            n_samples=n_samples,
            time_window_fs=time_window_fs,
        )
        for length_to_mirror_um in mirror_lengths_um
        for beam_radius_mm in beam_radii_mm
    )


def _relative_error(value: float, reference: float) -> float:
    if reference == 0.0:
        return abs(float(value))
    return abs(float(value) - float(reference)) / abs(float(reference))
