from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from abcdef_sim import BeamSpec, PulseSpec, StandaloneLaserSpec, run_abcdef, treacy_compressor_preset
from abcdef_sim.physics.abcdef.treacy import compute_treacy_analytic_metrics

DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM: tuple[float, ...] = (0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM: tuple[float, ...] = (
    0.0,
    25_000.0,
    50_000.0,
    75_000.0,
    100_000.0,
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
    abcdef_gdd_fs2: float
    analytic_gdd_fs2: float
    gdd_rel_error: float
    abcdef_tod_fs3: float
    analytic_tod_fs3: float
    tod_rel_error: float

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
    abcdef_gdd_fs2 = float(result.final_state.metrics["abcdef.gdd_fs2"])
    abcdef_tod_fs3 = float(result.final_state.metrics["abcdef.tod_fs3"])
    return TreacyBenchmarkPoint(
        beam_radius_mm=float(beam_radius_mm),
        length_to_mirror_um=float(length_to_mirror_um),
        abcdef_gdd_fs2=abcdef_gdd_fs2,
        analytic_gdd_fs2=float(analytic.gdd_fs2),
        gdd_rel_error=_relative_error(abcdef_gdd_fs2, analytic.gdd_fs2),
        abcdef_tod_fs3=abcdef_tod_fs3,
        analytic_tod_fs3=float(analytic.tod_fs3),
        tod_rel_error=_relative_error(abcdef_tod_fs3, analytic.tod_fs3),
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
