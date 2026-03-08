from __future__ import annotations

from collections.abc import Callable, Sequence
from time import perf_counter
from typing import Any

import numpy as np
import numpy.typing as npt

from abcdef_sim.data_models.states import RayState
from abcdef_sim.optics.thick_lens import ThickLens
from abcdef_sim.physics.abcd.gaussian import q_from_waist
from abcdef_sim.physics.abcd.lenses import (
    DoubletAssembly,
    SellmeierMaterial,
    ThickLensSpec,
    sample_doublet_beam_radius_profile,
    thick_lens_matrix_for_spec,
)
from abcdef_sim.physics.abcd.raytracing_ref import (
    from_raytracing_matrix,
    propagate_gaussian_beam_raytracing,
    raytracing_compose,
    raytracing_space,
    raytracing_thick_lens,
    sample_gaussian_beam_radii_raytracing,
)
from abcdef_sim.physics.abcdef.propagation import propagate_step
from abcdef_sim.physics.validation import BenchmarkComparison, ObservableComparison

NDArrayF = npt.NDArray[np.float64]

_C_UM_PER_FS = 0.299792458
_DEFAULT_WAVELENGTH_START_UM = 0.810
_DEFAULT_WAVELENGTH_STOP_UM = 1.550

__all__ = [
    "abcd_helper_doublet_beam_profile_comparisons",
    "abcd_helper_single_lens_effective_focal_length_comparison",
    "abcdef_runtime_doublet_effective_focal_length_comparison",
    "abcdef_runtime_doublet_ray_output_comparisons",
    "abcdef_runtime_single_lens_effective_focal_length_comparison",
    "abcdef_runtime_single_lens_ray_output_comparisons",
    "doublet_beam_profile_comparisons",
    "doublet_runtime_optics",
    "format_abcdef_runtime_benchmark_table",
    "format_default_benchmark_table",
    "raytracing_doublet_elements_mm",
    "reference_doublet",
    "run_abcdef_runtime_wavelength_tracking_benchmarks",
    "run_wavelength_tracking_benchmarks",
    "single_lens_batched_ray_output_comparisons",
    "single_lens_effective_focal_length_comparison",
    "single_lens_runtime_optic",
    "single_lens_wavelength_grid_um",
    "wavelength_um_to_omega_rad_per_fs",
]


def wavelength_um_to_omega_rad_per_fs(wavelength_um: npt.ArrayLike) -> NDArrayF:
    """Convert wavelength samples in micrometers to angular frequency."""

    wavelength_arr = np.asarray(wavelength_um, dtype=float).reshape(-1)
    if np.any(wavelength_arr <= 0.0):
        raise ValueError("wavelength_um must be > 0")
    return (2.0 * np.pi * _C_UM_PER_FS) / wavelength_arr


def single_lens_wavelength_grid_um(num_samples: int) -> NDArrayF:
    """Default chromatic sweep used by validation plots and benchmarks."""

    count = int(num_samples)
    if count < 1:
        raise ValueError("num_samples must be >= 1")
    return np.linspace(
        _DEFAULT_WAVELENGTH_START_UM,
        _DEFAULT_WAVELENGTH_STOP_UM,
        count,
        dtype=float,
    )


def abcd_helper_single_lens_spec() -> ThickLensSpec:
    """Canonical single-lens helper spec for lower-level ABCD validation."""

    return ThickLensSpec(
        refractive_index=_n_lak22_um(),
        R1=85.0,
        R2=-55.0,
        thickness=6.0,
        n_in=1.0,
        n_out=1.0,
    )


def abcd_helper_single_lens_effective_focal_length_comparison(
    wavelength_um: npt.ArrayLike,
) -> ObservableComparison:
    """Compare an ABCD helper lens against raytracing over a wavelength sweep."""

    wavelength_arr = np.asarray(wavelength_um, dtype=float).reshape(-1)
    spec = abcd_helper_single_lens_spec()
    local = np.array(
        [
            _effective_focal_length_from_matrix(
                thick_lens_matrix_for_spec(spec, wavelength=float(wavelength))
            )
            for wavelength in wavelength_arr
        ],
        dtype=float,
    )
    reference = np.array(
        [
            _effective_focal_length_from_matrix(
                from_raytracing_matrix(
                    raytracing_thick_lens(
                        n=_resolve_material_index(spec.refractive_index, float(wavelength)),
                        R1=spec.R1,
                        R2=spec.R2,
                        thickness=spec.thickness,
                        n_in=spec.n_in,
                        n_out=spec.n_out,
                    )
                )
            )
            for wavelength in wavelength_arr
        ],
        dtype=float,
    )

    return ObservableComparison(
        name="ABCD helper single thick lens effective focal length",
        observable_label="effective focal length (mm)",
        coordinates=wavelength_arr,
        coordinate_label="wavelength (um)",
        local=local,
        reference=reference,
    )


def reference_doublet() -> DoubletAssembly:
    """Reference achromatic doublet used by lower-level ABCD helper validation."""

    return DoubletAssembly(
        first=ThickLensSpec(
            refractive_index=_n_lak22_mm(),
            R1=112.88,
            R2=-112.88,
            thickness=6.0,
            n_in=1.0,
            n_out=1.0,
        ),
        second=ThickLensSpec(
            refractive_index=_n_sf6_mm(),
            R1=-112.88,
            R2=-1415.62,
            thickness=4.0,
            n_in=1.0,
            n_out=1.0,
        ),
        gap=0.0,
    )


def abcd_helper_doublet_beam_profile_comparisons(
    *,
    wavelengths_mm: Sequence[float] = (0.000810, 0.001550),
    z_samples: npt.ArrayLike | None = None,
) -> list[ObservableComparison]:
    """Compare helper-level Gaussian beam profiles at representative wavelengths."""

    spec = reference_doublet()
    z_arr = (
        np.linspace(0.0, 250.0, 161, dtype=float)
        if z_samples is None
        else np.asarray(z_samples, dtype=float).reshape(-1)
    )
    comparisons: list[ObservableComparison] = []
    for wavelength in wavelengths_mm:
        q_in = q_from_waist(waist_radius=1.024, wavelength=wavelength, distance_from_waist=0.0)
        local = sample_doublet_beam_radius_profile(q_in, spec, wavelength, z_arr)
        oracle_beam = propagate_gaussian_beam_raytracing(
            q_in,
            wavelength,
            raytracing_doublet_elements_mm(spec, wavelength),
        )
        reference = sample_gaussian_beam_radii_raytracing(oracle_beam, z_arr)
        comparisons.append(
            ObservableComparison(
                name=f"ABCD helper doublet beam profile ({wavelength * 1e6:.0f} nm)",
                observable_label="beam radius (mm)",
                coordinates=z_arr,
                coordinate_label="propagation distance (mm)",
                local=local,
                reference=reference,
            )
        )
    return comparisons


def single_lens_runtime_optic() -> ThickLens:
    """Return the canonical wavelength-dependent ABCDEF runtime thick-lens optic."""

    return ThickLens(
        name="ThickLens",
        instance_name="validation-single-lens",
        _length=6.0,
        R1=85.0,
        R2=-55.0,
        n_in=1.0,
        n_out=1.0,
        refractive_index_model=_n_lak22_um(),
    )


def doublet_runtime_optics() -> tuple[ThickLens, ThickLens]:
    """Return runtime optics for the ABCDEF validation doublet chain."""

    first = ThickLens(
        name="ThickLens",
        instance_name="validation-doublet-first",
        _length=6.0,
        R1=112.88,
        R2=-112.88,
        n_in=1.0,
        n_out=1.0,
        refractive_index_model=_n_lak22_um(),
    )
    second = ThickLens(
        name="ThickLens",
        instance_name="validation-doublet-second",
        _length=4.0,
        R1=-112.88,
        R2=-1415.62,
        n_in=1.0,
        n_out=1.0,
        refractive_index_model=_n_sf6_um(),
    )
    return first, second


def abcdef_runtime_single_lens_effective_focal_length_comparison(
    wavelength_um: npt.ArrayLike,
) -> ObservableComparison:
    """Compare the ABCDEF runtime thick-lens optic against raytracing."""

    wavelength_arr = np.asarray(wavelength_um, dtype=float).reshape(-1)
    omega = wavelength_um_to_omega_rad_per_fs(wavelength_arr)
    lens = single_lens_runtime_optic()
    local = _effective_focal_length_from_matrix_batch(lens.matrix(omega))
    reference = np.array(
        [
            _effective_focal_length_from_matrix(
                from_raytracing_matrix(
                    raytracing_thick_lens(
                        n=float(n_value),
                        R1=lens.R1,
                        R2=lens.R2,
                        thickness=lens.length,
                        n_in=lens.n_in,
                        n_out=lens.n_out,
                    )
                )
            )
            for n_value in lens.n(omega)
        ],
        dtype=float,
    )

    return ObservableComparison(
        name="ABCDEF runtime single thick lens effective focal length",
        observable_label="effective focal length (mm)",
        coordinates=wavelength_arr,
        coordinate_label="wavelength (um)",
        local=local,
        reference=reference,
    )


def abcdef_runtime_doublet_effective_focal_length_comparison(
    wavelength_um: npt.ArrayLike,
) -> ObservableComparison:
    """Compare the ABCDEF runtime doublet chain against raytracing."""

    wavelength_arr = np.asarray(wavelength_um, dtype=float).reshape(-1)
    omega = wavelength_um_to_omega_rad_per_fs(wavelength_arr)
    first, second = doublet_runtime_optics()
    local = _effective_focal_length_from_matrix_batch(second.matrix(omega) @ first.matrix(omega))
    reference = np.array(
        [
            _effective_focal_length_from_matrix(
                from_raytracing_matrix(
                    raytracing_compose(*_runtime_doublet_raytracing_elements(float(wavelength)))
                )
            )
            for wavelength in wavelength_arr
        ],
        dtype=float,
    )

    return ObservableComparison(
        name="ABCDEF runtime doublet chain effective focal length",
        observable_label="effective focal length (mm)",
        coordinates=wavelength_arr,
        coordinate_label="wavelength (um)",
        local=local,
        reference=reference,
    )


def abcdef_runtime_single_lens_ray_output_comparisons(
    wavelength_um: npt.ArrayLike,
    *,
    ray_height: float = 0.7,
    theta_in: float = 0.015,
) -> tuple[ObservableComparison, ObservableComparison]:
    """Compare ABCDEF runtime single-lens ray outputs against raytracing."""

    wavelength_arr = np.asarray(wavelength_um, dtype=float).reshape(-1)
    omega = wavelength_um_to_omega_rad_per_fs(wavelength_arr)
    lens = single_lens_runtime_optic()
    state_out = propagate_step(
        _input_ray_state(batch_size=omega.size, x=ray_height, theta=theta_in),
        lens.matrix(omega),
    )
    reference_rays = [
        _runtime_single_lens_raytracing_element(float(wavelength))
        * _import_raytracing().Ray(
            y=ray_height,
            theta=theta_in,
        )
        for wavelength in wavelength_arr
    ]
    return _ray_output_comparisons(
        wavelength_arr=wavelength_arr,
        state_out=state_out,
        reference_rays=reference_rays,
        name_prefix="ABCDEF runtime single thick lens",
    )


def abcdef_runtime_doublet_ray_output_comparisons(
    wavelength_um: npt.ArrayLike,
    *,
    ray_height: float = 0.7,
    theta_in: float = 0.015,
) -> tuple[ObservableComparison, ObservableComparison]:
    """Compare ABCDEF runtime doublet ray outputs against raytracing."""

    wavelength_arr = np.asarray(wavelength_um, dtype=float).reshape(-1)
    omega = wavelength_um_to_omega_rad_per_fs(wavelength_arr)
    first, second = doublet_runtime_optics()
    state_mid = propagate_step(
        _input_ray_state(batch_size=omega.size, x=ray_height, theta=theta_in),
        first.matrix(omega),
    )
    state_out = propagate_step(state_mid, second.matrix(omega))
    reference_rays = [
        raytracing_compose(*_runtime_doublet_raytracing_elements(float(wavelength)))
        * _import_raytracing().Ray(y=ray_height, theta=theta_in)
        for wavelength in wavelength_arr
    ]
    return _ray_output_comparisons(
        wavelength_arr=wavelength_arr,
        state_out=state_out,
        reference_rays=reference_rays,
        name_prefix="ABCDEF runtime doublet chain",
    )


def run_abcdef_runtime_wavelength_tracking_benchmarks(
    wavelength_counts: Sequence[int],
    *,
    warmup_runs: int = 2,
    measured_runs: int = 5,
) -> list[BenchmarkComparison]:
    """Benchmark ABCDEF runtime wavelength tracking against raytracing loops."""

    counts = [int(count) for count in wavelength_counts]
    if not counts:
        raise ValueError("wavelength_counts must be non-empty")
    if any(count < 1 for count in counts):
        raise ValueError("wavelength_counts must contain only values >= 1")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be >= 0")
    if measured_runs < 1:
        raise ValueError("measured_runs must be >= 1")

    comparisons: list[BenchmarkComparison] = []
    for count in counts:
        wavelength_um = single_lens_wavelength_grid_um(count)
        omega = wavelength_um_to_omega_rad_per_fs(wavelength_um)

        local_single = _median_elapsed_seconds(
            lambda: _single_lens_runtime_trace(omega),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        reference_single = _median_elapsed_seconds(
            lambda: _single_lens_raytracing_trace(wavelength_um),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        comparisons.append(
            BenchmarkComparison(
                scenario_name="ABCDEF runtime single thick lens ray trace",
                wavelength_count=count,
                local_seconds=local_single,
                reference_seconds=reference_single,
            )
        )

        local_doublet = _median_elapsed_seconds(
            lambda: _doublet_runtime_trace(omega),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        reference_doublet = _median_elapsed_seconds(
            lambda: _doublet_raytracing_trace(wavelength_um),
            warmup_runs=warmup_runs,
            measured_runs=measured_runs,
        )
        comparisons.append(
            BenchmarkComparison(
                scenario_name="ABCDEF runtime doublet chain ray trace",
                wavelength_count=count,
                local_seconds=local_doublet,
                reference_seconds=reference_doublet,
            )
        )
    return comparisons


def format_abcdef_runtime_benchmark_table(
    wavelength_counts: Sequence[int],
    *,
    warmup_runs: int = 2,
    measured_runs: int = 5,
) -> str:
    """Run ABCDEF runtime benchmarks and format them as Markdown."""

    from abcdef_sim.physics.validation import format_benchmark_table

    comparisons = run_abcdef_runtime_wavelength_tracking_benchmarks(
        wavelength_counts,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    return format_benchmark_table(comparisons)


def _single_lens_runtime_trace(omega: NDArrayF) -> NDArrayF:
    lens = single_lens_runtime_optic()
    state_out = propagate_step(
        _input_ray_state(batch_size=omega.size, x=0.7, theta=0.015),
        lens.matrix(omega),
    )
    return np.asarray(state_out.rays[:, :2, 0], dtype=float)


def _single_lens_raytracing_trace(wavelength_um: NDArrayF) -> NDArrayF:
    rt = _import_raytracing()
    outputs = np.empty((wavelength_um.size, 2), dtype=float)
    for idx, wavelength in enumerate(wavelength_um):
        ray_out = _runtime_single_lens_raytracing_element(float(wavelength)) * rt.Ray(
            y=0.7,
            theta=0.015,
        )
        outputs[idx, 0] = float(ray_out.y)
        outputs[idx, 1] = float(ray_out.theta)
    return outputs


def _doublet_runtime_trace(omega: NDArrayF) -> NDArrayF:
    first, second = doublet_runtime_optics()
    state_mid = propagate_step(
        _input_ray_state(batch_size=omega.size, x=0.7, theta=0.015),
        first.matrix(omega),
    )
    state_out = propagate_step(state_mid, second.matrix(omega))
    return np.asarray(state_out.rays[:, :2, 0], dtype=float)


def _doublet_raytracing_trace(wavelength_um: NDArrayF) -> NDArrayF:
    rt = _import_raytracing()
    outputs = np.empty((wavelength_um.size, 2), dtype=float)
    for idx, wavelength in enumerate(wavelength_um):
        ray_out = raytracing_compose(
            *_runtime_doublet_raytracing_elements(float(wavelength))
        ) * rt.Ray(
            y=0.7,
            theta=0.015,
        )
        outputs[idx, 0] = float(ray_out.y)
        outputs[idx, 1] = float(ray_out.theta)
    return outputs


def _ray_output_comparisons(
    *,
    wavelength_arr: NDArrayF,
    state_out: RayState,
    reference_rays: Sequence[object],
    name_prefix: str,
) -> tuple[ObservableComparison, ObservableComparison]:
    x_out = ObservableComparison(
        name=f"{name_prefix} output height",
        observable_label="x_out (mm)",
        coordinates=wavelength_arr,
        coordinate_label="wavelength (um)",
        local=state_out.rays[:, 0, 0],
        reference=np.array([float(getattr(ray, "y")) for ray in reference_rays], dtype=float),
    )
    theta_out = ObservableComparison(
        name=f"{name_prefix} output angle",
        observable_label="theta_out (rad)",
        coordinates=wavelength_arr,
        coordinate_label="wavelength (um)",
        local=state_out.rays[:, 1, 0],
        reference=np.array([float(getattr(ray, "theta")) for ray in reference_rays], dtype=float),
    )
    return x_out, theta_out


def _runtime_single_lens_raytracing_element(wavelength_um: float) -> object:
    lens = single_lens_runtime_optic()
    omega = wavelength_um_to_omega_rad_per_fs([wavelength_um])
    n_lens = float(lens.n(omega)[0])
    return raytracing_thick_lens(
        n=n_lens,
        R1=lens.R1,
        R2=lens.R2,
        thickness=lens.length,
        n_in=lens.n_in,
        n_out=lens.n_out,
    )


def _runtime_doublet_raytracing_elements(wavelength_um: float) -> tuple[object, object]:
    first, second = doublet_runtime_optics()
    omega = wavelength_um_to_omega_rad_per_fs([wavelength_um])
    first_n = float(first.n(omega)[0])
    second_n = float(second.n(omega)[0])
    return (
        raytracing_thick_lens(
            n=first_n,
            R1=first.R1,
            R2=first.R2,
            thickness=first.length,
            n_in=first.n_in,
            n_out=first.n_out,
        ),
        raytracing_thick_lens(
            n=second_n,
            R1=second.R1,
            R2=second.R2,
            thickness=second.length,
            n_in=second.n_in,
            n_out=second.n_out,
        ),
    )


def _median_elapsed_seconds(
    fn: Callable[[], object],
    *,
    warmup_runs: int,
    measured_runs: int,
) -> float:
    for _ in range(warmup_runs):
        fn()

    samples = np.empty(measured_runs, dtype=float)
    for idx in range(measured_runs):
        start = perf_counter()
        fn()
        samples[idx] = perf_counter() - start
    return float(np.median(samples))


def _input_ray_state(*, batch_size: int, x: float, theta: float) -> RayState:
    rays = np.array([x, theta, 1.0], dtype=float)
    return RayState(
        rays=np.repeat(rays[None, :, None], batch_size, axis=0),
        system=np.repeat(np.eye(3, dtype=float)[None, ...], batch_size, axis=0),
        meta={},
    )


def raytracing_doublet_elements_mm(spec: DoubletAssembly, wavelength_mm: float) -> list[object]:
    first_n = _resolve_material_index(spec.first.refractive_index, wavelength_mm)
    second_n = _resolve_material_index(spec.second.refractive_index, wavelength_mm)
    elements = [
        raytracing_thick_lens(
            n=float(first_n),
            R1=spec.first.R1,
            R2=spec.first.R2,
            thickness=spec.first.thickness,
            n_in=spec.first.n_in,
            n_out=spec.first.n_out,
        ),
        raytracing_thick_lens(
            n=float(second_n),
            R1=spec.second.R1,
            R2=spec.second.R2,
            thickness=spec.second.thickness,
            n_in=spec.second.n_in,
            n_out=spec.second.n_out,
        ),
    ]
    if spec.gap > 0.0:
        elements.insert(1, raytracing_space(d=spec.gap))
    return elements


def _effective_focal_length_from_matrix_batch(matrices: NDArrayF) -> NDArrayF:
    matrices_arr = np.asarray(matrices, dtype=float)
    return -1.0 / matrices_arr[:, 1, 0]


def _effective_focal_length_from_matrix(matrix: NDArrayF) -> float:
    return float(-1.0 / np.asarray(matrix, dtype=float)[1, 0])


def _n_lak22_mm() -> SellmeierMaterial:
    return SellmeierMaterial(
        name="N-LAK22-mm",
        b_terms=(1.14229781, 0.535138441, 1.04088385),
        c_terms=(0.00585778594e-6, 0.0198546147e-6, 100.834017e-6),
    )


def _n_sf6_mm() -> SellmeierMaterial:
    return SellmeierMaterial(
        name="N-SF6-mm",
        b_terms=(1.77931763, 0.338149866, 2.08734474),
        c_terms=(0.0133714182e-6, 0.0617533621e-6, 174.01759e-6),
    )


def _n_lak22_um() -> SellmeierMaterial:
    return SellmeierMaterial(
        name="N-LAK22-um",
        b_terms=(1.14229781, 0.535138441, 1.04088385),
        c_terms=(0.00585778594, 0.0198546147, 100.834017),
    )


def _n_sf6_um() -> SellmeierMaterial:
    return SellmeierMaterial(
        name="N-SF6-um",
        b_terms=(1.77931763, 0.338149866, 2.08734474),
        c_terms=(0.0133714182, 0.0617533621, 174.01759),
    )


def _resolve_material_index(
    refractive_index: float | SellmeierMaterial,
    wavelength: float,
) -> float:
    if isinstance(refractive_index, SellmeierMaterial):
        return float(refractive_index.refractive_index(wavelength))
    return float(refractive_index)


def _import_raytracing() -> Any:
    import raytracing as rt

    return rt


# Backward-compatible aliases retained for existing tests/imports.
single_lens_effective_focal_length_comparison = (
    abcdef_runtime_single_lens_effective_focal_length_comparison
)
single_lens_batched_ray_output_comparisons = abcdef_runtime_single_lens_ray_output_comparisons
doublet_beam_profile_comparisons = abcd_helper_doublet_beam_profile_comparisons
run_wavelength_tracking_benchmarks = run_abcdef_runtime_wavelength_tracking_benchmarks
format_default_benchmark_table = format_abcdef_runtime_benchmark_table
