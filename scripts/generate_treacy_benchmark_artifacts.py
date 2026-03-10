#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogFormatterMathtext

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from abcdef_sim.analytics import (  # noqa: E402
    DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM,
    DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM,
    build_output_plane_field_1d,
    run_treacy_mirror_heatmap,
    run_treacy_radius_convergence,
    summarize_output_plane_field,
)
from abcdef_sim import (  # noqa: E402
    BeamSpec,
    PulseSpec,
    StandaloneLaserSpec,
    run_abcdef,
    treacy_compressor_preset,
)

PUBLICATION_SPATIOSPECTRAL_CASE = {
    "label": "canonical",
    "beam_radius_mm": 1.0,
    "length_to_mirror_um": 0.0,
}
SELECTED_SPATIOSPECTRAL_CASES = (
    {"label": "small_radius", "beam_radius_mm": 0.1, "length_to_mirror_um": 0.0},
    PUBLICATION_SPATIOSPECTRAL_CASE,
    {"label": "large_radius", "beam_radius_mm": 10.0, "length_to_mirror_um": 0.0},
    {"label": "mirror_leg", "beam_radius_mm": 1.0, "length_to_mirror_um": 100_000.0},
)


def generate_artifacts(output_dir: Path) -> tuple[Path, ...]:
    output_dir.mkdir(parents=True, exist_ok=True)

    radius_points = run_treacy_radius_convergence()
    mirror_points = run_treacy_mirror_heatmap()
    publication_fields, spatiospectral_payload = _build_selected_spatiospectral_cases()

    radius_json = output_dir / "treacy_radius_convergence.json"
    radius_png = output_dir / "treacy_radius_convergence.png"
    heatmap_json = output_dir / "treacy_radius_mirror_heatmap.json"
    heatmap_png = output_dir / "treacy_radius_mirror_heatmap.png"
    spatial_radius_json = output_dir / "treacy_spatial_metrics_vs_radius.json"
    spatial_radius_png = output_dir / "treacy_spatial_metrics_vs_radius.png"
    spatial_mirror_json = output_dir / "treacy_spatial_metrics_vs_radius_mirror.json"
    spatial_mirror_png = output_dir / "treacy_spatial_metrics_vs_radius_mirror.png"
    spatiospectral_json = output_dir / "treacy_output_plane_spatiospectral.json"
    spatiospectral_png = output_dir / "treacy_output_plane_spatiospectral.png"

    radius_payload = {
        "plot_description": (
            "Full ABCDEF relative-error curves versus the analytic Treacy baseline "
            "at length_to_mirror_um = 0.0."
        ),
        "headline_metric": "relative_error_full_abcdef_vs_analytic",
        "beam_radii_mm": list(DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM),
        "length_to_mirror_um": 0.0,
        "points": [point.to_dict() for point in radius_points],
    }
    radius_json.write_text(json.dumps(radius_payload, indent=2))

    heatmap_payload = {
        "plot_description": (
            "Primary matched comparison heatmap: relative error of the full ABCDEF "
            "Treacy preset versus the analytic plane-wave Treacy GDD/TOD over beam "
            "radius and mirror length."
        ),
        "headline_metric": "relative_error_full_abcdef_vs_analytic",
        "relative_error_definition": "abs(full_abcdef - analytic) / abs(analytic)",
        "analytic_reference": {
            "gdd_fs2": float(radius_points[0].analytic_gdd_fs2),
            "tod_fs3": float(radius_points[0].analytic_tod_fs3),
        },
        "beam_radii_mm": list(DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM),
        "mirror_lengths_um": list(DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM),
        "points": [point.to_dict() for point in mirror_points],
    }
    spatial_radius_payload = {
        "plot_description": (
            "Full ABCDEF scalar-error and spatial-metric companion curves versus beam radius "
            "at length_to_mirror_um = 0.0."
        ),
        "beam_radii_mm": list(DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM),
        "length_to_mirror_um": 0.0,
        "points": [point.to_dict() for point in radius_points],
    }
    spatial_mirror_payload = {
        "plot_description": (
            "Full ABCDEF spatial-metric companion heatmaps versus beam radius and "
            "length_to_mirror."
        ),
        "beam_radii_mm": list(DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM),
        "mirror_lengths_um": list(DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM),
        "points": [point.to_dict() for point in mirror_points],
    }
    heatmap_json.write_text(json.dumps(heatmap_payload, indent=2))
    spatial_radius_json.write_text(json.dumps(spatial_radius_payload, indent=2))
    spatial_mirror_json.write_text(json.dumps(spatial_mirror_payload, indent=2))
    spatiospectral_json.write_text(json.dumps(spatiospectral_payload, indent=2))

    _plot_radius_convergence(radius_points, radius_png)
    _plot_mirror_heatmap(mirror_points, heatmap_png)
    _plot_spatial_metrics_vs_radius(radius_points, spatial_radius_png)
    _plot_spatial_metrics_vs_radius_mirror(mirror_points, spatial_mirror_png)
    _plot_output_plane_spatiospectral(
        full_field=publication_fields["full"],
        output_path=spatiospectral_png,
    )
    return (
        radius_json,
        radius_png,
        heatmap_json,
        heatmap_png,
        spatial_radius_json,
        spatial_radius_png,
        spatial_mirror_json,
        spatial_mirror_png,
        spatiospectral_json,
        spatiospectral_png,
    )


def _plot_radius_convergence(points: tuple[object, ...], output_path: Path) -> None:
    beam_radii = np.array([point.beam_radius_mm for point in points], dtype=np.float64)
    full_gdd_errors = np.array([point.full_gdd_rel_error for point in points], dtype=np.float64)
    full_tod_errors = np.array([point.full_tod_rel_error for point in points], dtype=np.float64)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, full_errors, title, color in [
        (axes[0], full_gdd_errors, "Full ABCDEF GDD relative error", "tab:red"),
        (axes[1], full_tod_errors, "Full ABCDEF TOD relative error", "tab:orange"),
    ]:
        ax.plot(
            beam_radii,
            full_errors,
            marker="o",
            color=color,
            label="Full ABCDEF vs analytic",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Input beam radius (mm)")
        ax.set_ylabel("Relative error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Treacy Full-ABCDEF Relative Error vs Analytic Baseline at length_to_mirror = 0 um")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_spatial_metrics_vs_radius(points: tuple[object, ...], output_path: Path) -> None:
    beam_radii = np.array([point.beam_radius_mm for point in points], dtype=np.float64)
    full_gdd_errors = np.array([point.full_gdd_rel_error for point in points], dtype=np.float64)
    normalized_spatial = np.array(
        [point.normalized_spatial_chirp_rms for point in points],
        dtype=np.float64,
    )
    mode_overlap = np.array([point.mode_overlap_with_center for point in points], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    axes[0].plot(
        beam_radii,
        full_gdd_errors,
        marker="o",
        color="tab:red",
        label="Full ABCDEF vs analytic",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("GDD rel. error")
    axes[0].set_title("Full ABCDEF Scalar Error")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(beam_radii, normalized_spatial, marker="o", color="tab:green")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("weighted x_rms / w")
    axes[1].set_title("Normalized Spatial Chirp")
    axes[1].grid(True, which="both", alpha=0.3)

    axes[2].plot(beam_radii, mode_overlap, marker="o", color="tab:purple")
    axes[2].set_xscale("log")
    axes[2].set_xlabel("Input beam radius (mm)")
    axes[2].set_ylabel("Mean overlap")
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_title("Output Mode Recombination")
    axes[2].grid(True, which="both", alpha=0.3)

    fig.suptitle("Treacy Full-ABCDEF Scalar Error and Spatial Metrics vs Beam Radius")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_mirror_heatmap(points: tuple[object, ...], output_path: Path) -> None:
    beam_radii = np.array(sorted({point.beam_radius_mm for point in points}), dtype=np.float64)
    mirror_lengths_um = np.array(
        sorted({point.length_to_mirror_um for point in points}),
        dtype=np.float64,
    )
    full_gdd_error_grid = np.zeros((mirror_lengths_um.size, beam_radii.size), dtype=np.float64)
    full_tod_error_grid = np.zeros((mirror_lengths_um.size, beam_radii.size), dtype=np.float64)
    for point in points:
        row = int(np.where(mirror_lengths_um == point.length_to_mirror_um)[0][0])
        col = int(np.where(beam_radii == point.beam_radius_mm)[0][0])
        full_gdd_error_grid[row, col] = point.full_gdd_rel_error
        full_tod_error_grid[row, col] = point.full_tod_rel_error

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, grid, title in [
        (axes[0], full_gdd_error_grid, "Full ABCDEF GDD relative error"),
        (axes[1], full_tod_error_grid, "Full ABCDEF TOD relative error"),
    ]:
        vmin, vmax, ticks = _decade_lognorm_bounds(grid)
        grid_display = np.maximum(grid, vmin)
        im = ax.imshow(
            grid_display,
            origin="lower",
            aspect="auto",
            cmap="viridis",
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
        )
        ax.set_xticks(range(beam_radii.size), [f"{value:g}" for value in beam_radii], rotation=45)
        ax.set_yticks(
            range(mirror_lengths_um.size),
            [f"{value / 1e3:g}" for value in mirror_lengths_um],
        )
        ax.set_xlabel("Input beam radius (mm)")
        ax.set_ylabel("length_to_mirror (mm)")
        ax.set_title(title)
        ax.text(
            0.02,
            0.96,
            (
                f"Analytic GDD = {points[0].analytic_gdd_fs2:.3e} fs^2\n"
                f"Analytic TOD = {points[0].analytic_tod_fs3:.3e} fs^3\n"
                "Error = abs(full ABCDEF - analytic) / abs(analytic)"
            ),
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "alpha": 0.35},
        )
        fig.colorbar(
            im,
            ax=ax,
            shrink=0.85,
            label="Relative error",
            ticks=ticks,
            format=LogFormatterMathtext(),
        )

    fig.suptitle("Matched Treacy Relative Error Heatmap: full ABCDEF vs analytic baseline")
    fig.savefig(output_path, dpi=400)
    plt.close(fig)


def _plot_spatial_metrics_vs_radius_mirror(points: tuple[object, ...], output_path: Path) -> None:
    beam_radii = np.array(sorted({point.beam_radius_mm for point in points}), dtype=np.float64)
    mirror_lengths_um = np.array(
        sorted({point.length_to_mirror_um for point in points}),
        dtype=np.float64,
    )
    normalized_spatial_grid = np.zeros((mirror_lengths_um.size, beam_radii.size), dtype=np.float64)
    mode_overlap_grid = np.zeros((mirror_lengths_um.size, beam_radii.size), dtype=np.float64)
    for point in points:
        row = int(np.where(mirror_lengths_um == point.length_to_mirror_um)[0][0])
        col = int(np.where(beam_radii == point.beam_radius_mm)[0][0])
        normalized_spatial_grid[row, col] = point.normalized_spatial_chirp_rms
        mode_overlap_grid[row, col] = point.mode_overlap_with_center

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, grid, title in [
        (axes[0], normalized_spatial_grid, "Normalized spatial chirp"),
        (axes[1], mode_overlap_grid, "Output mode overlap"),
    ]:
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="magma",
        )
        ax.set_xticks(range(beam_radii.size), [f"{value:g}" for value in beam_radii], rotation=45)
        ax.set_yticks(
            range(mirror_lengths_um.size),
            [f"{value / 1e3:g}" for value in mirror_lengths_um],
        )
        ax.set_xlabel("Input beam radius (mm)")
        ax.set_ylabel("length_to_mirror (mm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85)

    fig.suptitle("Treacy Spatial Metrics vs Beam Radius and Mirror Leg")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_output_plane_spatiospectral(
    *,
    full_field: object,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, field, title in [
        (axes[0], full_field, "Full ABCDEF: x-omega"),
        (axes[1], full_field, "Full ABCDEF: x-t"),
    ]:
        x_mm = np.asarray(field.x_um, dtype=np.float64) / 1e3
        if "x-omega" in title:
            image = _normalized_log_intensity(field.intensity_x_omega).T
            extent = [
                float(np.min(x_mm)),
                float(np.max(x_mm)),
                float(np.min(field.delta_omega_rad_per_fs)),
                float(np.max(field.delta_omega_rad_per_fs)),
            ]
            im = ax.imshow(image, origin="lower", aspect="auto", extent=extent, cmap="magma")
            ax.set_ylabel("delta omega (rad/fs)")
        else:
            image = _normalized_log_intensity(field.intensity_x_t).T
            extent = [
                float(np.min(x_mm)),
                float(np.max(x_mm)),
                float(np.min(field.t_fs)),
                float(np.max(field.t_fs)),
            ]
            im = ax.imshow(image, origin="lower", aspect="auto", extent=extent, cmap="magma")
            ax.set_ylabel("time (fs)")
        ax.set_xlabel("x (mm)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, shrink=0.85, label="log10(norm. intensity)")

    fig.suptitle("Treacy Output-Plane Spatio-Spectral Reconstruction from Full ABCDEF")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _normalized_log_intensity(intensity: np.ndarray) -> np.ndarray:
    intensity_arr = np.asarray(intensity, dtype=np.float64)
    max_value = max(float(np.max(intensity_arr)), 1e-30)
    return np.log10(np.maximum(intensity_arr / max_value, 1e-6))


def _decade_lognorm_bounds(grid: np.ndarray) -> tuple[float, float, np.ndarray]:
    grid_arr = np.asarray(grid, dtype=np.float64)
    positive = grid_arr[grid_arr > 0.0]
    if positive.size == 0:
        vmin_exp = -8
        vmax_exp = 1
    else:
        min_exp = int(np.floor(np.log10(float(np.min(positive)))))
        max_exp = int(np.ceil(np.log10(float(np.max(positive)))))
        vmin_exp = min_exp - 1
        vmax_exp = max_exp + 1
        if vmax_exp <= vmin_exp:
            vmax_exp = vmin_exp + 1
    vmin = 10.0**vmin_exp
    vmax = 10.0**vmax_exp
    ticks = np.logspace(vmin_exp, vmax_exp, num=(vmax_exp - vmin_exp) + 1, dtype=np.float64)
    return vmin, vmax, ticks


def _build_selected_spatiospectral_cases() -> tuple[dict[str, object], dict[str, object]]:
    publication_fields: dict[str, object] = {}
    payload_cases: list[dict[str, object]] = []

    for case in SELECTED_SPATIOSPECTRAL_CASES:
        result = _run_treacy_case(
            beam_radius_mm=float(case["beam_radius_mm"]),
            length_to_mirror_um=float(case["length_to_mirror_um"]),
        )
        full_field = build_output_plane_field_1d(result, phase_variant="full")
        if case["label"] == PUBLICATION_SPATIOSPECTRAL_CASE["label"]:
            publication_fields = {
                "full": full_field,
            }
        payload_cases.append(
            {
                "label": str(case["label"]),
                "beam_radius_mm": float(case["beam_radius_mm"]),
                "length_to_mirror_um": float(case["length_to_mirror_um"]),
                "x_window_um": [
                    float(np.min(full_field.x_um)),
                    float(np.max(full_field.x_um)),
                ],
                "x_samples": int(full_field.x_um.size),
                "full_summary": summarize_output_plane_field(full_field).to_dict(),
            }
        )

    payload = {
        "publication_case": dict(PUBLICATION_SPATIOSPECTRAL_CASE),
        "selected_cases": payload_cases,
    }
    return publication_fields, payload


def _run_treacy_case(*, beam_radius_mm: float, length_to_mirror_um: float) -> object:
    return run_abcdef(
        treacy_compressor_preset(length_to_mirror_um=length_to_mirror_um),
        StandaloneLaserSpec(
            pulse=PulseSpec(
                width_fs=100.0,
                center_wavelength_nm=1030.0,
                n_samples=256,
                time_window_fs=3000.0,
            ),
            beam=BeamSpec(radius_mm=beam_radius_mm, m2=1.0),
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "physics",
    )
    args = parser.parse_args()
    outputs = generate_artifacts(args.output_dir.resolve())
    for path in outputs:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
