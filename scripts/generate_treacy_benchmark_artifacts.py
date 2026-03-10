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
        "beam_radii_mm": list(DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM),
        "length_to_mirror_um": 0.0,
        "points": [point.to_dict() for point in radius_points],
    }
    radius_json.write_text(json.dumps(radius_payload, indent=2))

    heatmap_payload = {
        "beam_radii_mm": list(DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM),
        "mirror_lengths_um": list(DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM),
        "points": [point.to_dict() for point in mirror_points],
    }
    heatmap_json.write_text(json.dumps(heatmap_payload, indent=2))
    spatial_radius_json.write_text(json.dumps(radius_payload, indent=2))
    spatial_mirror_json.write_text(json.dumps(heatmap_payload, indent=2))
    spatiospectral_json.write_text(json.dumps(spatiospectral_payload, indent=2))

    _plot_radius_convergence(radius_points, radius_png)
    _plot_mirror_heatmap(mirror_points, heatmap_png)
    _plot_spatial_metrics_vs_radius(radius_points, spatial_radius_png)
    _plot_spatial_metrics_vs_radius_mirror(mirror_points, spatial_mirror_png)
    _plot_output_plane_spatiospectral(
        full_field=publication_fields["full"],
        without_phi2_field=publication_fields["without_phi2"],
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
    without_phi2_gdd_errors = np.array(
        [point.without_phi2_gdd_rel_error for point in points],
        dtype=np.float64,
    )
    full_tod_errors = np.array([point.full_tod_rel_error for point in points], dtype=np.float64)
    without_phi2_tod_errors = np.array(
        [point.without_phi2_tod_rel_error for point in points],
        dtype=np.float64,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    for ax, full_errors, without_phi2_errors, title in [
        (axes[0], full_gdd_errors, without_phi2_gdd_errors, "GDD relative error"),
        (axes[1], full_tod_errors, without_phi2_tod_errors, "TOD relative error"),
    ]:
        ax.plot(
            beam_radii,
            full_errors,
            marker="o",
            color="tab:red",
            label="Full ABCDEF",
        )
        ax.plot(
            beam_radii,
            without_phi2_errors,
            marker="s",
            color="tab:blue",
            label="ABCDEF without phi2",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Input beam radius (mm)")
        ax.set_ylabel("Relative error")
        ax.set_title(title)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Treacy Error vs Analytic Baseline at length_to_mirror = 0 um")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_spatial_metrics_vs_radius(points: tuple[object, ...], output_path: Path) -> None:
    beam_radii = np.array([point.beam_radius_mm for point in points], dtype=np.float64)
    full_gdd_errors = np.array([point.full_gdd_rel_error for point in points], dtype=np.float64)
    without_phi2_gdd_errors = np.array(
        [point.without_phi2_gdd_rel_error for point in points],
        dtype=np.float64,
    )
    normalized_spatial = np.array(
        [point.normalized_spatial_chirp_rms for point in points],
        dtype=np.float64,
    )
    mode_overlap = np.array([point.mode_overlap_with_center for point in points], dtype=np.float64)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True, constrained_layout=True)
    axes[0].plot(beam_radii, full_gdd_errors, marker="o", color="tab:red", label="Full ABCDEF")
    axes[0].plot(
        beam_radii,
        without_phi2_gdd_errors,
        marker="s",
        color="tab:blue",
        label="ABCDEF without phi2",
    )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("GDD rel. error")
    axes[0].set_title("Scalar Error")
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

    fig.suptitle("Treacy Scalar Error and Normalized Spatial Metrics vs Beam Radius")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_mirror_heatmap(points: tuple[object, ...], output_path: Path) -> None:
    beam_radii = np.array(sorted({point.beam_radius_mm for point in points}), dtype=np.float64)
    mirror_lengths_um = np.array(
        sorted({point.length_to_mirror_um for point in points}),
        dtype=np.float64,
    )
    full_gdd_grid = np.zeros((mirror_lengths_um.size, beam_radii.size), dtype=np.float64)
    without_phi2_gdd_grid = np.zeros((mirror_lengths_um.size, beam_radii.size), dtype=np.float64)
    full_tod_grid = np.zeros((mirror_lengths_um.size, beam_radii.size), dtype=np.float64)
    without_phi2_tod_grid = np.zeros((mirror_lengths_um.size, beam_radii.size), dtype=np.float64)
    for point in points:
        row = int(np.where(mirror_lengths_um == point.length_to_mirror_um)[0][0])
        col = int(np.where(beam_radii == point.beam_radius_mm)[0][0])
        full_gdd_grid[row, col] = point.full_gdd_rel_error
        without_phi2_gdd_grid[row, col] = point.without_phi2_gdd_rel_error
        full_tod_grid[row, col] = point.full_tod_rel_error
        without_phi2_tod_grid[row, col] = point.without_phi2_tod_rel_error

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, grid, title in [
        (axes[0, 0], full_gdd_grid, "Full ABCDEF GDD relative error"),
        (axes[0, 1], full_tod_grid, "Full ABCDEF TOD relative error"),
        (axes[1, 0], without_phi2_gdd_grid, "ABCDEF without phi2 GDD relative error"),
        (axes[1, 1], without_phi2_tod_grid, "ABCDEF without phi2 TOD relative error"),
    ]:
        im = ax.imshow(
            grid,
            origin="lower",
            aspect="auto",
            cmap="viridis",
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

    fig.suptitle("Treacy Error Surface vs Beam Radius and Mirror Leg")
    fig.savefig(output_path, dpi=200)
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

    fig.suptitle("Treacy Normalized Spatial Metrics vs Beam Radius and Mirror Leg")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_output_plane_spatiospectral(
    *,
    full_field: object,
    without_phi2_field: object,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for ax, field, title in [
        (axes[0, 0], full_field, "Full ABCDEF: x-omega"),
        (axes[0, 1], full_field, "Full ABCDEF: x-t"),
        (axes[1, 0], without_phi2_field, "ABCDEF without phi2: x-omega"),
        (axes[1, 1], without_phi2_field, "ABCDEF without phi2: x-t"),
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

    fig.suptitle("Treacy Output-Plane Spatio-Spectral Reconstruction")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _normalized_log_intensity(intensity: np.ndarray) -> np.ndarray:
    intensity_arr = np.asarray(intensity, dtype=np.float64)
    max_value = max(float(np.max(intensity_arr)), 1e-30)
    return np.log10(np.maximum(intensity_arr / max_value, 1e-6))


def _build_selected_spatiospectral_cases() -> tuple[dict[str, object], dict[str, object]]:
    publication_fields: dict[str, object] = {}
    payload_cases: list[dict[str, object]] = []

    for case in SELECTED_SPATIOSPECTRAL_CASES:
        result = _run_treacy_case(
            beam_radius_mm=float(case["beam_radius_mm"]),
            length_to_mirror_um=float(case["length_to_mirror_um"]),
        )
        full_field = build_output_plane_field_1d(result, phase_variant="full")
        without_phi2_field = build_output_plane_field_1d(result, phase_variant="without_phi2")
        if case["label"] == PUBLICATION_SPATIOSPECTRAL_CASE["label"]:
            publication_fields = {
                "full": full_field,
                "without_phi2": without_phi2_field,
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
                "without_phi2_summary": summarize_output_plane_field(without_phi2_field).to_dict(),
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
