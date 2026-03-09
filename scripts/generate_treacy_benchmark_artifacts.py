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
    run_treacy_mirror_heatmap,
    run_treacy_radius_convergence,
)


def generate_artifacts(output_dir: Path) -> tuple[Path, Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    radius_points = run_treacy_radius_convergence()
    mirror_points = run_treacy_mirror_heatmap()

    radius_json = output_dir / "treacy_radius_convergence.json"
    radius_png = output_dir / "treacy_radius_convergence.png"
    heatmap_json = output_dir / "treacy_radius_mirror_heatmap.json"
    heatmap_png = output_dir / "treacy_radius_mirror_heatmap.png"

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

    _plot_radius_convergence(radius_points, radius_png)
    _plot_mirror_heatmap(mirror_points, heatmap_png)
    return radius_json, radius_png, heatmap_json, heatmap_png


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
