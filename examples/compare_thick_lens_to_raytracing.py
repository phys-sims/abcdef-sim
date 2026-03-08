#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from abcdef_sim.physics.abcd.raytracing_validation import (
    doublet_beam_profile_comparisons,
    run_wavelength_tracking_benchmarks,
    single_lens_effective_focal_length_comparison,
    single_lens_wavelength_grid_um,
)
from abcdef_sim.physics.validation import format_benchmark_table


def _parse_wavelength_counts(raw: str) -> list[int]:
    counts = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("wavelength counts must be a comma-separated list")
    if any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("wavelength counts must all be >= 1")
    return counts


def write_similarity_plot(output_path: Path) -> Path:
    focal = single_lens_effective_focal_length_comparison(single_lens_wavelength_grid_um(151))
    beam_profiles = doublet_beam_profile_comparisons()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(focal.coordinates, focal.local, color="tab:blue", label="abcdef-sim")
    axes[0, 0].plot(
        focal.coordinates,
        focal.reference,
        color="black",
        linestyle="--",
        label="raytracing",
    )
    axes[0, 0].set_title(focal.name)
    axes[0, 0].set_xlabel("Wavelength (um)")
    axes[0, 0].set_ylabel(focal.observable_label)
    axes[0, 0].legend()

    axes[1, 0].plot(focal.coordinates, focal.residual(), color="tab:red")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("Wavelength (um)")
    axes[1, 0].set_ylabel("Residual (mm)")

    colors = ["tab:green", "tab:orange"]
    for comparison, color in zip(beam_profiles, colors, strict=True):
        axes[0, 1].plot(
            comparison.coordinates,
            comparison.local,
            color=color,
            label=f"{comparison.name} abcdef-sim",
        )
        axes[0, 1].plot(
            comparison.coordinates,
            comparison.reference,
            color=color,
            linestyle="--",
            label=f"{comparison.name} raytracing",
        )
        axes[1, 1].plot(
            comparison.coordinates,
            comparison.residual(),
            color=color,
            label=comparison.name,
        )

    axes[0, 1].set_title("Doublet Gaussian beam radius")
    axes[0, 1].set_xlabel("Propagation distance (mm)")
    axes[0, 1].set_ylabel("Beam radius (mm)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("Propagation distance (mm)")
    axes[1, 1].set_ylabel("Residual (mm)")
    axes[1, 1].legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_benchmark_report(
    output_path: Path,
    *,
    wavelength_counts: list[int],
    warmup_runs: int,
    measured_runs: int,
) -> Path:
    comparisons = run_wavelength_tracking_benchmarks(
        wavelength_counts,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    content = "\n".join(
        [
            "# Wavelength tracking benchmark",
            "",
            f"- Warmup runs per case: {warmup_runs}",
            f"- Measured runs per case: {measured_runs}",
            "- Timings are machine-dependent and are documented here only as a sample run.",
            "",
            format_benchmark_table(comparisons),
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "physics",
    )
    parser.add_argument(
        "--wavelength-counts",
        type=_parse_wavelength_counts,
        default=[1, 8, 32, 128, 512],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    plot_path = write_similarity_plot(output_dir / "thick_lens_similarity.png")
    benchmark_path = write_benchmark_report(
        output_dir / "wavelength_tracking_benchmarks.md",
        wavelength_counts=args.wavelength_counts,
        warmup_runs=args.warmup,
        measured_runs=args.repeats,
    )
    print(f"Wrote {plot_path}")
    print(f"Wrote {benchmark_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
