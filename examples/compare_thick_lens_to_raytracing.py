#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _ensure_src_on_path() -> None:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


def _parse_wavelength_counts(raw: str) -> list[int]:
    counts = [int(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not counts:
        raise argparse.ArgumentTypeError("wavelength counts must be a comma-separated list")
    if any(count < 1 for count in counts):
        raise argparse.ArgumentTypeError("wavelength counts must all be >= 1")
    return counts


def _plot_overlay_and_residual(
    axes_top: object,
    axes_bottom: object,
    comparison: object,
    *,
    color: str,
    local_label: str,
) -> None:
    local_style = {
        "linewidth": 2.0,
        "linestyle": "-",
        "marker": "o",
        "markersize": 5.5,
        "markerfacecolor": "white",
        "markeredgewidth": 1.2,
        "markevery": 8,
        "zorder": 2,
    }
    reference_style = {
        "linewidth": 1.8,
        "linestyle": "--",
        "marker": "x",
        "markersize": 6.0,
        "markeredgewidth": 1.4,
        "markevery": 8,
        "zorder": 3,
    }
    residual_style = {
        "linewidth": 1.8,
        "marker": "o",
        "markersize": 4.5,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "markevery": 8,
    }

    axes_top.plot(
        comparison.coordinates,
        comparison.local,
        color=color,
        label=local_label,
        **local_style,
    )
    axes_top.plot(
        comparison.coordinates,
        comparison.reference,
        color="black",
        label="raytracing",
        **reference_style,
    )
    axes_top.set_title(comparison.name)
    axes_top.set_xlabel(comparison.coordinate_label)
    axes_top.set_ylabel(comparison.observable_label)
    axes_top.legend()

    axes_bottom.plot(
        comparison.coordinates,
        comparison.residual(),
        color=color,
        **residual_style,
    )
    axes_bottom.axhline(0.0, color="black", linewidth=0.8)
    axes_bottom.set_xlabel(comparison.coordinate_label)
    axes_bottom.set_ylabel(
        f"Residual ({local_label} - raytracing) ({_units_suffix(comparison.observable_label)})"
    )
    axes_bottom.text(
        0.02,
        0.96,
        f"Residual = {local_label} - raytracing",
        transform=axes_bottom.transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )


def _units_suffix(observable_label: str) -> str:
    if "(" in observable_label and observable_label.endswith(")"):
        return observable_label.rsplit("(", maxsplit=1)[1][:-1]
    return observable_label


def write_abcdef_runtime_similarity_plot(output_path: Path) -> Path:
    _ensure_src_on_path()
    from abcdef_sim.physics.abcd.raytracing_validation import (
        abcdef_runtime_doublet_effective_focal_length_comparison,
        abcdef_runtime_single_lens_effective_focal_length_comparison,
        single_lens_wavelength_grid_um,
    )

    wavelength_um = single_lens_wavelength_grid_um(151)
    single_lens = abcdef_runtime_single_lens_effective_focal_length_comparison(wavelength_um)
    doublet = abcdef_runtime_doublet_effective_focal_length_comparison(wavelength_um)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_overlay_and_residual(
        axes[0, 0],
        axes[1, 0],
        single_lens,
        color="tab:blue",
        local_label="abcdef-sim",
    )
    _plot_overlay_and_residual(
        axes[0, 1],
        axes[1, 1],
        doublet,
        color="tab:orange",
        local_label="abcdef-sim",
    )

    fig.suptitle("ABCDEF runtime vs raytracing")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def write_abcd_helper_similarity_plot(output_path: Path) -> Path:
    _ensure_src_on_path()
    from abcdef_sim.physics.abcd.raytracing_validation import (
        abcd_helper_doublet_beam_profile_comparisons,
        abcd_helper_single_lens_effective_focal_length_comparison,
        single_lens_wavelength_grid_um,
    )

    focal = abcd_helper_single_lens_effective_focal_length_comparison(
        single_lens_wavelength_grid_um(151)
    )
    beam_profiles = abcd_helper_doublet_beam_profile_comparisons()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    _plot_overlay_and_residual(
        axes[0, 0],
        axes[1, 0],
        focal,
        color="tab:green",
        local_label="ABCD helper",
    )

    local_style = {
        "linewidth": 2.0,
        "linestyle": "-",
        "marker": "o",
        "markersize": 5.0,
        "markerfacecolor": "white",
        "markeredgewidth": 1.2,
        "markevery": 12,
        "zorder": 2,
    }
    reference_style = {
        "linewidth": 1.8,
        "linestyle": "--",
        "marker": "x",
        "markersize": 5.5,
        "markeredgewidth": 1.3,
        "markevery": 12,
        "zorder": 3,
    }
    residual_style = {
        "linewidth": 1.8,
        "marker": "o",
        "markersize": 4.0,
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "markevery": 12,
    }
    colors = ["tab:purple", "tab:brown"]
    for comparison, color in zip(beam_profiles, colors, strict=True):
        axes[0, 1].plot(
            comparison.coordinates,
            comparison.local,
            color=color,
            label=f"{comparison.name} helper",
            **local_style,
        )
        axes[0, 1].plot(
            comparison.coordinates,
            comparison.reference,
            color=color,
            label=f"{comparison.name} raytracing",
            **reference_style,
        )
        axes[1, 1].plot(
            comparison.coordinates,
            comparison.residual(),
            color=color,
            label=comparison.name,
            **residual_style,
        )

    axes[0, 1].set_title("ABCD helper doublet beam radius")
    axes[0, 1].set_xlabel("propagation distance (mm)")
    axes[0, 1].set_ylabel("beam radius (mm)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("propagation distance (mm)")
    axes[1, 1].set_ylabel("Residual (ABCD helper - raytracing) (mm)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].text(
        0.02,
        0.96,
        "Residual = ABCD helper - raytracing",
        transform=axes[1, 1].transAxes,
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )

    fig.suptitle("ABCD helper validation only (not full ABCDEF runtime)")
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
    _ensure_src_on_path()
    from abcdef_sim.physics.abcd.raytracing_validation import (
        run_abcdef_runtime_wavelength_tracking_benchmarks,
    )
    from abcdef_sim.physics.validation import format_benchmark_table

    comparisons = run_abcdef_runtime_wavelength_tracking_benchmarks(
        wavelength_counts,
        warmup_runs=warmup_runs,
        measured_runs=measured_runs,
    )
    content = "\n".join(
        [
            "# ABCDEF runtime wavelength tracking benchmark",
            "",
            f"- Warmup runs per case: {warmup_runs}",
            f"- Measured runs per case: {measured_runs}",
            "- Timings are machine-dependent and are documented here only as a sample run.",
            "- These timings compare the ABCDEF runtime `ThickLens.matrix(...)` plus batched"
            " `propagate_step(...)` path against scalar `raytracing` loops.",
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
    runtime_plot = write_abcdef_runtime_similarity_plot(
        output_dir / "abcdef_runtime_similarity.png"
    )
    helper_plot = write_abcd_helper_similarity_plot(output_dir / "abcd_helper_similarity.png")
    benchmark_path = write_benchmark_report(
        output_dir / "abcdef_runtime_wavelength_tracking_benchmarks.md",
        wavelength_counts=args.wavelength_counts,
        warmup_runs=args.warmup,
        measured_runs=args.repeats,
    )
    print(f"Wrote {runtime_plot}")
    print(f"Wrote {helper_plot}")
    print(f"Wrote {benchmark_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
