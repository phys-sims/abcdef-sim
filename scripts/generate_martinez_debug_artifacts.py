#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def generate_artifacts(output_dir: Path) -> tuple[Path, ...]:
    import matplotlib.pyplot as plt
    import numpy as np

    from abcdef_sim.analytics.martinez_debug import (
        DEFAULT_SPAN_SCALES,
        run_martinez_section_iv_coefficient_audit,
        run_martinez_section_iv_phase_study,
        run_treacy_partition_span_study,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    section_iv_coeff_points = run_martinez_section_iv_coefficient_audit()
    section_iv_phase_points = run_martinez_section_iv_phase_study()
    treacy_partition_points = run_treacy_partition_span_study()

    coeff_json = output_dir / "martinez_section_iv_matrix_audit.json"
    coeff_png = output_dir / "martinez_section_iv_matrix_audit.png"
    phase_json = output_dir / "martinez_section_iv_phase_partition.json"
    phase_png = output_dir / "martinez_section_iv_phase_partition.png"
    treacy_json = output_dir / "treacy_phi_partition_span_study.json"
    treacy_png = output_dir / "treacy_phi_partition_span_study.png"

    coeff_json.write_text(
        json.dumps(
            {
                "plot_description": (
                    "Martinez Section IV matrix audit from Eq. (31)-(34). "
                    "The total system matrix is compared against the first-order "
                    "coefficient formulas for A, B, C, D, E, and F."
                ),
                "points": [point.to_dict() for point in section_iv_coeff_points],
            },
            indent=2,
        )
    )
    phase_json.write_text(
        json.dumps(
            {
                "plot_description": (
                    "Martinez Section IV phase study. Variants compare the paper-style "
                    "phi3 construction using sample-k vs center-k and first- vs second-order "
                    "F truncations against the direct Eq. (36) phase reference."
                ),
                "span_scales": list(DEFAULT_SPAN_SCALES),
                "points": [point.to_dict() for point in section_iv_phase_points],
            },
            indent=2,
        )
    )
    treacy_json.write_text(
        json.dumps(
            {
                "plot_description": (
                    "Matched double-pass Treacy raw-runtime span study. Variants compare "
                    "axial vs geometric phi0-like propagation and current vs truncated phi3."
                ),
                "span_scales": list(DEFAULT_SPAN_SCALES),
                "points": [point.to_dict() for point in treacy_partition_points],
            },
            indent=2,
        )
    )

    _plot_section_iv_matrix_audit(section_iv_coeff_points, coeff_png, plt, np)
    _plot_section_iv_phase_partition(section_iv_phase_points, phase_png, plt, np)
    _plot_treacy_partition_span_study(treacy_partition_points, treacy_png, plt, np)
    return coeff_json, coeff_png, phase_json, phase_png, treacy_json, treacy_png


def _plot_section_iv_matrix_audit(points, output_path: Path, plt, np) -> None:
    delta = np.array([point.delta_omega_rad_per_fs for point in points], dtype=np.float64)
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    entries = [
        ("A", [point.system_a for point in points], [point.expected_a for point in points]),
        ("B (um)", [point.system_b_um for point in points], [point.expected_b_um for point in points]),
        ("C (1/um)", [point.system_c_per_um for point in points], [point.expected_c_per_um for point in points]),
        ("D", [point.system_d for point in points], [point.expected_d for point in points]),
        ("E (um)", [point.system_e_um for point in points], [point.expected_e_um for point in points]),
        ("F", [point.system_f for point in points], [point.expected_f for point in points]),
    ]
    for ax, (title, actual, expected) in zip(axes.flat, entries, strict=True):
        ax.plot(delta, np.asarray(actual, dtype=np.float64), label="matrix", color="tab:blue")
        ax.plot(delta, np.asarray(expected, dtype=np.float64), label="Eq. 34", color="tab:orange")
        ax.set_title(title)
        ax.set_xlabel("delta omega (rad/fs)")
        ax.grid(True, alpha=0.3)
        if title in {"B (um)", "E (um)", "F", "C (1/um)"}:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].legend()
    fig.suptitle("Martinez Section IV Matrix Audit")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_section_iv_phase_partition(points, output_path: Path, plt, np) -> None:
    variants = sorted({point.variant for point in points})
    span_scales = np.array(sorted({point.span_scale for point in points}), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    for variant in variants:
        variant_points = [point for point in points if point.variant == variant]
        variant_points.sort(key=lambda point: point.span_scale)
        axes[0].plot(
            span_scales,
            [point.gdd_abs_error_fs2 for point in variant_points],
            marker="o",
            label=variant,
        )
        axes[1].plot(
            span_scales,
            [point.tod_abs_error_fs3 for point in variant_points],
            marker="o",
            label=variant,
        )

    for ax, ylabel in zip(
        axes,
        ("|GDD - Eq.36 reference| (fs^2)", "|TOD - Eq.36 reference| (fs^3)"),
        strict=True,
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Relative spectral span scale")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Martinez Section IV Phase Partition Study")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_treacy_partition_span_study(points, output_path: Path, plt, np) -> None:
    variants = sorted({point.variant for point in points})
    span_scales = np.array(sorted({point.span_scale for point in points}), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    for variant in variants:
        variant_points = [point for point in points if point.variant == variant]
        variant_points.sort(key=lambda point: point.span_scale)
        axes[0].plot(
            span_scales,
            [point.raw_abcdef_gdd_rel_error for point in variant_points],
            marker="o",
            label=variant,
        )
        axes[1].plot(
            span_scales,
            [point.raw_abcdef_tod_rel_error for point in variant_points],
            marker="o",
            label=variant,
        )

    for ax, ylabel in zip(
        axes,
        ("GDD relative error", "TOD relative error"),
        strict=True,
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Relative spectral span scale")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Treacy Raw Runtime Partition Study vs Spectral Span")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "physics" / "martinez_debug",
    )
    args = parser.parse_args()
    outputs = generate_artifacts(args.output_dir.resolve())
    for path in outputs:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
