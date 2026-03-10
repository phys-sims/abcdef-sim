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
        run_martinez_section_iv_phi3_variant_study,
        run_treacy_phi3_per_grating_budget,
        run_treacy_phi3_sign_audit,
        run_treacy_phi3_variant_comparison,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    section_iv_coeff_points = run_martinez_section_iv_coefficient_audit()
    section_iv_phi3_points = run_martinez_section_iv_phi3_variant_study()
    treacy_variant_points = run_treacy_phi3_variant_comparison()
    best_variant = _best_variant_name(treacy_variant_points)
    treacy_budget_points = run_treacy_phi3_per_grating_budget(
        variants=("runtime_exact_centerk", best_variant)
    )
    sign_stage_points, sign_variant_points = run_treacy_phi3_sign_audit(variant_name=best_variant)

    coeff_json = output_dir / "martinez_section_iv_matrix_audit.json"
    coeff_png = output_dir / "martinez_section_iv_matrix_audit.png"
    phase_json = output_dir / "martinez_section_iv_phi3_variants.json"
    phase_png = output_dir / "martinez_section_iv_phi3_variants.png"
    treacy_json = output_dir / "treacy_phi3_variant_comparison.json"
    treacy_png = output_dir / "treacy_phi3_variant_comparison.png"
    budget_json = output_dir / "treacy_phi3_per_grating_budget.json"
    budget_png = output_dir / "treacy_phi3_per_grating_budget.png"
    sign_json = output_dir / "treacy_phi3_sign_audit.json"
    sign_png = output_dir / "treacy_phi3_sign_audit.png"

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
                    "Martinez Section IV phi3 variant study. Variants compare runtime-style "
                    "transport-exact phi3 against Martinez F0 / Eq. (11) phase-side models "
                    "using the Eq. (36)/(37)/(41) reference as the oracle."
                ),
                "span_scales": list(DEFAULT_SPAN_SCALES),
                "points": [point.to_dict() for point in section_iv_phi3_points],
            },
            indent=2,
        )
    )
    treacy_json.write_text(
        json.dumps(
            {
                "plot_description": (
                    "Matched double-pass Treacy phi3 variant comparison on top of the current "
                    "geometry-backed propagation baseline."
                ),
                "beam_radii_mm": sorted(
                    {float(point.beam_radius_mm) for point in treacy_variant_points}
                ),
                "points": [point.to_dict() for point in treacy_variant_points],
            },
            indent=2,
        )
    )
    budget_json.write_text(
        json.dumps(
            {
                "plot_description": (
                    "Per-grating phi3 budget for Treacy. Bars compare the current runtime-style "
                    "transport-exact phi3 against the best large-beam candidate variant, plus "
                    "the residual contribution projected from the analytic mismatch."
                ),
                "selected_variants": ["runtime_exact_centerk", best_variant],
                "points": [point.to_dict() for point in treacy_budget_points],
            },
            indent=2,
        )
    )
    sign_json.write_text(
        json.dumps(
            {
                "plot_description": (
                    "Treacy phi3 sign/orientation audit. Stage metadata captures local frame "
                    "angles and sign conventions; variant points compare current signs against "
                    "return-pass and all-grating sign flips."
                ),
                "selected_variant": best_variant,
                "stage_points": [point.to_dict() for point in sign_stage_points],
                "variant_points": [point.to_dict() for point in sign_variant_points],
            },
            indent=2,
        )
    )

    _plot_section_iv_matrix_audit(section_iv_coeff_points, coeff_png, plt, np)
    _plot_section_iv_phi3_variants(section_iv_phi3_points, phase_png, plt, np)
    _plot_treacy_phi3_variant_comparison(treacy_variant_points, treacy_png, plt, np)
    _plot_treacy_phi3_per_grating_budget(treacy_budget_points, budget_png, plt, np)
    _plot_treacy_phi3_sign_audit(sign_stage_points, sign_variant_points, sign_png, plt, np)
    return (
        coeff_json,
        coeff_png,
        phase_json,
        phase_png,
        treacy_json,
        treacy_png,
        budget_json,
        budget_png,
        sign_json,
        sign_png,
    )


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


def _plot_section_iv_phi3_variants(points, output_path: Path, plt, np) -> None:
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

    fig.suptitle("Martinez Section IV Phi3 Variant Study")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_treacy_phi3_variant_comparison(points, output_path: Path, plt, np) -> None:
    variants = sorted({point.variant for point in points})
    beam_radii = np.array(sorted({point.beam_radius_mm for point in points}), dtype=np.float64)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), constrained_layout=True)

    for variant in variants:
        variant_points = [point for point in points if point.variant == variant]
        variant_points.sort(key=lambda point: point.beam_radius_mm)
        axes[0].plot(
            beam_radii,
            [point.raw_abcdef_gdd_rel_error for point in variant_points],
            marker="o",
            label=variant,
        )
        axes[1].plot(
            beam_radii,
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
        ax.set_xlabel("Beam radius (mm)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.suptitle("Treacy Phi3 Variant Comparison")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_treacy_phi3_per_grating_budget(points, output_path: Path, plt, np) -> None:
    variants = sorted({point.variant for point in points})
    stages = sorted({point.instance_name for point in points}, key=lambda name: ("return" in name, name))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(stages), dtype=np.float64)
    width = 0.35 if len(variants) <= 2 else 0.8 / max(len(variants), 1)

    for idx, variant in enumerate(variants):
        variant_points = [point for point in points if point.variant == variant]
        point_map = {point.instance_name: point for point in variant_points}
        offsets = x + ((idx - (len(variants) - 1) / 2.0) * width)
        axes[0].bar(
            offsets,
            [point_map[stage].phi3_gdd_fs2 for stage in stages],
            width=width,
            label=f"{variant} phi3",
        )
        axes[1].bar(
            offsets,
            [point_map[stage].required_residual_gdd_fs2 for stage in stages],
            width=width,
            label=f"{variant} required residual",
        )

    axes[0].set_title("Per-grating phi3 GDD")
    axes[1].set_title("Projected required residual GDD")
    for ax in axes:
        ax.set_xticks(x, stages, rotation=20, ha="right")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend()

    fig.suptitle("Treacy Per-grating Phi3 Budget")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _plot_treacy_phi3_sign_audit(stage_points, variant_points, output_path: Path, plt, np) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(stage_points), dtype=np.float64)
    axes[0].scatter(
        x,
        [point.local_axis_angle_deg for point in stage_points],
        label="local z axis",
        color="tab:blue",
    )
    axes[0].scatter(
        x,
        [point.positive_x_axis_angle_deg for point in stage_points],
        label="positive x axis",
        color="tab:orange",
    )
    for idx, point in enumerate(stage_points):
        axes[0].text(
            x[idx],
            point.local_axis_angle_deg,
            point.instance_name,
            fontsize=8,
            rotation=15,
            ha="left",
            va="bottom",
        )
    axes[0].set_title("Local frame orientation by grating stage")
    axes[0].set_ylabel("Angle (deg)")
    axes[0].set_xticks([])
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    sign_cases = [point.sign_case for point in variant_points]
    axes[1].bar(
        np.arange(len(sign_cases), dtype=np.float64) - 0.15,
        [point.raw_abcdef_gdd_rel_error for point in variant_points],
        width=0.3,
        label="GDD error",
    )
    axes[1].bar(
        np.arange(len(sign_cases), dtype=np.float64) + 0.15,
        [point.raw_abcdef_tod_rel_error for point in variant_points],
        width=0.3,
        label="TOD error",
    )
    axes[1].set_xticks(np.arange(len(sign_cases), dtype=np.float64), sign_cases, rotation=15)
    axes[1].set_yscale("log")
    axes[1].set_title("Sign-variant error comparison")
    axes[1].grid(True, axis="y", which="both", alpha=0.3)
    axes[1].legend()

    fig.suptitle("Treacy Phi3 Sign Audit")
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def _best_variant_name(points) -> str:
    grouped: dict[str, list[object]] = {}
    for point in points:
        grouped.setdefault(point.variant, []).append(point)
    return min(
        grouped.items(),
        key=lambda item: sum(
            point.raw_abcdef_gdd_rel_error + point.raw_abcdef_tod_rel_error
            for point in item[1]
            if point.beam_radius_mm >= 10.0
        ),
    )[0]


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
