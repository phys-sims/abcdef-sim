#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from abcdef_sim.physics.abcd.gaussian import q_from_waist
from abcdef_sim.physics.abcd.lenses import (
    DoubletAssembly,
    SellmeierMaterial,
    ThickLensSpec,
    sample_doublet_beam_radius_profile,
)
from abcdef_sim.physics.abcd.matrices import thick_lens
from abcdef_sim.physics.abcd.ray import Ray, propagate_ray
from abcdef_sim.physics.abcd.raytracing_ref import (
    propagate_gaussian_beam_raytracing,
    raytracing_space,
    raytracing_thick_lens,
    sample_gaussian_beam_radii_raytracing,
)
from abcdef_sim.physics.validation import ObservableComparison


def _n_lak22() -> SellmeierMaterial:
    return SellmeierMaterial(
        name="N-LAK22",
        b_terms=(1.14229781, 0.535138441, 1.04088385),
        c_terms=(0.00585778594e-6, 0.0198546147e-6, 100.834017e-6),
    )


def _n_sf6() -> SellmeierMaterial:
    return SellmeierMaterial(
        name="N-SF6",
        b_terms=(1.77931763, 0.338149866, 2.08734474),
        c_terms=(0.0133714182e-6, 0.0617533621e-6, 174.01759e-6),
    )


def _build_reference_doublet() -> DoubletAssembly:
    return DoubletAssembly(
        first=ThickLensSpec(
            refractive_index=_n_lak22(),
            R1=112.88,
            R2=-112.88,
            thickness=6.0,
            n_in=1.0,
            n_out=1.0,
        ),
        second=ThickLensSpec(
            refractive_index=_n_sf6(),
            R1=-112.88,
            R2=-1415.62,
            thickness=4.0,
            n_in=1.0,
            n_out=1.0,
        ),
        gap=0.0,
    )


def _monochromatic_ray_comparison() -> ObservableComparison:
    import raytracing as rt

    input_heights = np.linspace(-2.0, 2.0, 41, dtype=float)
    matrix = thick_lens(n_lens=1.5, R1=50.0, R2=-50.0, thickness=10.0)
    oracle = rt.ThickLens(n=1.5, R1=50.0, R2=-50.0, thickness=10.0)

    local = np.array(
        [propagate_ray(Ray(y=float(height), theta=0.0), matrix).theta for height in input_heights],
        dtype=float,
    )
    reference = np.array(
        [(oracle * rt.Ray(y=float(height), theta=0.0)).theta for height in input_heights],
        dtype=float,
    )
    return ObservableComparison(
        name="Monochromatic thick lens",
        observable_label="theta_out (rad)",
        coordinates=input_heights,
        coordinate_label="input ray height",
        local=local,
        reference=reference,
    )


def _doublet_profile_comparisons() -> list[ObservableComparison]:
    doublet = _build_reference_doublet()
    z_samples = np.linspace(0.0, 250.0, 161, dtype=float)
    comparisons: list[ObservableComparison] = []

    for wavelength, label in [(0.810, "810 nm"), (1.550, "1550 nm")]:
        q_in = q_from_waist(waist_radius=1.024, wavelength=wavelength, distance_from_waist=0.0)
        local = sample_doublet_beam_radius_profile(q_in, doublet, wavelength, z_samples)

        first_n = float(doublet.first.refractive_index.refractive_index(wavelength))
        second_n = float(doublet.second.refractive_index.refractive_index(wavelength))
        elements = [
            raytracing_thick_lens(
                n=first_n,
                R1=doublet.first.R1,
                R2=doublet.first.R2,
                thickness=doublet.first.thickness,
                n_in=doublet.first.n_in,
                n_out=doublet.first.n_out,
            )
        ]
        if doublet.gap > 0.0:
            elements.append(raytracing_space(d=doublet.gap))
        elements.append(
            raytracing_thick_lens(
                n=second_n,
                R1=doublet.second.R1,
                R2=doublet.second.R2,
                thickness=doublet.second.thickness,
                n_in=doublet.second.n_in,
                n_out=doublet.second.n_out,
            )
        )
        oracle_beam = propagate_gaussian_beam_raytracing(q_in, wavelength, elements)
        reference = sample_gaussian_beam_radii_raytracing(oracle_beam, z_samples)
        comparisons.append(
            ObservableComparison(
                name=f"Wavelength-dependent doublet ({label})",
                observable_label="beam radius (mm)",
                coordinates=z_samples,
                coordinate_label="propagation distance",
                local=local,
                reference=reference,
            )
        )

    return comparisons


def generate_plot(output_path: Path) -> Path:
    mono = _monochromatic_ray_comparison()
    profiles = _doublet_profile_comparisons()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    axes[0, 0].plot(mono.coordinates, mono.local, color="tab:blue", label="abcdef-sim")
    axes[0, 0].plot(
        mono.coordinates,
        mono.reference,
        color="black",
        linestyle="--",
        label="RayTracing",
    )
    axes[0, 0].set_title(mono.name)
    axes[0, 0].set_xlabel("Input ray height (mm)")
    axes[0, 0].set_ylabel(mono.observable_label)
    axes[0, 0].legend()

    axes[1, 0].plot(mono.coordinates, mono.residual(), color="tab:red")
    axes[1, 0].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 0].set_xlabel("Input ray height (mm)")
    axes[1, 0].set_ylabel("Residual")

    colors = ["tab:green", "tab:orange"]
    for comparison, color in zip(profiles, colors, strict=True):
        axes[0, 1].plot(
            comparison.coordinates,
            comparison.local,
            color=color,
            label=f"{comparison.name} local",
        )
        axes[0, 1].plot(
            comparison.coordinates,
            comparison.reference,
            color=color,
            linestyle="--",
            label=f"{comparison.name} reference",
        )
        axes[1, 1].plot(
            comparison.coordinates,
            comparison.residual(),
            color=color,
            label=comparison.name,
        )

    axes[0, 1].set_title("Wavelength-dependent doublet beam radius")
    axes[0, 1].set_xlabel("Propagation distance (mm)")
    axes[0, 1].set_ylabel("Beam radius (mm)")
    axes[0, 1].legend(fontsize=8)

    axes[1, 1].axhline(0.0, color="black", linewidth=0.8)
    axes[1, 1].set_xlabel("Propagation distance (mm)")
    axes[1, 1].set_ylabel("Residual")
    axes[1, 1].legend(fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "physics" / "thick_lens_vs_raytracing.png",
    )
    args = parser.parse_args()
    output_path = generate_plot(args.output.resolve())
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
