from __future__ import annotations

from typing import Literal

from abcdef_sim.data_models.specs import AbcdefCfg, FrameTransformCfg, FreeSpaceCfg, GratingCfg

__all__ = ["treacy_compressor_preset"]


def treacy_compressor_preset(
    *,
    name: str = "treacy_compressor",
    line_density_lpmm: float = 1200.0,
    incidence_angle_deg: float = 35.0,
    separation_um: float = 100_000.0,
    length_to_mirror_um: float = 0.0,
    diffraction_order: int = -1,
    n_passes: Literal[1, 2] = 2,
    immersion_refractive_index: float = 1.0,
    gap_medium_refractive_index: float = 1.0,
) -> AbcdefCfg:
    """Return a Treacy compressor preset assembled from existing optic primitives."""

    if separation_um < 0.0:
        raise ValueError("separation_um must be >= 0.")
    if length_to_mirror_um < 0.0:
        raise ValueError("length_to_mirror_um must be >= 0.")
    if n_passes not in (1, 2):
        raise ValueError("n_passes must be 1 or 2.")

    optics: list[FrameTransformCfg | FreeSpaceCfg | GratingCfg] = [
        GratingCfg(
            instance_name="g1",
            line_density_lpmm=float(line_density_lpmm),
            incidence_angle_deg=float(incidence_angle_deg),
            diffraction_order=int(diffraction_order),
            immersion_refractive_index=float(immersion_refractive_index),
        ),
        FreeSpaceCfg(
            instance_name="gap_12",
            length=float(separation_um),
            medium_refractive_index=float(gap_medium_refractive_index),
        ),
        GratingCfg(
            instance_name="g2",
            line_density_lpmm=float(line_density_lpmm),
            incidence_angle_deg=float(incidence_angle_deg),
            diffraction_order=int(diffraction_order),
            immersion_refractive_index=float(immersion_refractive_index),
        ),
    ]
    if n_passes == 2:
        optics.extend(
            [
                FreeSpaceCfg(
                    instance_name="to_fold",
                    length=float(length_to_mirror_um),
                    medium_refractive_index=float(gap_medium_refractive_index),
                ),
                FrameTransformCfg(
                    instance_name="fold_frame",
                    x_prime_scale=-1,
                ),
                FreeSpaceCfg(
                    instance_name="from_fold",
                    length=float(length_to_mirror_um),
                    medium_refractive_index=float(gap_medium_refractive_index),
                ),
                GratingCfg(
                    instance_name="g2_return",
                    line_density_lpmm=float(line_density_lpmm),
                    incidence_angle_deg=float(incidence_angle_deg),
                    diffraction_order=int(diffraction_order),
                    immersion_refractive_index=float(immersion_refractive_index),
                ),
                FreeSpaceCfg(
                    instance_name="gap_21",
                    length=float(separation_um),
                    medium_refractive_index=float(gap_medium_refractive_index),
                ),
                GratingCfg(
                    instance_name="g1_return",
                    line_density_lpmm=float(line_density_lpmm),
                    incidence_angle_deg=float(incidence_angle_deg),
                    diffraction_order=int(diffraction_order),
                    immersion_refractive_index=float(immersion_refractive_index),
                ),
            ]
        )

    return AbcdefCfg(
        name=name,
        optics=tuple(optics),
        tags={
            "preset_kind": "treacy_compressor",
            "line_density_lpmm": float(line_density_lpmm),
            "incidence_angle_deg": float(incidence_angle_deg),
            "separation_um": float(separation_um),
            "length_to_mirror_um": float(length_to_mirror_um),
            "diffraction_order": int(diffraction_order),
            "n_passes": int(n_passes),
        },
    )
