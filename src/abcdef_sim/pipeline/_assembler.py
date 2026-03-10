from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from abcdef_sim._phys_pipeline import PolicyBag, SequentialPipeline
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.specs import LaserSpec, OpticSpec, SystemPreset
from abcdef_sim.data_models.stages import AbcdefOpticStage
from abcdef_sim.optics.registry import OpticFactory


@dataclass
class SystemAssembler:
    """
    Pure assembly layer.
    Turns (SystemPreset, LaserSpec) into a list of stages (each has cfg).
    """

    factory: OpticFactory
    cfg_gen: OpticStageCfgGenerator

    def build_optic_cfgs(
        self,
        preset: SystemPreset,
        laser: LaserSpec,
        *,
        policy: PolicyBag | None = None,
    ) -> list[OpticStageCfg]:
        w = laser.omega()
        delta_w = laser.delta_omega()
        omega0 = laser.omega0_rad_per_fs

        cfgs: list[OpticStageCfg] = []
        specs = list(preset.optics)
        for spec in specs:
            optic = self.factory.build(spec)
            cfg = self.cfg_gen.build(
                optic,
                w,
                delta_omega=delta_w,
                omega0_rad_per_fs=omega0,
                tags={"preset": preset.name, "optic_kind": spec.kind},
                policy=policy,
            )
            cfgs.append(cfg)
        return _attach_phi_geom_surface_metadata(cfgs, specs)

    def build_pipeline(
        self,
        preset: SystemPreset,
        laser: LaserSpec,
        *,
        policy: PolicyBag | None = None,
        pipeline_name: str | None = None,
    ) -> Any:
        cfgs = self.build_optic_cfgs(preset, laser, policy=policy)
        stages = [AbcdefOpticStage(cfg=c) for c in cfgs]
        return SequentialPipeline(stages=stages, name=pipeline_name or preset.name)


def _attach_phi_geom_surface_metadata(
    cfgs: list[OpticStageCfg],
    specs: list[OpticSpec],
) -> list[OpticStageCfg]:
    if len(cfgs) != len(specs):
        raise ValueError("cfgs and specs must have matching lengths")

    updated_cfgs: list[OpticStageCfg] = []
    for idx, (cfg, spec) in enumerate(zip(cfgs, specs, strict=True)):
        if spec.kind != "FreeSpace":
            updated_cfgs.append(cfg)
            continue

        next_planar_incidence_angle_rad = _next_planar_surface_incidence_angle_rad(
            specs, start=idx + 1
        )
        if next_planar_incidence_angle_rad is None:
            updated_cfgs.append(cfg)
            continue

        updated_cfgs.append(
            cfg.model_copy(
                update={
                    "phi_geom_model": "free_space_to_planar_surface",
                    "next_surface_incidence_angle_rad": next_planar_incidence_angle_rad,
                }
            )
        )
    return updated_cfgs


def _next_planar_surface_incidence_angle_rad(
    specs: list[OpticSpec],
    *,
    start: int,
) -> float | None:
    for spec in specs[start:]:
        if spec.kind == "FrameTransform":
            continue
        if spec.kind == "Grating":
            return math.radians(float(spec.params["incidence_angle_deg"]))
        return None
    return None
