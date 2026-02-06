from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from phys_pipeline.v1.policy import PolicyBag
from phys_pipeline.v1.types import PipelineStage, StageResult, State 

from abcdef_sim.data_models.specs import SystemPreset, LaserSpec
from abcdef_sim.data_models.factory import OpticFactory
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.stages import AbcdefOpticStage

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

        cfgs: list[OpticStageCfg] = []
        for spec in preset.optics:
            optic = self.factory.build(spec)
            cfg = self.cfg_gen.build(
                optic,
                w,
                tags={"preset": preset.name, "optic_kind": spec.kind},
                policy=policy,
            )
            cfgs.append(cfg)
        return cfgs

    def build_pipeline(
        self,
        preset: SystemPreset,
        laser: LaserSpec,
        *,
        policy: PolicyBag | None = None,
        pipeline_name: str | None = None,
    ):
        from phys_pipeline.v1.pipeline import SequentialPipeline  # current executor 
        cfgs = self.build_optic_cfgs(preset, laser, policy=policy)
        stages = [AbcdefOpticStage(cfg=c) for c in cfgs]
        return SequentialPipeline(stages=stages, name=pipeline_name or preset.name)
