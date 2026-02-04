from __future__ import annotations

from dataclasses import dataclass

from phys_pipeline.policy import PolicyBag
from phys_pipeline.types import PipelineStage, StageResult, State  


from abcdef_sim.data_models.configs import OpticStageCfg

@dataclass
class AbcdefOpticStage(PipelineStage[State, OpticStageCfg]):
    """
    One optic as a pipeline stage.
    (Implement actual physics in process.)
    """
    cfg: OpticStageCfg
    name: str = ""
    version: str = "v1"

    def __post_init__(self):
        if not self.name:
            self.name = f"{self.cfg.optic_name}:{self.cfg.instance_name}"

    def process(self, state: State, *, policy: PolicyBag | None = None) -> StageResult:
        # TODO: apply cfg.abcdef/cfg.refractive_index to update state
        # This is where ray/beam propagation happens.
        return StageResult(state=state)  # placeholder