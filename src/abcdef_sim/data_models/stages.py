from __future__ import annotations

from dataclasses import dataclass

from abcdef_sim._phys_pipeline import PipelineStage, PolicyBag, StageResult, State
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.states import PHASE_CONTRIBUTIONS_META_KEY, RayState
from abcdef_sim.physics.abcdef.adapters import apply_cfg


@dataclass
class AbcdefOpticStage(PipelineStage[State, OpticStageCfg]):
    """One optic as a pipeline stage wired to the ABCDEF adapter boundary."""

    cfg: OpticStageCfg
    name: str = ""
    version: str = "v1"

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.cfg.optic_name}:{self.cfg.instance_name}"

    def process(self, state: State, *, policy: PolicyBag | None = None) -> StageResult:
        if not isinstance(state, RayState):
            raise TypeError(f"AbcdefOpticStage requires RayState input; got {type(state).__name__}")

        new_state, contribution = apply_cfg(state=state, cfg=self.cfg, policy=policy)
        existing_contributions = new_state.meta.get(PHASE_CONTRIBUTIONS_META_KEY, ())
        if not isinstance(existing_contributions, tuple):
            raise TypeError(
                "RayState.meta['phase_contributions'] must be a tuple "
                "for deterministic accumulation"
            )

        new_state.meta[PHASE_CONTRIBUTIONS_META_KEY] = existing_contributions + (contribution,)
        return StageResult(state=new_state)
