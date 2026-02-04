from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

from abcdef_sim.data_models.specs import OpticSpec, OpticKind
from abcdef_sim.data_models.optics import Optic, FreeSpace

BuilderFn = Callable[[OpticSpec], Optic]


@dataclass
class OpticFactory:
    """
    Converts OpticSpec -> Optic instance.
    Centralizing this avoids scattered ad-hoc instantiation logic.
    """
    registry: Dict[OpticKind, BuilderFn]

    @staticmethod
    def default() -> "OpticFactory":
        def build_free_space(spec: OpticSpec) -> Optic:
            L = spec.params.get("L", 0.0)
            return FreeSpace(name="FreeSpace", instance_name=spec.instance_name, _length=L)

        # def build_grating(spec: OpticSpec) -> Optic:
        #     return Grating(name="Grating", instance_name=spec.instance_name, **spec.params)

        reg: Dict[OpticKind, BuilderFn] = {
            "FreeSpace": build_free_space,
            # "Grating": build_grating,
        }
        return OpticFactory(registry=reg)

    def build(self, spec: OpticSpec) -> Optic:
        try:
            fn = self.registry[spec.kind]
        except KeyError as e:
            raise ValueError(f"No optic builder registered for kind={spec.kind!r}") from e
        return fn(spec)
