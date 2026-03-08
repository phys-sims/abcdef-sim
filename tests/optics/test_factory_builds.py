from __future__ import annotations

from abcdef_sim.data_models.specs import OpticSpec
from abcdef_sim.optics.freespace import FreeSpace
from abcdef_sim.optics.registry import OpticFactory


def test_factory_builds_freespace_for_pascal_case_kind() -> None:
    factory = OpticFactory.default()

    optic = factory.build(OpticSpec(kind="FreeSpace", instance_name="fs-p", params={"L": 1.0}))

    assert isinstance(optic, FreeSpace)
