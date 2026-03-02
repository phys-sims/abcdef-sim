from __future__ import annotations

from abcdef_sim.data_models.specs import OpticSpec
from abcdef_sim.optics.freespace import FreeSpace
from abcdef_sim.optics.registry import OpticFactory


def test_factory_builds_same_optic_class_for_pascal_and_snake_case_kind() -> None:
    factory = OpticFactory.default()

    pascal = factory.build(OpticSpec(kind="FreeSpace", instance_name="fs-p", params={"L": 1.0}))
    snake = factory.build(OpticSpec(kind="free_space", instance_name="fs-s", params={"L": 1.0}))

    assert isinstance(pascal, FreeSpace)
    assert isinstance(snake, FreeSpace)
    assert type(pascal) is type(snake)
