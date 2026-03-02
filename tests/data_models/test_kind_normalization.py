from __future__ import annotations

from abcdef_sim.data_models.specs import OpticSpec


def test_optic_kind_pascal_and_snake_case_normalize_to_same_internal_value() -> None:
    pascal = OpticSpec(kind="FreeSpace", instance_name="fs-p")
    snake = OpticSpec(kind="free_space", instance_name="fs-s")

    assert pascal.kind == snake.kind == "FreeSpace"
