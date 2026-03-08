from __future__ import annotations

import pytest
from pydantic import ValidationError

from abcdef_sim.data_models.specs import OpticSpec


def test_optic_kind_pascal_case_is_preserved() -> None:
    spec = OpticSpec(kind="FreeSpace", instance_name="fs-p")

    assert spec.kind == "FreeSpace"


def test_optic_kind_snake_case_is_rejected() -> None:
    with pytest.raises(ValidationError, match="Unknown OpticSpec.kind"):
        OpticSpec(kind="free_space", instance_name="fs-s")
