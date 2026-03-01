from __future__ import annotations

from types import ModuleType

import pytest

from abcdef_sim import _phys_pipeline


class _Pkg:
    __path__: list[str] = []


def test_try_load_returns_none_when_required_export_is_missing() -> None:
    root = "phys_pipeline"

    policy = ModuleType("policy")
    policy.PolicyBag = object

    pipeline = ModuleType("pipeline")
    pipeline.SequentialPipeline = object

    types = ModuleType("types")
    types.PipelineStage = object
    types.StageConfig = object
    types.StageResult = object
    types.State = object
    # Deliberately omit hash_ndarray.

    modules = {
        f"{root}.policy": policy,
        f"{root}.pipeline": pipeline,
        f"{root}.types": types,
    }

    def fake_import_module(name: str) -> ModuleType:
        return modules[name]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(_phys_pipeline.importlib, "import_module", fake_import_module)
    try:
        assert _phys_pipeline._try_load(root) is None
    finally:
        monkeypatch.undo()


def test_load_exports_falls_back_to_versioned_root_when_unversioned_root_incomplete() -> None:
    pkg = _Pkg()

    policy_incomplete = ModuleType("policy")
    policy_incomplete.PolicyBag = object

    pipeline_incomplete = ModuleType("pipeline")
    pipeline_incomplete.SequentialPipeline = object

    types_incomplete = ModuleType("types")
    types_incomplete.PipelineStage = object
    types_incomplete.StageConfig = object
    types_incomplete.StageResult = object
    types_incomplete.State = object
    # Deliberately omit hash_ndarray to force fallback.

    policy_v1 = ModuleType("policy")
    policy_v1.PolicyBag = object

    pipeline_v1 = ModuleType("pipeline")
    pipeline_v1.SequentialPipeline = object

    types_v1 = ModuleType("types")
    types_v1.PipelineStage = object
    types_v1.StageConfig = object
    types_v1.StageResult = object
    types_v1.State = object
    types_v1.hash_ndarray = lambda value: value

    modules: dict[str, ModuleType] = {
        "phys_pipeline": pkg,
        "phys_pipeline.policy": policy_incomplete,
        "phys_pipeline.pipeline": pipeline_incomplete,
        "phys_pipeline.types": types_incomplete,
        "phys_pipeline.v1.policy": policy_v1,
        "phys_pipeline.v1.pipeline": pipeline_v1,
        "phys_pipeline.v1.types": types_v1,
    }

    class _ModuleInfo:
        def __init__(self, name: str) -> None:
            self.name = name

    def fake_import_module(name: str) -> ModuleType:
        return modules[name]

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(_phys_pipeline.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(
        _phys_pipeline.pkgutil,
        "iter_modules",
        lambda _path: [_ModuleInfo("v1")],
    )

    try:
        exports = _phys_pipeline._load_exports()
    finally:
        monkeypatch.undo()

    assert exports.hash_ndarray("ok") == "ok"
