"""Compatibility imports for phys-pipeline API surfaces."""

from __future__ import annotations

import importlib
import pkgutil
import re
from collections.abc import Sequence
from types import ModuleType

_REQUIRED_EXPORTS: tuple[str, ...] = (
    "PipelineStage",
    "PolicyBag",
    "SequentialPipeline",
    "StageConfig",
    "StageResult",
    "State",
    "hash_ndarray",
)


def _candidate_roots() -> list[str]:
    """Return candidate phys-pipeline roots without hardcoding a version preference."""

    roots: list[str] = ["phys_pipeline"]
    try:
        pkg = importlib.import_module("phys_pipeline")
    except ModuleNotFoundError:
        return roots

    if hasattr(pkg, "__path__"):
        versioned: list[tuple[int, str]] = []
        for module_info in pkgutil.iter_modules(pkg.__path__):
            name = module_info.name
            match = re.fullmatch(r"v(\d+)", name)
            if match:
                versioned.append((int(match.group(1)), f"phys_pipeline.{name}"))

        # Prefer higher version numbers when multiple version namespaces exist.
        roots.extend(root for _, root in sorted(versioned, reverse=True))

    return roots


def _try_load(root: str) -> ModuleType | None:
    """Load policy/pipeline/types modules for a given import root if available."""

    try:
        policy = importlib.import_module(f"{root}.policy")
        pipeline = importlib.import_module(f"{root}.pipeline")
        types = importlib.import_module(f"{root}.types")
    except ModuleNotFoundError:
        return None

    namespace = ModuleType("phys_pipeline_compat")
    try:
        setattr(namespace, "PolicyBag", getattr(policy, "PolicyBag"))
        setattr(
            namespace,
            "SequentialPipeline",
            getattr(pipeline, "SequentialPipeline"),
        )
        setattr(namespace, "PipelineStage", getattr(types, "PipelineStage"))
        setattr(namespace, "StageConfig", getattr(types, "StageConfig"))
        setattr(namespace, "StageResult", getattr(types, "StageResult"))
        setattr(namespace, "State", getattr(types, "State"))
        setattr(namespace, "hash_ndarray", getattr(types, "hash_ndarray"))
    except AttributeError:
        return None

    return namespace


def _load_exports() -> ModuleType:
    for root in _candidate_roots():
        namespace = _try_load(root)
        if namespace is not None:
            return namespace
    raise ModuleNotFoundError(
        "Could not locate compatible phys_pipeline modules under any known import root."
    )


_exports = _load_exports()

PipelineStage = _exports.PipelineStage
PolicyBag = _exports.PolicyBag
SequentialPipeline = _exports.SequentialPipeline
StageConfig = _exports.StageConfig
StageResult = _exports.StageResult
State = _exports.State
hash_ndarray = _exports.hash_ndarray

__all__: Sequence[str] = _REQUIRED_EXPORTS
