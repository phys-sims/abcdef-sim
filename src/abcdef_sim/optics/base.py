"""Base types for optics abstractions.

This module is intentionally a placeholder for refactor scaffolding.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OpticModel:
    """Stub base optic model for package import scaffolding."""

    kind: str = "placeholder"
