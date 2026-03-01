"""Optical component abstractions and registration utilities."""

from abcdef_sim.optics.base import OpticModel
from abcdef_sim.optics.freespace import FreeSpaceModel
from abcdef_sim.optics.registry import OpticsRegistry

__all__ = ["OpticModel", "FreeSpaceModel", "OpticsRegistry"]
