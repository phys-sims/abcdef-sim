"""Backward-compatible optic runtime imports.

DEPRECATED: Runtime optic behavior moved out of ``abcdef_sim.data_models``.
Import from ``abcdef_sim.optics.base`` and ``abcdef_sim.optics.freespace`` instead.
"""

from abcdef_sim.optics.base import Optic
from abcdef_sim.optics.freespace import FreeSpace

__all__ = ["Optic", "FreeSpace"]
