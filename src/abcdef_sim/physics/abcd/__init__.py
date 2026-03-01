"""ABCD paraxial optics primitives and adapters."""

from abcdef_sim.physics.abcd.gaussian import q_propagate, rayleigh_range_from_waist
from abcdef_sim.physics.abcd.matrices import compose, free_space, interface, thick_lens, thin_lens
from abcdef_sim.physics.abcd.ray import Ray, propagate_ray
from abcdef_sim.physics.abcd.raytracing_ref import from_raytracing_matrix

__all__ = [
    "Ray",
    "compose",
    "free_space",
    "from_raytracing_matrix",
    "interface",
    "propagate_ray",
    "q_propagate",
    "rayleigh_range_from_waist",
    "thick_lens",
    "thin_lens",
]
