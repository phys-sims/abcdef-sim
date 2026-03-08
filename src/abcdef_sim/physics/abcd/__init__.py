"""ABCD paraxial optics primitives and adapters."""

from abcdef_sim.physics.abcd.gaussian import (
    beam_radius_from_q,
    distance_from_waist,
    q_from_waist,
    q_propagate,
    rayleigh_range_from_q,
    rayleigh_range_from_waist,
    sample_beam_radius_profile,
    waist_position_from_plane,
    waist_radius_from_q,
)
from abcdef_sim.physics.abcd.lenses import (
    DoubletAssembly,
    SellmeierMaterial,
    ThickLensSpec,
    doublet_matrix,
    propagate_q_through_doublet,
    propagate_q_through_thick_lens,
    refractive_index_sellmeier,
    sample_doublet_beam_radius_profile,
    sample_thick_lens_beam_radius_profile,
    thick_lens_matrix_for_spec,
)
from abcdef_sim.physics.abcd.matrices import compose, free_space, interface, thick_lens, thin_lens
from abcdef_sim.physics.abcd.ray import Ray, propagate_ray
from abcdef_sim.physics.abcd.raytracing_ref import from_raytracing_matrix

__all__ = [
    "Ray",
    "DoubletAssembly",
    "SellmeierMaterial",
    "ThickLensSpec",
    "beam_radius_from_q",
    "compose",
    "distance_from_waist",
    "doublet_matrix",
    "free_space",
    "from_raytracing_matrix",
    "interface",
    "propagate_ray",
    "propagate_q_through_doublet",
    "propagate_q_through_thick_lens",
    "q_from_waist",
    "q_propagate",
    "rayleigh_range_from_q",
    "rayleigh_range_from_waist",
    "refractive_index_sellmeier",
    "sample_beam_radius_profile",
    "sample_doublet_beam_radius_profile",
    "sample_thick_lens_beam_radius_profile",
    "thick_lens",
    "thick_lens_matrix_for_spec",
    "thin_lens",
    "waist_position_from_plane",
    "waist_radius_from_q",
]
