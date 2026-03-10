"""Analytic and benchmarking helpers built on top of the runtime package surface."""

from abcdef_sim.analytics.spatiospectral import (
    OutputPlaneField1D,
    OutputPlaneFieldSummary,
    OutputPlaneSpatialMetrics,
    build_output_plane_field_1d,
    summarize_output_plane_field,
    summarize_output_plane_geometry,
)
from abcdef_sim.analytics.treacy_benchmark import (
    DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM,
    DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM,
    TreacyBenchmarkPoint,
    run_treacy_benchmark_point,
    run_treacy_mirror_heatmap,
    run_treacy_radius_convergence,
)

__all__ = [
    "DEFAULT_TREACY_BENCHMARK_BEAM_RADII_MM",
    "DEFAULT_TREACY_BENCHMARK_MIRROR_LENGTHS_UM",
    "OutputPlaneField1D",
    "OutputPlaneFieldSummary",
    "OutputPlaneSpatialMetrics",
    "TreacyBenchmarkPoint",
    "build_output_plane_field_1d",
    "run_treacy_benchmark_point",
    "run_treacy_mirror_heatmap",
    "run_treacy_radius_convergence",
    "summarize_output_plane_field",
    "summarize_output_plane_geometry",
]
