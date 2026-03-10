from __future__ import annotations

import math

import numpy as np
import pytest

from abcdef_sim import treacy_compressor_preset
from abcdef_sim.cache.backend import NullCacheBackend
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.specs import LaserSpec
from abcdef_sim.data_models.states import RayState
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.physics.abcdef import compute_pipeline_result
from abcdef_sim.physics.abcdef.action import (
    center_ray_path_to_surface_um,
    group_delay_from_path_length_fs,
    path_length_to_surface_intersection_um,
    phase_from_group_delay_rad,
)
from abcdef_sim.physics.abcdef.pulse import omega0_from_wavelength_nm
from abcdef_sim.physics.geometry.frames import LocalFrame1D
from abcdef_sim.physics.geometry.state import ChiefRayGeometryState
from abcdef_sim.physics.geometry.surfaces import SurfacePlane1D
from abcdef_sim.pipeline._assembler import SystemAssembler
from abcdef_sim.runner import _resolve_runtime_cfg

pytestmark = pytest.mark.physics


def test_center_ray_path_to_surface_reduces_to_axial_length_at_normal_incidence() -> None:
    omega = np.array([1.7, 1.8, 1.9], dtype=np.float64)
    frame = LocalFrame1D(origin_x_um=0.0, origin_z_um=0.0, axis_angle_rad=0.0)
    plane = SurfacePlane1D(point_x_um=0.0, point_z_um=100_000.0, normal_angle_rad=0.0)

    path = center_ray_path_to_surface_um(frame=frame, plane=plane)
    group_delay = group_delay_from_path_length_fs(np.full_like(omega, path), np.ones_like(omega))
    phase_geom = phase_from_group_delay_rad(
        omega,
        group_delay,
        omega0_rad_per_fs=float(np.mean(omega)),
    )

    assert path == pytest.approx(100_000.0, rel=0.0, abs=1e-12)
    expected = ((omega - float(np.mean(omega))) * 100_000.0) / 0.299792458
    np.testing.assert_allclose(phase_geom, expected, rtol=1e-12, atol=1e-12)


def test_path_length_to_surface_matches_parallel_plane_geometry() -> None:
    frame = LocalFrame1D(origin_x_um=0.0, origin_z_um=0.0, axis_angle_rad=0.0)
    plane = SurfacePlane1D(point_x_um=0.0, point_z_um=100.0, normal_angle_rad=0.0)
    x_prime = np.array([0.0, 0.05], dtype=np.float64)
    n = np.ones_like(x_prime)

    path = path_length_to_surface_intersection_um(
        x_um=np.zeros_like(x_prime),
        x_prime=x_prime,
        refractive_index=n,
        frame=frame,
        plane=plane,
    )

    slope = x_prime / n
    expected = 100.0 * np.sqrt(1.0 + slope**2)
    np.testing.assert_allclose(path, expected, rtol=1e-12, atol=1e-12)


def test_chief_ray_reflection_changes_lab_direction() -> None:
    state = ChiefRayGeometryState(point_x_um=0.0, point_z_um=0.0, axis_angle_rad=0.0)

    reflected = state.reflected_about_normal(normal_angle_rad=0.0)

    assert reflected.axis_angle_rad == pytest.approx(math.pi, rel=0.0, abs=1e-12)


def test_compute_pipeline_result_prefers_phi_geom_when_available() -> None:
    omega = np.array([2.0, 2.2], dtype=np.float64)
    contribution = PhaseContribution(
        optic_name="space",
        instance_name="gap",
        omega=omega,
        phi0_rad=np.array([10.0, 11.0], dtype=np.float64),
        phi_geom_rad=np.array([1.0, 2.0], dtype=np.float64),
        phi3_rad=np.array([0.1, 0.2], dtype=np.float64),
    )
    state = RayState(
        rays=np.array(
            [
                [[0.0], [0.0], [1.0]],
                [[0.0], [0.0], [1.0]],
            ],
            dtype=np.float64,
        ),
        system=np.repeat(np.eye(3, dtype=np.float64)[None, ...], 2, axis=0),
    )

    result = compute_pipeline_result(state, state, (contribution,))

    np.testing.assert_allclose(result.phi0_axial_total_rad, contribution.phi0_rad)
    np.testing.assert_allclose(result.phi_geom_total_rad, contribution.phi_geom_rad)
    np.testing.assert_allclose(
        result.phi_total_rad, contribution.phi_geom_rad + contribution.phi3_rad
    )


def test_treacy_assembly_attaches_surface_intersection_metadata() -> None:
    laser = LaserSpec(
        w0=omega0_from_wavelength_nm(1030.0),
        span=0.25,
        N=64,
        pulse={"center_wavelength_nm": 1030.0},
    )
    assembler = SystemAssembler(
        factory=OpticFactory.default(),
        cfg_gen=OpticStageCfgGenerator(cache=NullCacheBackend()),
    )
    resolved_cfg = _resolve_runtime_cfg(
        treacy_compressor_preset(length_to_mirror_um=50_000.0), laser
    )
    cfgs = assembler.build_optic_cfgs(resolved_cfg.to_preset(), laser)
    by_name = {cfg.instance_name: cfg for cfg in cfgs}

    assert by_name["gap_12"].action_model == "surface_intersection"
    assert by_name["gap_21"].action_model == "surface_intersection"
    assert by_name["to_fold"].action_model == "surface_intersection"
    assert by_name["from_fold"].action_model == "surface_intersection"
    assert by_name["gap_12"].transport_length_um is not None
    assert by_name["gap_12"].transport_length_um > by_name["gap_12"].length
    assert by_name["gap_12"].next_surface_normal_angle_rad == pytest.approx(
        by_name["g2"].entrance_surface_normal_angle_rad,
        rel=0.0,
        abs=1e-12,
    )
