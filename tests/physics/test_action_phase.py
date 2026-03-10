from __future__ import annotations

import math

import numpy as np
import pytest

from abcdef_sim import treacy_compressor_preset
from abcdef_sim.cache.backend import NullCacheBackend
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.specs import LaserSpec
from abcdef_sim.data_models.results import PhaseContribution
from abcdef_sim.data_models.states import RayState
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.physics.abcdef import compute_pipeline_result
from abcdef_sim.physics.abcdef.action import (
    free_space_path_to_planar_surface_um,
    group_delay_from_path_length_fs,
    phase_from_group_delay_rad,
)
from abcdef_sim.physics.abcdef.pulse import omega0_from_wavelength_nm
from abcdef_sim.pipeline._assembler import SystemAssembler
from abcdef_sim.runner import _resolve_runtime_cfg

pytestmark = pytest.mark.physics


def test_free_space_path_to_planar_surface_reduces_to_axial_length_at_normal_incidence() -> None:
    omega = np.array([1.7, 1.8, 1.9], dtype=np.float64)
    n = np.ones_like(omega)
    path = free_space_path_to_planar_surface_um(
        axial_length_um=100_000.0,
        x_prime=np.zeros_like(omega),
        refractive_index=n,
        surface_incidence_angle_rad=0.0,
    )
    group_delay = group_delay_from_path_length_fs(path, n)
    phase_geom = phase_from_group_delay_rad(omega, group_delay, omega0_rad_per_fs=float(np.mean(omega)))

    np.testing.assert_allclose(path, 100_000.0, rtol=0.0, atol=1e-12)
    expected = ((omega - float(np.mean(omega))) * 100_000.0) / 0.299792458
    np.testing.assert_allclose(phase_geom, expected, rtol=1e-12, atol=1e-12)


def test_free_space_path_to_planar_surface_matches_line_plane_intersection_geometry() -> None:
    x_prime = np.array([0.0, 0.05], dtype=np.float64)
    n = np.ones_like(x_prime)
    gamma = math.radians(30.0)
    path = free_space_path_to_planar_surface_um(
        axial_length_um=100.0,
        x_prime=x_prime,
        refractive_index=n,
        surface_incidence_angle_rad=gamma,
    )

    slope = x_prime / n
    expected = 100.0 * np.sqrt(1.0 + slope**2) / (math.cos(gamma) + slope * math.sin(gamma))
    np.testing.assert_allclose(path, expected, rtol=1e-12, atol=1e-12)


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
        result.phi_total_rad,
        contribution.phi_geom_rad + contribution.phi3_rad,
    )


def test_treacy_assembly_enables_phi_geom_only_for_gaps_with_next_grating_surface() -> None:
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
    resolved_cfg = _resolve_runtime_cfg(treacy_compressor_preset(length_to_mirror_um=50_000.0), laser)
    cfgs = assembler.build_optic_cfgs(
        resolved_cfg.to_preset(),
        laser,
    )
    by_name = {cfg.instance_name: cfg for cfg in cfgs}

    assert by_name["gap_12"].phi_geom_model == "free_space_to_planar_surface"
    assert by_name["gap_21"].phi_geom_model == "free_space_to_planar_surface"
    assert by_name["to_fold"].phi_geom_model == "none"
    assert by_name["from_fold"].phi_geom_model == "free_space_to_planar_surface"
    assert by_name["gap_12"].next_surface_incidence_angle_rad == pytest.approx(
        math.radians(41.484970499511796)
    )
