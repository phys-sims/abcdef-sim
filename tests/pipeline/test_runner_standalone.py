from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim import (
    AbcdefCfg,
    BeamSpec,
    FreeSpaceCfg,
    InputRayCfg,
    PulseSpec,
    SellmeierMaterialCfg,
    StandaloneLaserSpec,
    ThickLensCfg,
    run_abcdef,
)
from abcdef_sim.physics.abcd.lenses import (
    SellmeierMaterial,
    ThickLensSpec,
    thick_lens_matrix_for_spec,
)
from abcdef_sim.physics.abcd.matrices import compose, free_space
from abcdef_sim.physics.abcdef.conventions import theta_abcd_to_xprime_abcdef
from abcdef_sim.physics.abcdef.pulse import (
    build_standalone_laser_state,
    update_beam_state_from_abcd,
)

pytestmark = pytest.mark.integration

_C_UM_PER_FS = 0.299792458


def _omega_to_wavelength_um(omega: float) -> float:
    return (2.0 * np.pi * _C_UM_PER_FS) / float(omega)


def test_run_abcdef_returns_fit_phase_and_mutated_beam_state() -> None:
    cfg = AbcdefCfg(optics=(FreeSpaceCfg(instance_name="fs1", length=50_000.0),))
    laser_spec = StandaloneLaserSpec(
        pulse=PulseSpec(width_fs=100.0, center_wavelength_nm=1030.0, n_samples=128),
        beam=BeamSpec(radius_mm=1.0, m2=1.2),
    )

    result = run_abcdef(cfg, laser_spec)

    assert result.fit.coefficients_rad.shape == (5,)
    assert result.pipeline_result.phi_total_rad.shape == result.fit.phi_fit_rad.shape
    np.testing.assert_allclose(
        np.abs(result.final_state.pulse.field_w),
        np.abs(result.initial_state.pulse.field_w),
        rtol=1e-9,
        atol=1e-9,
    )
    assert result.final_state.beam.radius_mm > result.initial_state.beam.radius_mm
    assert result.final_state.meta["abcdef"]["per_optic"][0]["instance_name"] == "fs1"


def test_run_abcdef_matches_abcd_for_free_space_thick_lens_chain() -> None:
    lens2_material_cfg = SellmeierMaterialCfg(
        b_terms=(1.0, 0.25, 0.12),
        c_terms_um2=(0.01, 0.05, 140.0),
    )
    lens2_material = SellmeierMaterial(
        name="lens-2-demo",
        b_terms=lens2_material_cfg.b_terms,
        c_terms=lens2_material_cfg.c_terms_um2,
    )
    cfg = AbcdefCfg(
        optics=(
            FreeSpaceCfg(instance_name="fs1", length=120.0),
            ThickLensCfg(
                instance_name="lens1",
                R1=90.0,
                R2=-75.0,
                thickness=6.0,
                refractive_index=1.49,
            ),
            FreeSpaceCfg(instance_name="fs2", length=80.0),
            ThickLensCfg(
                instance_name="lens2",
                R1=65.0,
                R2=-110.0,
                thickness=4.5,
                refractive_index=lens2_material_cfg,
            ),
        ),
        input_ray=InputRayCfg(x=0.4, x_prime=0.012),
    )
    laser_spec = StandaloneLaserSpec(
        pulse=PulseSpec(width_fs=90.0, center_wavelength_nm=1030.0, n_samples=64),
        beam=BeamSpec(radius_mm=0.9, m2=1.1),
    )

    result = run_abcdef(cfg, laser_spec)
    omega = result.pipeline_result.omega
    actual_system = result.pipeline_result.final_state.system
    actual_rays = result.pipeline_result.final_state.rays[:, :, 0]

    expected_system = []
    expected_rays = []
    input_theta_ray = np.array([cfg.input_ray.x, cfg.input_ray.x_prime], dtype=float)
    for omega_i in omega:
        wavelength_um = _omega_to_wavelength_um(float(omega_i))
        theta_system = compose(
            free_space(120.0),
            thick_lens_matrix_for_spec(
                ThickLensSpec(
                    refractive_index=1.49,
                    R1=90.0,
                    R2=-75.0,
                    thickness=6.0,
                )
            ),
            free_space(80.0),
            thick_lens_matrix_for_spec(
                ThickLensSpec(
                    refractive_index=lens2_material,
                    R1=65.0,
                    R2=-110.0,
                    thickness=4.5,
                ),
                wavelength=wavelength_um,
            ),
        )
        expected_system.append(theta_abcd_to_xprime_abcdef(theta_system, n_in=1.0, n_out=1.0))
        theta_ray_out = theta_system @ input_theta_ray
        expected_rays.append([theta_ray_out[0], theta_ray_out[1], 1.0])

    np.testing.assert_allclose(actual_system, np.stack(expected_system), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(actual_rays, np.asarray(expected_rays), rtol=1e-12, atol=1e-12)
    assert [item["instance_name"] for item in result.final_state.meta["abcdef"]["per_optic"]] == [
        "fs1",
        "lens1",
        "fs2",
        "lens2",
    ]


@pytest.mark.physics
def test_run_abcdef_beam_update_respects_free_space_medium_index() -> None:
    cfg = AbcdefCfg(
        optics=(
            FreeSpaceCfg(
                instance_name="fs-medium",
                length=50_000.0,
                medium_refractive_index=1.5,
            ),
        )
    )
    laser_spec = StandaloneLaserSpec(
        pulse=PulseSpec(width_fs=100.0, center_wavelength_nm=1030.0, n_samples=128),
        beam=BeamSpec(radius_mm=1.0, m2=1.2),
    )

    result = run_abcdef(cfg, laser_spec)
    expected_final_state = update_beam_state_from_abcd(
        build_standalone_laser_state(laser_spec),
        free_space(50_000.0 / 1.5),
    )

    assert result.final_state.beam.radius_mm == pytest.approx(
        expected_final_state.beam.radius_mm,
        rel=1e-12,
        abs=1e-12,
    )
