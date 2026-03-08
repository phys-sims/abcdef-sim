from __future__ import annotations

import numpy as np
import pytest

from abcdef_sim import AbcdefCfg, BeamSpec, FreeSpaceCfg, PulseSpec, StandaloneLaserSpec, run_abcdef

pytestmark = pytest.mark.integration


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
