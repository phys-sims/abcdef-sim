from __future__ import annotations

import numpy as np

from abcdef_sim.cache.backend import NullCacheBackend
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.results import AbcdefRunResult
from abcdef_sim.data_models.specs import (
    AbcdefCfg,
    FreeSpaceCfg,
    GratingCfg,
    LaserSpec,
    SellmeierMaterialCfg,
    ThickLensCfg,
)
from abcdef_sim.data_models.standalone import LaserState, StandaloneLaserSpec
from abcdef_sim.data_models.states import PHASE_CONTRIBUTIONS_META_KEY, RayState
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.physics.abcd.lenses import (
    SellmeierMaterial,
    ThickLensSpec,
    thick_lens_matrix_for_spec,
)
from abcdef_sim.physics.abcd.matrices import compose, free_space
from abcdef_sim.physics.abcdef.dispersion import (
    fit_phase_taylor,
    fod_from_phase_coeffs,
    gdd_from_phase_coeffs,
    tod_from_phase_coeffs,
)
from abcdef_sim.physics.abcdef.postprocess import compute_pipeline_result
from abcdef_sim.physics.abcdef.pulse import (
    apply_phase_to_state,
    build_standalone_laser_state,
    omega0_from_wavelength_nm,
    update_beam_state_from_abcd,
)
from abcdef_sim.pipeline._assembler import SystemAssembler
from abcdef_sim.pipeline.stages import AbcdefOpticStage


def run_abcdef(
    cfg: AbcdefCfg,
    laser_spec: StandaloneLaserSpec,
    *,
    policy: object | None = None,
) -> AbcdefRunResult:
    initial_state = build_standalone_laser_state(laser_spec)
    return run_abcdef_on_state(cfg, initial_state, policy=policy)


def run_abcdef_on_state(
    cfg: AbcdefCfg,
    state: LaserState,
    *,
    policy: object | None = None,
) -> AbcdefRunResult:
    initial_state = state.deepcopy()
    spectral_weights = np.abs(np.asarray(initial_state.pulse.field_w, dtype=np.complex128)) ** 2
    internal_laser_spec = _laser_spec_from_state(initial_state)

    assembler = SystemAssembler(
        factory=OpticFactory.default(),
        cfg_gen=OpticStageCfgGenerator(cache=NullCacheBackend()),
    )
    stages = [
        AbcdefOpticStage(cfg=stage_cfg)
        for stage_cfg in assembler.build_optic_cfgs(
            cfg.to_preset(),
            internal_laser_spec,
            policy=policy,
        )
    ]

    ray_state_in = _initial_ray_state(cfg=cfg, laser=internal_laser_spec)
    ray_state_out = ray_state_in
    for stage in stages:
        ray_state_out = stage.process(ray_state_out, policy=policy).state

    contributions = tuple(ray_state_out.meta.get(PHASE_CONTRIBUTIONS_META_KEY, ()))
    pipeline_result = compute_pipeline_result(ray_state_in, ray_state_out, contributions)
    fit = fit_phase_taylor(
        pipeline_result.delta_omega_rad_per_fs,
        pipeline_result.phi_total_rad,
        omega0_rad_per_fs=pipeline_result.omega0_rad_per_fs,
        weights=spectral_weights,
        order=4,
    )
    delta_omega = np.asarray(pipeline_result.delta_omega_rad_per_fs, dtype=np.float64).reshape(-1)
    coefficients = tuple(
        float(value) for value in np.asarray(fit.coefficients_rad, dtype=np.float64)
    )

    pulse_state_out = apply_phase_to_state(initial_state, fit.phi_fit_rad)
    beam_matrix = _beam_matrix_for_cfg(cfg, omega0_rad_per_fs=internal_laser_spec.omega0_rad_per_fs)
    final_state = update_beam_state_from_abcd(pulse_state_out, beam_matrix)

    final_state.meta["abcdef"] = {
        "omega0_rad_per_fs": float(pipeline_result.omega0_rad_per_fs),
        "delta_omega_rad_per_fs": delta_omega.tolist(),
        "phi_total_rad": pipeline_result.phi_total_rad.tolist(),
        "phi_fit_rad": fit.phi_fit_rad.tolist(),
        "fit_residual_rad": fit.residual_rad.tolist(),
        "coefficients_rad": list(coefficients),
        "weighted_rms_rad": float(fit.weighted_rms_rad),
        "max_abs_residual_rad": float(fit.max_abs_residual_rad),
        "per_optic": [_phase_contribution_payload(contribution) for contribution in contributions],
    }
    final_state.metrics.update(
        {
            "abcdef.fit_weighted_rms_rad": float(fit.weighted_rms_rad),
            "abcdef.fit_max_abs_residual_rad": float(fit.max_abs_residual_rad),
            "abcdef.gdd_fs2": gdd_from_phase_coeffs(coefficients),
            "abcdef.tod_fs3": tod_from_phase_coeffs(coefficients),
            "abcdef.fod_fs4": fod_from_phase_coeffs(coefficients),
            "abcdef.beam_radius_out_mm": float(final_state.beam.radius_mm),
        }
    )

    return AbcdefRunResult(
        initial_state=initial_state,
        final_state=final_state,
        pipeline_result=pipeline_result,
        fit=fit,
    )


def _laser_spec_from_state(state: LaserState) -> LaserSpec:
    delta_omega = np.asarray(state.pulse.grid.w, dtype=np.float64).reshape(-1)
    omega0 = omega0_from_wavelength_nm(state.pulse.grid.center_wavelength_nm)
    span = float(0.5 * (np.max(delta_omega) - np.min(delta_omega)))
    return LaserSpec(
        w0=omega0,
        span=span,
        N=int(delta_omega.size),
        delta_omega_rad_per_fs=tuple(float(value) for value in delta_omega),
        pulse={"center_wavelength_nm": float(state.pulse.grid.center_wavelength_nm)},
        beam={"radius_mm": float(state.beam.radius_mm), "m2": float(state.beam.m2)},
    )


def _initial_ray_state(cfg: AbcdefCfg, laser: LaserSpec) -> RayState:
    batch_size = int(laser.N)
    return RayState(
        rays=cfg.input_ray.to_column(batch_size=batch_size),
        system=np.repeat(np.eye(3, dtype=np.float64)[None, ...], batch_size, axis=0),
        meta={},
    )


def _beam_matrix_for_cfg(cfg: AbcdefCfg, *, omega0_rad_per_fs: float) -> np.ndarray:
    wavelength_um = (2.0 * np.pi * 0.299792458) / float(omega0_rad_per_fs)
    matrices: list[np.ndarray] = []
    for optic in cfg.optics:
        if isinstance(optic, FreeSpaceCfg):
            # Gaussian q propagation through a uniform medium uses B = L / n.
            matrices.append(free_space(float(optic.length) / float(optic.medium_refractive_index)))
        elif isinstance(optic, GratingCfg):
            from abcdef_sim.optics.grating import Grating

            matrix = Grating(
                line_density_lpmm=optic.line_density_lpmm,
                incidence_angle_deg=optic.incidence_angle_deg,
                diffraction_order=optic.diffraction_order,
                immersion_refractive_index=optic.immersion_refractive_index,
            ).matrix(np.array([omega0_rad_per_fs], dtype=np.float64), omega0=omega0_rad_per_fs)[0]
            matrices.append(matrix[:2, :2])
        else:
            assert isinstance(optic, ThickLensCfg)
            matrices.append(_thick_lens_abcd(optic, wavelength_um=wavelength_um))
    return compose(*matrices)


def _thick_lens_abcd(optic: ThickLensCfg, *, wavelength_um: float) -> np.ndarray:
    refractive_index: float | SellmeierMaterial
    if isinstance(optic.refractive_index, SellmeierMaterialCfg):
        refractive_index = SellmeierMaterial(
            name=f"{optic.instance_name}:sellmeier",
            b_terms=tuple(optic.refractive_index.b_terms),
            c_terms=tuple(optic.refractive_index.c_terms_um2),
        )
    else:
        refractive_index = float(optic.refractive_index)

    return thick_lens_matrix_for_spec(
        ThickLensSpec(
            refractive_index=refractive_index,
            R1=optic.R1,
            R2=optic.R2,
            thickness=optic.thickness,
            n_in=optic.n_in,
            n_out=optic.n_out,
        ),
        wavelength=wavelength_um,
    )


def _phase_contribution_payload(contribution: object) -> dict[str, object]:
    from abcdef_sim.data_models.results import PhaseContribution

    if not isinstance(contribution, PhaseContribution):
        raise TypeError(
            "phase contribution payload generation requires PhaseContribution values; "
            f"got {type(contribution).__name__}"
        )

    def _maybe_list(value: object) -> list[float] | None:
        if value is None:
            return None
        return np.asarray(value, dtype=np.float64).reshape(-1).tolist()

    return {
        "optic_name": contribution.optic_name,
        "instance_name": contribution.instance_name,
        "backend_id": contribution.backend_id,
        "omega": np.asarray(contribution.omega, dtype=np.float64).reshape(-1).tolist(),
        "delta_omega_rad_per_fs": _maybe_list(contribution.delta_omega_rad_per_fs),
        "omega0_rad_per_fs": float(contribution.omega0_rad_per_fs),
        "phi0_rad": np.asarray(contribution.phi0_rad, dtype=np.float64).reshape(-1).tolist(),
        "phi3_rad": np.asarray(contribution.phi3_rad, dtype=np.float64).reshape(-1).tolist(),
        "filter_amp": _maybe_list(contribution.filter_amp),
        "filter_phase_rad": _maybe_list(contribution.filter_phase_rad),
    }
