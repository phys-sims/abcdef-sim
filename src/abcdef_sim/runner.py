from __future__ import annotations

from typing import Literal, cast

import numpy as np

from abcdef_sim.cache.backend import NullCacheBackend
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.results import AbcdefRunResult
from abcdef_sim.data_models.specs import (
    AbcdefCfg,
    FrameTransformCfg,
    FreeSpaceCfg,
    GratingCfg,
    LaserSpec,
    SellmeierMaterialCfg,
    ThickLensCfg,
)
from abcdef_sim.data_models.standalone import LaserState, StandaloneLaserSpec
from abcdef_sim.data_models.states import PHASE_CONTRIBUTIONS_META_KEY, RayState
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.physics.abcd.gaussian import beam_radius_from_q, q_from_waist, q_propagate
from abcdef_sim.physics.abcd.lenses import (
    SellmeierMaterial,
    ThickLensSpec,
    thick_lens_matrix_for_spec,
)
from abcdef_sim.physics.abcd.matrices import compose, free_space
from abcdef_sim.physics.abcdef.dispersion import (
    fit_phase_taylor_affine_detrended,
    fod_from_phase_coeffs,
    gdd_from_phase_coeffs,
    tod_from_phase_coeffs,
)
from abcdef_sim.physics.abcdef.phase_terms import martinez_k_center, phi4_rad
from abcdef_sim.physics.abcdef.postprocess import compute_pipeline_result
from abcdef_sim.physics.abcdef.pulse import (
    apply_phase_to_state,
    build_standalone_laser_state,
    omega0_from_wavelength_nm,
    update_beam_state_from_abcd,
)
from abcdef_sim.physics.abcdef.treacy import (
    compute_grating_diffraction_angle_deg,
)
from abcdef_sim.pipeline._assembler import SystemAssembler
from abcdef_sim.pipeline.stages import AbcdefOpticStage
from abcdef_sim.presets import _build_treacy_optics


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
    resolved_cfg = _resolve_runtime_cfg(cfg, internal_laser_spec)

    assembler = SystemAssembler(
        factory=OpticFactory.default(),
        cfg_gen=OpticStageCfgGenerator(cache=NullCacheBackend()),
    )
    stages = [
        AbcdefOpticStage(cfg=stage_cfg)
        for stage_cfg in assembler.build_optic_cfgs(
            resolved_cfg.to_preset(),
            internal_laser_spec,
            policy=policy,
        )
    ]

    ray_state_in = _initial_ray_state(cfg=resolved_cfg, laser=internal_laser_spec)
    ray_state_out = ray_state_in
    for stage in stages:
        ray_state_out = stage.process(ray_state_out, policy=policy).state

    contributions = tuple(ray_state_out.meta.get(PHASE_CONTRIBUTIONS_META_KEY, ()))
    q_in, w_in, w_out, q_out = _beam_phase_inputs(
        omega=np.asarray(internal_laser_spec.omega(), dtype=np.float64),
        beam_radius_mm=float(initial_state.beam.radius_mm),
        m2=float(initial_state.beam.m2),
        final_system=np.asarray(ray_state_out.system, dtype=np.float64),
    )
    phi4 = None
    if _is_treacy_preset(resolved_cfg):
        phi4 = phi4_rad(
            martinez_k_center(np.asarray(internal_laser_spec.omega(), dtype=np.float64)),
            0.0,
            ray_state_out.rays[:, 0, 0],
            q_out,
        )
    pipeline_result = compute_pipeline_result(
        ray_state_in,
        ray_state_out,
        contributions,
        q_in=q_in,
        w_in=w_in,
        w_out=w_out,
        phi4_rad=phi4,
    )
    fit = fit_phase_taylor_affine_detrended(
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
    center_idx = int(
        np.argmin(np.abs(np.asarray(internal_laser_spec.delta_omega(), dtype=np.float64)))
    )
    beam_matrix = np.asarray(ray_state_out.system[center_idx, :2, :2], dtype=np.float64)
    final_state = update_beam_state_from_abcd(pulse_state_out, beam_matrix)

    final_state.meta["abcdef"] = {
        "omega0_rad_per_fs": float(pipeline_result.omega0_rad_per_fs),
        "delta_omega_rad_per_fs": delta_omega.tolist(),
        "phi_total_rad": pipeline_result.phi_total_rad.tolist(),
        "phi_fit_rad": fit.phi_fit_rad.tolist(),
        "fit_residual_rad": fit.residual_rad.tolist(),
        "phi0_axial_total_rad": np.asarray(
            pipeline_result.phi0_axial_total_rad,
            dtype=np.float64,
        ).tolist(),
        "phi_geom_total_rad": None
        if pipeline_result.phi_geom_total_rad is None
        else np.asarray(pipeline_result.phi_geom_total_rad, dtype=np.float64).tolist(),
        "phi3_transport_like_total_rad": None
        if pipeline_result.phi3_transport_like_total_rad is None
        else np.asarray(
            pipeline_result.phi3_transport_like_total_rad,
            dtype=np.float64,
        ).tolist(),
        "phi3_phase_total_rad": None
        if pipeline_result.phi3_phase_total_rad is None
        else np.asarray(pipeline_result.phi3_phase_total_rad, dtype=np.float64).tolist(),
        "phi3_total_rad": np.asarray(pipeline_result.phi3_total_rad, dtype=np.float64).tolist(),
        "phi1_rad": None
        if pipeline_result.phi1_rad is None
        else np.asarray(pipeline_result.phi1_rad, dtype=np.float64).tolist(),
        "phi2_rad": None
        if pipeline_result.phi2_rad is None
        else np.asarray(pipeline_result.phi2_rad, dtype=np.float64).tolist(),
        "phi4_rad": None
        if pipeline_result.phi4_rad is None
        else np.asarray(pipeline_result.phi4_rad, dtype=np.float64).tolist(),
        "coefficients_rad": list(coefficients),
        "weighted_rms_rad": float(fit.weighted_rms_rad),
        "max_abs_residual_rad": float(fit.max_abs_residual_rad),
        "x_out_um": np.asarray(ray_state_out.rays[:, 0, 0], dtype=np.float64).tolist(),
        "x_prime_out": np.asarray(ray_state_out.rays[:, 1, 0], dtype=np.float64).tolist(),
        "w_out_um": np.asarray(w_out, dtype=np.float64).tolist(),
        "q_out_real_um": np.asarray(np.real(q_out), dtype=np.float64).tolist(),
        "q_out_imag_um": np.asarray(np.imag(q_out), dtype=np.float64).tolist(),
        "spectral_power_au": np.asarray(spectral_weights, dtype=np.float64).tolist(),
        "per_optic": [_phase_contribution_payload(contribution) for contribution in contributions],
    }
    if _is_treacy_preset(resolved_cfg):
        resolved_diffraction_angle_deg = _resolved_treacy_diffraction_angle_deg(
            resolved_cfg,
            center_wavelength_nm=float(internal_laser_spec.pulse["center_wavelength_nm"]),
        )
        final_state.meta["abcdef"]["treacy_resolved_diffraction_angle_deg"] = (
            resolved_diffraction_angle_deg
        )
        final_state.meta["abcdef"]["treacy_resolved_optics"] = [
            {
                "instance_name": optic.instance_name,
                "kind": optic.kind,
                **(
                    {"incidence_angle_deg": float(optic.incidence_angle_deg)}
                    if isinstance(optic, GratingCfg)
                    else {}
                ),
                **(
                    {"x_prime_scale": float(optic.x_prime_scale)}
                    if isinstance(optic, FrameTransformCfg)
                    else {}
                ),
            }
            for optic in resolved_cfg.optics
        ]
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


def _beam_phase_inputs(
    *,
    omega: np.ndarray,
    beam_radius_mm: float,
    m2: float,
    final_system: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the per-frequency Gaussian-beam inputs required by Martinez eq. 25."""

    omega_arr = np.asarray(omega, dtype=np.float64).reshape(-1)
    system_arr = np.asarray(final_system, dtype=np.float64)
    if system_arr.shape != (omega_arr.size, 3, 3):
        raise ValueError(
            "final_system must have shape (N,3,3) matching omega: "
            f"expected {(omega_arr.size, 3, 3)}, got {system_arr.shape}"
        )

    beam_radius_um = float(beam_radius_mm) * 1e3
    wavelength_um = (2.0 * np.pi * 0.299792458) / omega_arr
    effective_wavelength_um = wavelength_um * float(m2)

    q_in = np.empty(omega_arr.size, dtype=np.complex128)
    q_out = np.empty(omega_arr.size, dtype=np.complex128)
    w_out = np.empty(omega_arr.size, dtype=np.float64)
    for idx, wavelength_i in enumerate(effective_wavelength_um):
        q_in_i = q_from_waist(beam_radius_um, float(wavelength_i))
        q_in[idx] = q_in_i
        q_out_i = q_propagate(q_in_i, system_arr[idx, :2, :2])
        q_out[idx] = q_out_i
        w_out[idx] = beam_radius_from_q(q_out_i, float(wavelength_i))

    w_in = np.full(omega_arr.size, beam_radius_um, dtype=np.float64)
    return q_in, w_in, w_out, q_out


def _is_treacy_preset(cfg: AbcdefCfg) -> bool:
    return str(cfg.tags.get("preset_kind", "")) == "treacy_compressor"


def _resolve_runtime_cfg(cfg: AbcdefCfg, laser: LaserSpec) -> AbcdefCfg:
    if not _is_treacy_preset(cfg):
        return cfg

    n_passes = int(cfg.tags["n_passes"])
    if n_passes not in (1, 2):
        raise ValueError(f"Unsupported Treacy n_passes={n_passes}; expected 1 or 2.")

    resolved_optics = _build_treacy_optics(
        line_density_lpmm=float(cfg.tags["line_density_lpmm"]),
        incidence_angle_deg=float(cfg.tags["incidence_angle_deg"]),
        separation_um=float(cfg.tags["separation_um"]),
        length_to_mirror_um=float(cfg.tags["length_to_mirror_um"]),
        diffraction_order=int(cfg.tags["diffraction_order"]),
        n_passes=cast(Literal[1, 2], n_passes),
        immersion_refractive_index=_treacy_immersion_refractive_index(cfg),
        gap_medium_refractive_index=_treacy_gap_medium_refractive_index(cfg),
        center_wavelength_nm=float(laser.pulse["center_wavelength_nm"]),
    )
    return cfg.model_copy(update={"optics": resolved_optics})


def _treacy_immersion_refractive_index(cfg: AbcdefCfg) -> float:
    for optic in cfg.optics:
        if isinstance(optic, GratingCfg):
            return float(optic.immersion_refractive_index)
    return 1.0


def _treacy_gap_medium_refractive_index(cfg: AbcdefCfg) -> float:
    for optic in cfg.optics:
        if isinstance(optic, FreeSpaceCfg) and optic.instance_name == "gap_12":
            return float(optic.medium_refractive_index)
    return 1.0


def _resolved_treacy_diffraction_angle_deg(cfg: AbcdefCfg, *, center_wavelength_nm: float) -> float:
    return compute_grating_diffraction_angle_deg(
        line_density_lpmm=float(cfg.tags["line_density_lpmm"]),
        incidence_angle_deg=float(cfg.tags["incidence_angle_deg"]),
        wavelength_nm=float(center_wavelength_nm),
        diffraction_order=int(cfg.tags["diffraction_order"]),
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
        elif isinstance(optic, FrameTransformCfg):
            matrices.append(np.eye(2, dtype=np.float64))
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
        "phi_geom_rad": _maybe_list(contribution.phi_geom_rad),
        "phi3_transport_like_rad": _maybe_list(contribution.phi3_transport_like_rad),
        "phi3_phase_rad": _maybe_list(contribution.phi3_phase_rad),
        "phi3_rad": np.asarray(contribution.phi3_rad, dtype=np.float64).reshape(-1).tolist(),
        "path_length_um": _maybe_list(contribution.path_length_um),
        "group_delay_fs": _maybe_list(contribution.group_delay_fs),
        "filter_amp": _maybe_list(contribution.filter_amp),
        "filter_phase_rad": _maybe_list(contribution.filter_phase_rad),
    }
