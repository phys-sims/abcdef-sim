from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from abcdef_sim._phys_pipeline import PolicyBag, SequentialPipeline
from abcdef_sim.cfg_generator import OpticStageCfgGenerator
from abcdef_sim.data_models.configs import OpticStageCfg
from abcdef_sim.data_models.specs import LaserSpec, OpticSpec, SystemPreset
from abcdef_sim.data_models.stages import AbcdefOpticStage
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.physics.abcdef.action import center_ray_path_to_surface_um
from abcdef_sim.physics.abcdef.treacy import compute_grating_diffraction_angle_deg
from abcdef_sim.physics.geometry.intersection import point_from_local_coordinates_um
from abcdef_sim.physics.geometry.state import ChiefRayGeometryState
from abcdef_sim.physics.geometry.surfaces import SurfacePlane1D


@dataclass
class SystemAssembler:
    """
    Pure assembly layer.
    Turns (SystemPreset, LaserSpec) into a list of stages (each has cfg).
    """

    factory: OpticFactory
    cfg_gen: OpticStageCfgGenerator

    def build_optic_cfgs(
        self,
        preset: SystemPreset,
        laser: LaserSpec,
        *,
        policy: PolicyBag | None = None,
    ) -> list[OpticStageCfg]:
        w = laser.omega()
        delta_w = laser.delta_omega()
        omega0 = laser.omega0_rad_per_fs

        cfgs: list[OpticStageCfg] = []
        specs = list(preset.optics)
        for spec in specs:
            optic = self.factory.build(spec)
            cfg = self.cfg_gen.build(
                optic,
                w,
                delta_omega=delta_w,
                omega0_rad_per_fs=omega0,
                tags={"preset": preset.name, "optic_kind": spec.kind},
                policy=policy,
            )
            cfgs.append(cfg)
        return _attach_geometry_metadata(cfgs, specs, laser)

    def build_pipeline(
        self,
        preset: SystemPreset,
        laser: LaserSpec,
        *,
        policy: PolicyBag | None = None,
        pipeline_name: str | None = None,
    ) -> Any:
        cfgs = self.build_optic_cfgs(preset, laser, policy=policy)
        stages = [AbcdefOpticStage(cfg=c) for c in cfgs]
        return SequentialPipeline(stages=stages, name=pipeline_name or preset.name)


def _attach_geometry_metadata(
    cfgs: list[OpticStageCfg],
    specs: list[OpticSpec],
    laser: LaserSpec,
) -> list[OpticStageCfg]:
    if len(cfgs) != len(specs):
        raise ValueError("cfgs and specs must have matching lengths")

    center_wavelength_nm = float(laser.pulse.get("center_wavelength_nm", 1030.0))
    updated_cfgs: list[OpticStageCfg] = []
    geometry_state = ChiefRayGeometryState(point_x_um=0.0, point_z_um=0.0, axis_angle_rad=0.0)

    for idx, (cfg, spec) in enumerate(zip(cfgs, specs, strict=True)):
        if spec.kind == "Grating":
            plane = _grating_surface_plane(geometry_state, spec)
            updated_cfgs.append(
                cfg.model_copy(
                    update={
                        "local_axis_angle_rad": float(geometry_state.axis_angle_rad),
                        "entrance_surface_point_x_um": float(plane.point_x_um),
                        "entrance_surface_point_z_um": float(plane.point_z_um),
                        "entrance_surface_normal_angle_rad": float(plane.normal_angle_rad),
                    }
                )
            )
            geometry_state = _propagate_geometry_through_grating(
                geometry_state,
                spec,
                center_wavelength_nm=center_wavelength_nm,
            )
            continue

        if spec.kind == "FrameTransform":
            role = str(spec.params.get("geometry_role", "basis_only"))
            if role == "reflect_chief_ray":
                plane = SurfacePlane1D(
                    point_x_um=float(geometry_state.point_x_um),
                    point_z_um=float(geometry_state.point_z_um),
                    normal_angle_rad=float(geometry_state.axis_angle_rad),
                )
                updated_cfgs.append(
                    cfg.model_copy(
                        update={
                            "local_axis_angle_rad": float(geometry_state.axis_angle_rad),
                            "entrance_surface_point_x_um": float(plane.point_x_um),
                            "entrance_surface_point_z_um": float(plane.point_z_um),
                            "entrance_surface_normal_angle_rad": float(plane.normal_angle_rad),
                        }
                    )
                )
                geometry_state = geometry_state.reflected_about_normal(
                    normal_angle_rad=float(plane.normal_angle_rad)
                )
            else:
                updated_cfgs.append(
                    cfg.model_copy(
                        update={"local_axis_angle_rad": float(geometry_state.axis_angle_rad)}
                    )
                )
            continue

        if spec.kind != "FreeSpace":
            updated_cfgs.append(
                cfg.model_copy(
                    update={"local_axis_angle_rad": float(geometry_state.axis_angle_rad)}
                )
            )
            continue

        next_surface_normal_angle_rad = _next_surface_normal_angle_rad(
            geometry_state,
            specs,
            start=idx + 1,
        )
        if next_surface_normal_angle_rad is None:
            transport_length_um = float(cfg.length)
            updated_cfgs.append(
                cfg.model_copy(
                    update={
                        "transport_length_um": transport_length_um,
                        "local_axis_angle_rad": float(geometry_state.axis_angle_rad),
                    }
                )
            )
            geometry_state = _advance_center_ray(geometry_state, transport_length_um)
            continue

        next_surface = _place_next_surface(
            geometry_state,
            normal_angle_rad=next_surface_normal_angle_rad,
            physical_length_um=float(cfg.length),
            geometry_mode=str(spec.params.get("geometry_mode", "path_length")),
        )
        transport_length_um = center_ray_path_to_surface_um(
            frame=geometry_state.frame(),
            plane=next_surface,
        )
        updated_cfgs.append(
            _update_free_space_cfg_with_geometry(
                cfg,
                geometry_state=geometry_state,
                next_surface=next_surface,
                transport_length_um=transport_length_um,
            )
        )
        geometry_state = ChiefRayGeometryState(
            point_x_um=float(next_surface.point_x_um),
            point_z_um=float(next_surface.point_z_um),
            axis_angle_rad=float(geometry_state.axis_angle_rad),
        )

    return updated_cfgs


def _update_free_space_cfg_with_geometry(
    cfg: OpticStageCfg,
    *,
    geometry_state: ChiefRayGeometryState,
    next_surface: SurfacePlane1D,
    transport_length_um: float,
) -> OpticStageCfg:
    matrices = np.asarray(cfg.abcdef, dtype=np.float64).copy()
    refractive_index = np.asarray(cfg.refractive_index, dtype=np.float64).reshape(-1)
    matrices[:, 0, 1] = float(transport_length_um) / refractive_index
    return cfg.model_copy(
        update={
            "transport_length_um": float(transport_length_um),
            "abcdef": matrices,
            "action_model": "surface_intersection",
            "local_axis_angle_rad": float(geometry_state.axis_angle_rad),
            "entrance_surface_point_x_um": float(geometry_state.point_x_um),
            "entrance_surface_point_z_um": float(geometry_state.point_z_um),
            "entrance_surface_normal_angle_rad": float(geometry_state.axis_angle_rad),
            "next_surface_point_x_um": float(next_surface.point_x_um),
            "next_surface_point_z_um": float(next_surface.point_z_um),
            "next_surface_normal_angle_rad": float(next_surface.normal_angle_rad),
        }
    )


def _grating_surface_plane(
    geometry_state: ChiefRayGeometryState,
    spec: OpticSpec,
) -> SurfacePlane1D:
    return SurfacePlane1D(
        point_x_um=float(geometry_state.point_x_um),
        point_z_um=float(geometry_state.point_z_um),
        normal_angle_rad=float(geometry_state.axis_angle_rad)
        + math.radians(float(spec.params["incidence_angle_deg"])),
    )


def _propagate_geometry_through_grating(
    geometry_state: ChiefRayGeometryState,
    spec: OpticSpec,
    *,
    center_wavelength_nm: float,
) -> ChiefRayGeometryState:
    surface = _grating_surface_plane(geometry_state, spec)
    diffraction_angle_deg = compute_grating_diffraction_angle_deg(
        line_density_lpmm=float(spec.params["line_density_lpmm"]),
        incidence_angle_deg=float(spec.params["incidence_angle_deg"]),
        wavelength_nm=float(center_wavelength_nm),
        diffraction_order=int(spec.params.get("diffraction_order", -1)),
    )
    outgoing_axis_angle = float(surface.normal_angle_rad) - math.radians(diffraction_angle_deg)
    return ChiefRayGeometryState(
        point_x_um=float(geometry_state.point_x_um),
        point_z_um=float(geometry_state.point_z_um),
        axis_angle_rad=math.atan2(math.sin(outgoing_axis_angle), math.cos(outgoing_axis_angle)),
    )


def _next_surface_normal_angle_rad(
    geometry_state: ChiefRayGeometryState,
    specs: list[OpticSpec],
    *,
    start: int,
) -> float | None:
    for spec in specs[start:]:
        if spec.kind == "FrameTransform":
            role = str(spec.params.get("geometry_role", "basis_only"))
            if role == "basis_only":
                continue
            if role == "reflect_chief_ray":
                return float(geometry_state.axis_angle_rad)
            return None
        if spec.kind == "Grating":
            return float(geometry_state.axis_angle_rad) + math.radians(
                float(spec.params["incidence_angle_deg"])
            )
        if spec.kind == "ThickLens":
            return float(geometry_state.axis_angle_rad)
        return None
    return None


def _place_next_surface(
    geometry_state: ChiefRayGeometryState,
    *,
    normal_angle_rad: float,
    physical_length_um: float,
    geometry_mode: str,
) -> SurfacePlane1D:
    if geometry_mode == "normal_spacing":
        point_x, point_z = point_from_local_coordinates_um(
            geometry_state.frame(),
            x_um=0.0,
            z_um=0.0,
        )
        normal_x = math.sin(float(normal_angle_rad))
        normal_z = math.cos(float(normal_angle_rad))
        return SurfacePlane1D(
            point_x_um=float(point_x) + (float(physical_length_um) * normal_x),
            point_z_um=float(point_z) + (float(physical_length_um) * normal_z),
            normal_angle_rad=float(normal_angle_rad),
        )
    if geometry_mode == "path_length":
        point_x, point_z = point_from_local_coordinates_um(
            geometry_state.frame(),
            x_um=0.0,
            z_um=float(physical_length_um),
        )
        return SurfacePlane1D(
            point_x_um=float(point_x),
            point_z_um=float(point_z),
            normal_angle_rad=float(normal_angle_rad),
        )
    raise ValueError(f"Unsupported free-space geometry_mode={geometry_mode!r}")


def _advance_center_ray(
    geometry_state: ChiefRayGeometryState,
    path_length_um: float,
) -> ChiefRayGeometryState:
    point_x, point_z = point_from_local_coordinates_um(
        geometry_state.frame(),
        x_um=0.0,
        z_um=float(path_length_um),
    )
    return ChiefRayGeometryState(
        point_x_um=float(point_x),
        point_z_um=float(point_z),
        axis_angle_rad=float(geometry_state.axis_angle_rad),
    )
