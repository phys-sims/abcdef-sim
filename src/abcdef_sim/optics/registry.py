from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from abcdef_sim.data_models.specs import OpticKind, OpticSpec
from abcdef_sim.optics.base import Optic
from abcdef_sim.optics.frame_transform import FrameTransform
from abcdef_sim.optics.freespace import FreeSpace
from abcdef_sim.optics.grating import Grating
from abcdef_sim.optics.thick_lens import ThickLens
from abcdef_sim.physics.abcd.lenses import SellmeierMaterial

BuilderFn = Callable[[OpticSpec], Optic]


@dataclass
class OpticFactory:
    """
    Converts OpticSpec -> Optic instance.
    Centralizing this avoids scattered ad-hoc instantiation logic.
    """

    registry: dict[OpticKind, BuilderFn]

    @staticmethod
    def default() -> OpticFactory:
        def build_free_space(spec: OpticSpec) -> Optic:
            length = float(spec.params.get("L", 0.0))
            medium_refractive_index = float(spec.params.get("medium_refractive_index", 1.0))
            return FreeSpace(
                name="FreeSpace",
                instance_name=spec.instance_name,
                length=length,
                medium_refractive_index=medium_refractive_index,
            )

        def build_grating(spec: OpticSpec) -> Optic:
            return Grating(
                name="Grating",
                instance_name=spec.instance_name,
                line_density_lpmm=float(spec.params["line_density_lpmm"]),
                incidence_angle_deg=float(spec.params["incidence_angle_deg"]),
                diffraction_order=int(spec.params.get("diffraction_order", -1)),
                immersion_refractive_index=float(
                    spec.params.get("immersion_refractive_index", 1.0)
                ),
            )

        def build_frame_transform(spec: OpticSpec) -> Optic:
            return FrameTransform(
                name="FrameTransform",
                instance_name=spec.instance_name,
                x_offset_um=float(spec.params.get("x_offset_um", 0.0)),
                x_prime_offset=float(spec.params.get("x_prime_offset", 0.0)),
                x_prime_scale=float(spec.params.get("x_prime_scale", 1.0)),
            )

        def build_thick_lens(spec: OpticSpec) -> Optic:
            refractive_index_model = spec.params["refractive_index_model"]
            refractive_index: float | SellmeierMaterial
            if isinstance(refractive_index_model, dict):
                refractive_index = SellmeierMaterial(
                    name=f"{spec.instance_name}:sellmeier",
                    b_terms=tuple(float(value) for value in refractive_index_model["b_terms"]),
                    c_terms=tuple(float(value) for value in refractive_index_model["c_terms_um2"]),
                )
            else:
                refractive_index = float(refractive_index_model)

            return ThickLens(
                name="ThickLens",
                instance_name=spec.instance_name,
                _length=float(spec.params["thickness"]),
                R1=None if spec.params.get("R1") is None else float(spec.params["R1"]),
                R2=None if spec.params.get("R2") is None else float(spec.params["R2"]),
                n_in=float(spec.params.get("n_in", 1.0)),
                n_out=float(spec.params.get("n_out", 1.0)),
                refractive_index_model=refractive_index,
            )

        reg: dict[OpticKind, BuilderFn] = {
            "FreeSpace": build_free_space,
            "Grating": build_grating,
            "FrameTransform": build_frame_transform,
            "ThickLens": build_thick_lens,
        }
        return OpticFactory(registry=reg)

    def build(self, spec: OpticSpec) -> Optic:
        try:
            fn = self.registry[spec.kind]
        except KeyError as e:
            raise ValueError(f"No optic builder registered for kind={spec.kind!r}") from e
        return fn(spec)


class OpticsRegistry:
    """Minimal registry placeholder for package import scaffolding."""

    def register(self, *_args: object, **_kwargs: object) -> None:
        """Accept registrations without behavior until implementation lands."""

        return None
