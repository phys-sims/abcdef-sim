from __future__ import annotations

from abcdef_sim.data_models.specs import OpticSpec
from abcdef_sim.optics.freespace import FreeSpace
from abcdef_sim.optics.grating import Grating
from abcdef_sim.optics.registry import OpticFactory
from abcdef_sim.optics.thick_lens import ThickLens


def test_factory_builds_freespace_for_pascal_case_kind() -> None:
    factory = OpticFactory.default()

    optic = factory.build(OpticSpec(kind="FreeSpace", instance_name="fs-p", params={"L": 1.0}))

    assert isinstance(optic, FreeSpace)


def test_factory_builds_grating_for_pascal_case_kind() -> None:
    factory = OpticFactory.default()

    optic = factory.build(
        OpticSpec(
            kind="Grating",
            instance_name="g1",
            params={
                "line_density_lpmm": 1200.0,
                "incidence_angle_deg": 35.0,
                "diffraction_order": -1,
                "immersion_refractive_index": 1.0,
            },
        )
    )

    assert isinstance(optic, Grating)


def test_factory_builds_thick_lens_for_pascal_case_kind() -> None:
    factory = OpticFactory.default()

    optic = factory.build(
        OpticSpec(
            kind="ThickLens",
            instance_name="l1",
            params={
                "R1": 100.0,
                "R2": -100.0,
                "thickness": 5.0,
                "n_in": 1.0,
                "n_out": 1.0,
                "refractive_index_model": 1.5,
            },
        )
    )

    assert isinstance(optic, ThickLens)
