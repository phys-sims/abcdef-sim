from __future__ import annotations

from abcdef_sim import FreeSpaceCfg, GratingCfg, treacy_compressor_preset


def test_treacy_preset_defaults_to_double_pass_with_explicit_return_leg() -> None:
    cfg = treacy_compressor_preset()

    assert [optic.instance_name for optic in cfg.optics] == [
        "g1",
        "gap_12",
        "g2",
        "mirror_leg_round_trip",
        "g2_return",
        "gap_21",
        "g1_return",
    ]
    assert isinstance(cfg.optics[0], GratingCfg)
    assert isinstance(cfg.optics[3], FreeSpaceCfg)
    assert cfg.optics[3].length == 0.0
    assert cfg.tags["preset_kind"] == "treacy_compressor"
    assert cfg.tags["n_passes"] == 2


def test_treacy_preset_folds_one_way_mirror_length_into_round_trip_segment() -> None:
    cfg = treacy_compressor_preset(length_to_mirror_um=37_500.0)

    mirror_leg = next(
        optic for optic in cfg.optics if optic.instance_name == "mirror_leg_round_trip"
    )
    assert isinstance(mirror_leg, FreeSpaceCfg)
    assert mirror_leg.length == 75_000.0


def test_treacy_preset_single_pass_ignores_mirror_leg_parameter() -> None:
    cfg = treacy_compressor_preset(n_passes=1, length_to_mirror_um=50_000.0)

    assert [optic.instance_name for optic in cfg.optics] == ["g1", "gap_12", "g2"]
    assert all(optic.instance_name != "mirror_leg_round_trip" for optic in cfg.optics)


def test_treacy_preset_instance_names_are_deterministic() -> None:
    cfg_a = treacy_compressor_preset(length_to_mirror_um=12_345.0)
    cfg_b = treacy_compressor_preset(length_to_mirror_um=12_345.0)

    assert tuple(optic.instance_name for optic in cfg_a.optics) == tuple(
        optic.instance_name for optic in cfg_b.optics
    )
