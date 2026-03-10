from __future__ import annotations

from abcdef_sim import FrameTransformCfg, FreeSpaceCfg, GratingCfg, treacy_compressor_preset


def test_treacy_preset_defaults_to_double_pass_with_explicit_fold_geometry() -> None:
    cfg = treacy_compressor_preset()

    assert [optic.instance_name for optic in cfg.optics] == [
        "g1",
        "post_g1_frame",
        "gap_12",
        "g2",
        "post_g2_frame",
        "to_fold",
        "fold_frame",
        "from_fold",
        "g2_return",
        "post_g2_return_frame",
        "gap_21",
        "g1_return",
        "post_g1_return_frame",
    ]
    assert isinstance(cfg.optics[0], GratingCfg)
    assert isinstance(cfg.optics[1], FrameTransformCfg)
    assert cfg.optics[1].x_prime_scale == -1.0
    assert isinstance(cfg.optics[5], FreeSpaceCfg)
    assert cfg.optics[5].length == 0.0
    assert isinstance(cfg.optics[6], FrameTransformCfg)
    assert cfg.optics[6].x_prime_scale == -1.0
    assert cfg.tags["preset_kind"] == "treacy_compressor"
    assert cfg.tags["n_passes"] == 2


def test_treacy_preset_uses_one_way_fold_distances_on_each_leg() -> None:
    cfg = treacy_compressor_preset(length_to_mirror_um=37_500.0)

    to_fold = next(optic for optic in cfg.optics if optic.instance_name == "to_fold")
    from_fold = next(optic for optic in cfg.optics if optic.instance_name == "from_fold")
    assert isinstance(to_fold, FreeSpaceCfg)
    assert isinstance(from_fold, FreeSpaceCfg)
    assert to_fold.length == 37_500.0
    assert from_fold.length == 37_500.0


def test_treacy_preset_single_pass_ignores_mirror_leg_parameter() -> None:
    cfg = treacy_compressor_preset(n_passes=1, length_to_mirror_um=50_000.0)

    assert [optic.instance_name for optic in cfg.optics] == [
        "g1",
        "post_g1_frame",
        "gap_12",
        "g2",
        "post_g2_frame",
    ]
    assert all(optic.instance_name != "to_fold" for optic in cfg.optics)
    assert all(optic.instance_name != "fold_frame" for optic in cfg.optics)
    assert all(optic.instance_name != "from_fold" for optic in cfg.optics)


def test_treacy_preset_instance_names_are_deterministic() -> None:
    cfg_a = treacy_compressor_preset(length_to_mirror_um=12_345.0)
    cfg_b = treacy_compressor_preset(length_to_mirror_um=12_345.0)

    assert tuple(optic.instance_name for optic in cfg_a.optics) == tuple(
        optic.instance_name for optic in cfg_b.optics
    )
