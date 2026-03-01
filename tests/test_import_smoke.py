def test_import_smoke() -> None:
    import abcdef_sim
    import abcdef_sim.data_models.configs
    import abcdef_sim.data_models.states
    import abcdef_sim.optics
    import abcdef_sim.physics.abcdef

    assert abcdef_sim is not None
