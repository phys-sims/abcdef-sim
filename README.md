## abcdef-sim
`abcdef-sim` is an **architecture-first simulation scaffold** for frequency-dependent ray-transfer (ABCDEF) optics on top of `phys-pipeline`.

> Current state: the repository includes data models, assembly, caching, and pipeline wiring. The actual stage physics update is still a placeholder from an ongoing refactor.

## What is implemented today

- Immutable, validated input models:
  - `SystemPreset` / `OpticSpec`
  - `LaserSpec`
- Runtime state model:
  - `RayState`
- Optic abstraction and factory:
  - `Optic` base class
  - `FreeSpace` optic
  - `OpticFactory.default()` registry
- Stage config generation:
  - `OpticStageCfgGenerator`
  - optional two-level cfg cache (`L1` grid + `L2` per-omega)
- Pipeline assembly:
  - `SystemAssembler.build_optic_cfgs(...)`
  - `SystemAssembler.build_pipeline(...)`
  - `AbcdefOpticStage` wrapper stages

## What is intentionally not implemented yet

- Physics propagation in `AbcdefOpticStage.process(...)` (currently returns input state unchanged).
- Additional optic implementations (e.g., `Grating` builder is stubbed/not registered).
- Stage-result caching at pipeline execution level.

## Pure physics layer and validation

New additive pure-physics modules live under:
- `src/abcdef_sim/physics/abcd` for paraxial ABCD transfer math (matrices, rays, Gaussian q propagation).
- `src/abcdef_sim/physics/abcdef` for structured dispersion-aware ABCDEF placeholders.

To run reference validation tests against the external `raytracing` package:

```bash
pip install -e '.[validation]'
pytest tests/physics/test_abcd_against_raytracing.py
```

If `raytracing` is not installed, the validation test module is skipped.
