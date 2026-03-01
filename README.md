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


## Running tests by marker

Use pytest markers to select the right validation depth for your change:

```bash
# all tests
pytest -q

# fast gate (exclude runtime-heavy tests)
pytest -m "not slow" -q

# physics validation lane (oracle/analytical/regression)
pytest -m "physics" -q

# quick physics checks only
pytest -m "physics and not slow" -q
```

When to use each:
- `pytest -q`: full local confidence before merging.
- `pytest -m "not slow" -q`: fast feedback while iterating on most changes.
- `pytest -m "physics" -q`: validate optical transfer behavior against physics checks and references.
- `pytest -m "physics and not slow" -q`: rapid physics sanity checks during refactors.
