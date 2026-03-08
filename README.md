## abcdef-sim
`abcdef-sim` is an **architecture-first simulation scaffold** for frequency-dependent ray-transfer (ABCDEF) optics on top of `phys-pipeline`.

> Current state: the repository includes validated data models, pipeline assembly, Martinez-aligned stage propagation, and pure-data result synthesis helpers.

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
- Stage physics + result synthesis:
  - `physics.abcdef.adapters.apply_cfg(...)`
  - `physics.abcdef.compute_pipeline_result(...)`

## What is intentionally not implemented yet

- Additional optic implementations (e.g., `Grating` builder is stubbed/not registered).
- Stage-result caching at pipeline execution level.

## Pure physics layer and validation

New additive pure-physics modules live under:
- `src/abcdef_sim/physics/abcd` for paraxial ABCD transfer math (matrices, rays, Gaussian q propagation).
- `src/abcdef_sim/physics/abcdef` for structured ABCDEF conventions, Martinez phase helpers, batched propagation kernels, and pure-data pipeline result synthesis.

Martinez phase bookkeeping is stored in radians. The per-optic `phi3_rad` term uses the post-element displacement sign already validated in `tests/physics/test_martinez_phase_terms.py`.

To run reference validation tests against the external `raytracing` package:

```bash
pip install -e '.[validation]'
pytest tests/physics/test_abcd_against_raytracing.py
```

If `raytracing` is not installed, the validation test module is skipped.

To generate the graphical validation artifact described by the physics tests:

```bash
python scripts/generate_abcd_validation_plot.py
```

This writes a deterministic PNG to `artifacts/physics/thick_lens_vs_raytracing.png`.

To generate the Treacy compressor validation and benchmarking artifacts:

```bash
python scripts/generate_treacy_benchmark_artifacts.py
```

This writes deterministic JSON/PNG artifacts to `artifacts/physics/`:

- `treacy_radius_convergence.json`
- `treacy_radius_convergence.png`
- `treacy_radius_mirror_heatmap.json`
- `treacy_radius_mirror_heatmap.png`

The benchmark uses the local Treacy analytic model as the plane-wave baseline and
plots the finite-beam comparison phase after removing the Martinez ray-centering
term `phi2`, which does not belong to the plane-wave model and does not vanish
with beam radius. The large-beam limit of that reduced comparison should
approach the analytic baseline while mirror-leg length still perturbs the
finite-beam result.


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
