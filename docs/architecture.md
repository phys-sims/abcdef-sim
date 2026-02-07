# abcdef-sim architecture

This document explains the architecture for the CPA optics simulation on top of `phys-pipeline`. It focuses on **reasoning**, **invariants**, **caching**, and **extension points** needed for optimization and topology search.

> **Scope:** The components described here are the intended architecture; where an item is not yet implemented, it is labeled **Future work**.

## High-level flow

```text
SystemPreset (OpticSpec[])  +  LaserSpec  +  PolicyBag
              |                    |            |
              |                    |            |
              v                    v            v
                      SystemAssembler
                          |  (uses OpticFactory)
                          v
          OpticStageCfg[] (aligned to omega grid)
                          |
                          v
                   Pipeline stages
                          |
                          v
                SequentialPipeline.run
                          |
                          v
                     LaserState -> StageResult
```

### Key invariants

- `SystemPreset` is **pure data** and **immutable**.
- `OpticSpec.instance_name` is **unique within a preset**.
- `LaserSpec` fully determines the omega grid used for stage configuration.
- For every stage, `cfg.omega[i]` aligns with `ABCDEF[i]` and `n[i]`.
- Policies that affect numeric output must be part of cache keys.

## Components

### SystemPreset + OpticSpec

**Purpose:** Provide a cache-efficient, optimizer-friendly representation of the optics layout.

- `SystemPreset` holds an **ordered list** of `OpticSpec` entries.
- `OpticSpec` includes:
  - `kind`: optic type (e.g., `grating`, `lens`).
  - `instance_name`: unique ID for hashing and topology edits.
  - `params`: pure configuration parameters.
  - `tags`: optional metadata for optimizers or policies.

**Why:** Presets must be immutable and hashable to support caching and topology search. Live `Optic` objects are hard to serialize and hash deterministically.

### OpticFactory

**Purpose:** Central registry for constructing runtime `Optic` objects from `OpticSpec`.

- Maps `kind` → constructor.
- Enforces consistent instantiation logic.

**Why:** Keeps `SystemPreset` pure data while enabling controlled construction of runtime objects.

### SystemAssembler

**Purpose:** Assemble a preset into pipeline stages for a specific laser grid and policy.

**Inputs:** `(SystemPreset, LaserSpec, PolicyBag)`.

**Outputs:**
1. `OpticStageCfg[]` aligned to the laser omega grid.
2. Runtime stage list (e.g., `AbcdefOpticStage(cfg=...)`).

**Why:** Separates immutable data from runtime concerns (policy, caching, grid-specific cfg generation).

### LaserSpec vs LaserState

- **LaserSpec (immutable):** Pydantic frozen model with pulse/beam parameters + omega grid. Hashable and validated.
- **LaserState (mutable):** Runtime arrays evolving through GLNSE/fiber/amp stages; a `State` subclass with `hashable_repr` for caching.

**Why:** Pydantic models are ideal for stable config hashing, not for large mutable numpy arrays.

### PolicyBag

**Purpose:** Run-wide knobs (cache toggles, numeric approximation mode, precision, etc.).

- Passed into `SequentialPipeline.run(state, policy=...)`.
- Propagated to `stage.process(state, policy=policy)`.

**Caching rule:** If a policy can affect numeric output, it must be part of cache keys.

## Caching architecture

We maintain **two distinct caches**. Do not conflate them.

### A) Internal cfg-generation cache (expensive optics)

**Purpose:** Speed up `OpticStageCfg` generation for optics that require per-omega calculations.

- **L2 (per-omega):**
  - Key: `(optic_instance_key, omega)`
  - Value: `(ABCDEF(omega), n(omega))`
- **L1 (per-grid):**
  - Key: `(optic_instance_key, grid_key)`
  - Value: full arrays (`ABCDEF[]`, `n[]`)

**Why separate L1/L2:**
- L1 supports O(1) whole-grid return for identical grids.
- L2 allows partial overlap for nearby grids.
- Avoid pointer aliasing and eviction coupling between different cache granularities.

### B) Pipeline stage-result cache (future work)

**Purpose:** Cache exact repeated calls:

- Key: `(state_hash, cfg_hash, policy_hash, stage_version)`
- Value: `StageResult`

**Note:** If the laser state changes every iteration, cache hits may be rare but still valuable for repeated evaluations and batch workflows.

## Type system choices (phys-pipeline)

To support wrappers like abcdef-sim and pipeline-as-stage, the pipeline types must preserve state types:

- `PipelineStage[S, C]` uses variance:
  - **State is contravariant** to accept broader state types.
  - **Cfg is covariant** to allow more specific cfg types.
- `StageResult[S]` is generic in the state type.
- `SequentialPipeline` is `Generic[S]` and accepts `Sequence[PipelineStage[S, Any]]`.

**Why it matters:** Without this, wrappers require unsafe casts and lose state typing across the pipeline loop.

## Extension points

### Add a new optic kind

1. **Define schema:** extend `OpticSpec` parameters for the new `kind`.
2. **Implement optic:** create an `Optic` runtime class with cfg-generation logic.
3. **Register:** add the new kind to `OpticFactory`.
4. **Tests:** validate cfg alignment invariants and caching behavior.

### Add a new stage

1. **Define cfg:** create a `StageCfg` model.
2. **Implement stage:** subclass `PipelineStage` with `process(state, policy=...)`.
3. **Integrate:** update `SystemAssembler` to emit the new stage when needed.

### Plug in optimization loops + topology search

- **Continuous optimization:** adjust `OpticSpec.params` for numeric optimization; presets remain hashable.
- **Topology changes:** insert/remove `OpticSpec` entries in the preset; hash changes cleanly.
- **Policy sweeps:** vary `PolicyBag` per run without rebuilding the preset.

## Common failure modes

- **Hash instability:** storing live objects in presets or stateful params breaks deterministic hashes.
- **Policy mismatch:** omitting policy from cache keys can yield incorrect results.
- **Float grid jitter:** minor `omega` differences prevent cache reuse; use stable grid construction.
- **Quantization collisions:** coarse grid-key quantization can cause incorrect L1 hits.
- **Instance name reuse:** non-unique `instance_name` can confuse caches and provenance.

## Future work

- Implement stage-result caching once `policy_hash` and `stage_version` are defined.
- Add validation utilities for presets and grid alignment invariants.
- Provide configurable cache storage backends and eviction policies.
