# abcdef-sim quickstart (happy path)

This guide shows the minimal, recommended flow for constructing an optics layout, assembling it into a pipeline, and running it with policies. It is intentionally explicit about invariants and hashing to make future optimization and caching work predictable.

> **Note:** Code snippets use conceptual names (`SystemPreset`, `OpticSpec`, `PolicyBag`) that are part of the planned architecture; implementers should map them to the actual modules/classes as they are added.

## 1) Define a `SystemPreset` with `OpticSpec`

A preset is **pure data**: ordered `OpticSpec` entries with `kind`, `instance_name`, and `params`. Presets are immutable and hashable.

```python
preset = SystemPreset(
    optics=(
        OpticSpec(
            kind="grating",
            instance_name="g1",
            params={"line_density": 1200, "incidence_deg": 30.0},
            tags={"compressor"},
        ),
        OpticSpec(
            kind="lens",
            instance_name="l1",
            params={"f_mm": 200.0},
            tags=set(),
        ),
    )
)
```

**Invariant:** `instance_name` must be unique within the preset. Presets must not store live `Optic` instances.

## 2) Define an immutable `LaserSpec`

`LaserSpec` is a frozen Pydantic model. It is **pure config**: pulse/beam parameters + omega grid definition.

```python
laser_spec = LaserSpec(
    center_wavelength_nm=1030.0,
    energy_uj=200.0,
    duration_fs=250.0,
    grid=OmegaGridSpec(w0=2.0e15, span=4.0e14, n=4096),
)
```

**Invariant:** the grid defined by `LaserSpec` drives `OpticStageCfg` alignment.

## 3) Assemble the system with `SystemAssembler`

Assembly depends on the laser grid and policies, so it stays out of `SystemPreset`.

```python
policy = PolicyBag(
    cfg_cache_enabled=True,
    numeric_mode="taylor",
    precision="float64",
)

assembler = SystemAssembler(OpticFactory.registry())

cfgs, stages = assembler.build(
    preset=preset,
    laser_spec=laser_spec,
    policy=policy,
)

pipeline = SequentialPipeline(stages)
```

**Invariant:** `cfg.omega[i]` aligns with `ABCDEF[i]` and `n[i]` for every optic stage.

## 4) Initialize a mutable `LaserState`

`LaserState` holds arrays and evolves through the pipeline. It is **not** a Pydantic model.

```python
state = LaserState.from_spec(laser_spec)
```

For GLNSE + fiber coupling, the laser state evolves while the preset remains fixed; only the policy and state change per run or per iteration.

## 5) Run the pipeline with a `PolicyBag`

Policies are passed **at runtime** to keep the pipeline reusable without rebuilding stages.

```python
result = pipeline.run(state, policy=policy)
```

To toggle caching or numeric modes between runs, create a new `PolicyBag` and pass it into `run`:

```python
policy_no_cache = policy.model_copy(update={"cfg_cache_enabled": False})

result = pipeline.run(state, policy=policy_no_cache)
```

### Caching behavior (expected)

- **Cfg-generation cache** (internal to optics): speeds up `OpticStageCfg` creation for expensive optics.
- **Stage-result cache** (future): caches stage results for identical `(state, cfg, policy, stage_version)`.

## Common failure modes

- **Hash instability:** using mutable objects inside presets breaks caching; keep presets pure data.
- **Policy mismatch:** if policy changes numeric output but is excluded from cache keys, cached results are invalid.
- **Float grid jitter:** tiny differences in `omega` grid can cause cache misses; stabilize grid construction.
- **Quantization collisions:** overly coarse grid-key quantization can produce incorrect L1 cache hits.

---

For full architecture and extension points, see [docs/architecture.md](architecture.md).
