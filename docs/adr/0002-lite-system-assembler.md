**Title:** Separate SystemPreset from SystemAssembler build steps
**ADR ID:** 0002-lite
**Status:** Accepted
**Date:** 2026-02-07

**Context:** Presets must remain immutable and hashable. Assembly depends on policies (caching toggles, numeric approximation modes) and on the laser grid, which can vary per run. Merging assembly logic into `SystemPreset` would couple it to mutable runtime concerns and contaminate hashing.

**Options:**
- **A) `SystemPreset` owns assembly (builds stages itself).**
  - Pros: fewer objects, straightforward API.
  - Cons: mixes immutable config with runtime policy/cache, breaks hashing guarantees.
- **B) `SystemAssembler` takes `(SystemPreset, LaserSpec, PolicyBag)` and produces cfgs/stages.**
  - Pros: preset stays pure-data; policy/caching can vary per run; assembly is testable in isolation.
  - Cons: introduces a build step and a separate object.

**Decision:** Use **Option B**. Implement `SystemAssembler` (or builder) that consumes `(SystemPreset, LaserSpec, PolicyBag)` and emits:
1) Ordered `OpticStageCfg` list aligned to the laser omega grid.
2) Ordered stage list (e.g., `AbcdefOpticStage(cfg=...)`) to feed into `SequentialPipeline`.

**Consequences:**
- **Positive:** hashing/serialization stays clean; policies and caches can be swapped without mutating presets.
- **Invariant:** `cfg.omega[i]` aligns with `ABCDEF[i]` and `n[i]` for every optic stage.
- **Future work:** Add builder hooks for stage instrumentation (metrics, tracing) without changing presets.

**References:** docs/architecture.md; docs/how-to-use.md

---
