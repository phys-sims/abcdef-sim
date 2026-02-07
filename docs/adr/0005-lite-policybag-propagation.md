**Title:** PolicyBag propagation and cache-key correctness
**ADR ID:** 0005-lite
**Status:** Accepted
**Date:** 2026-02-07

**Context:** Simulation runs need run-wide knobs (numeric approximation mode, precision, cache toggles) that can vary without rebuilding the pipeline. These policies can affect numeric output and thus must participate in caching/provenance.

**Options:**
- **A) Bake policies into stages at construction time.**
  - Pros: straightforward, fewer runtime parameters.
  - Cons: requires rebuilding pipeline to change policies; policy changes can invalidate caches implicitly.
- **B) Make `PolicyBag` a first-class runtime parameter.**
  - Pros: policies can change per run; easy to override; explicit in cache keys.
  - Cons: stages must accept policy arguments.

**Decision:** Use **Option B**. `SequentialPipeline.run(state, policy=...)` passes `policy` into each stage via `stage.process(state, policy=policy)`. If a policy can change numeric output, it must be part of cache keys (config-generation provenance and any stage-result caching).

**Consequences:**
- **Positive:** policy changes are explicit and auditable; caching correctness can be enforced.
- **Invariant:** policy differences that affect numeric output must change cache keys.
- **Future work:** define a canonical `policy_hash` and include it in stage-result cache keys.

**References:** docs/architecture.md; docs/how-to-use.md

---
