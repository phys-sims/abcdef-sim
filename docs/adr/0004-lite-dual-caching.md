**Title:** Separate cfg-generation cache from stage-result cache
**ADR ID:** 0004-lite
**Status:** Accepted
**Date:** 2026-02-07

**Context:** Expensive optics (e.g., gratings) require per-omega computations to generate `OpticStageCfg`. Separately, the pipeline may eventually cache stage results for exact repeated calls. Conflating these caching layers risks incorrect invalidation and muddles optimization behavior.

**Options:**
- **A) Single unified cache for cfgs and stage results.**
  - Pros: simpler plumbing.
  - Cons: incompatible key semantics; hard to reason about correctness; higher risk of stale results.
- **B) Two distinct caches: (1) internal cfg-generation cache, (2) pipeline stage-result cache.**
  - Pros: clear keying; avoids mixing policy/laser state dependencies.
  - Cons: more components to manage.

**Decision:** Use **Option B** with two caches:
- **A) Internal cfg-generation cache (for expensive optics):**
  - **L2 (per-omega):** `(optic_instance_key, omega)` → `(ABCDEF(omega), n(omega))`.
  - **L1 (per-grid):** `(optic_instance_key, grid_key)` → full arrays.
- **B) Pipeline stage-result cache (future):**
  - Key: `(state_hash, cfg_hash, policy_hash, stage_version)` → `StageResult`.

**Consequences:**
- **Positive:** L1 returns whole grids in O(1); L2 supports partial overlap for nearby grids; stage-result caching stays policy- and state-safe.
- **Failure mode to avoid:** pointer aliasing or eviction coupling between L1/L2; keep them separate to simplify invalidation.
- **Future work:** Implement stage-result cache once pipeline stages stabilize and policy hashing is defined.

**References:** docs/architecture.md

---
