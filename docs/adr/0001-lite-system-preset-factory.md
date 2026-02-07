**Title:** Pure-data SystemPreset with OpticSpec + Factory
**ADR ID:** 0001-lite
**Status:** Accepted
**Date:** 2026-02-07

**Context:** We need a representation of an optics layout that is cache-efficient, optimizer-friendly, and portable across runs. Live `Optic` instances are difficult to hash deterministically, hard to serialize, and complicate topology search (e.g., insert/remove optics). The layout must be used by the assembly layer to produce runtime stages for `SequentialPipeline` without carrying mutable object state.

**Options:**
- **A) Store live Optic instances in the layout.**
  - Pros: direct runtime use, fewer conversion steps.
  - Cons: weak hashing/serialization, hard to compare, messy for topology edits.
- **B) Store pure data in a SystemPreset and instantiate later (factory).**
  - Pros: immutable + hashable; easy serialization; topology edits are data-only; decoupled runtime construction.
  - Cons: requires a factory/registry and an assembly step.

**Decision:** Use **Option B**. Define a pure-data `SystemPreset` containing ordered `OpticSpec` entries (`kind`, `instance_name`, `params`, `tags`) and keep it immutable/hashable. Use an `OpticFactory` registry to construct runtime `Optic` objects from `OpticSpec` at assembly time.

**Consequences:**
- **Positive:** clean hashing for caching and optimizer loops; presets are portable; topology search becomes a pure-data operation.
- **Constraints:** `instance_name` must be unique within a preset; `OpticFactory` must be the only place where runtime construction occurs.
- **TODO/Future:** Add validation helpers to assert uniqueness and schema compliance when presets are created or loaded.

**References:** docs/architecture.md

---
