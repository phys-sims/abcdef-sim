**Title:** Split LaserSpec (immutable) from LaserState (mutable)
**ADR ID:** 0003-lite
**Status:** Accepted
**Date:** 2026-02-07

**Context:** The simulation needs both a stable, hashable description of the input laser and a mutable, high-volume state that evolves through fiber/GLNSE/amp stages. Pydantic models are excellent for validation and stable hashing but poor for holding large mutable numpy arrays.

**Options:**
- **A) Single Laser model for both config and runtime state.**
  - Pros: fewer types, simple signature.
  - Cons: mutable arrays complicate hashing and caching; config changes are conflated with runtime state.
- **B) Separate `LaserSpec` (immutable config) and `LaserState` (mutable runtime).**
  - Pros: stable hashing for config; clear lifecycle; optimized container for numpy arrays.
  - Cons: requires mapping from spec to initial state.

**Decision:** Use **Option B**. Define `LaserSpec` as a frozen Pydantic model containing pulse/beam parameters and omega grid parameters (`w0`, `span`, `N`, etc.). Define `LaserState` as a `State` subclass (dataclass is fine) containing mutable arrays and a custom `hashable_repr` for caching.

**Consequences:**
- **Positive:** stable hashing for presets + specs; runtime state can evolve freely.
- **Invariant:** `LaserSpec` must fully determine the omega grid used for stage configuration.
- **Future work:** Provide explicit `LaserState.from_spec(LaserSpec)` constructor utilities for consistent initialization.

**References:** docs/architecture.md; docs/how-to-use.md

---
