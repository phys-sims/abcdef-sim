**Title:** `Split immutable LaserSpec from mutable LaserState`

- **ADR ID:** `0003`
- **Status:** `Accepted`
- **Date:** `2026-02-07`
- **Deciders:** `abcdef-sim maintainers`
- **Area:** `abcdef-sim`
- **Related:** `docs/architecture.md, docs/how-to-use.md`
- **Tags:** `data-model, caching, api`

### Context
- **Problem statement.** We need both validated, hashable input configuration and mutable large-array runtime state.
- **In/Out of scope.** In scope: modeling boundary for laser config vs evolving state. Out of scope: propagation algorithm internals.
- **Constraints.** Preserve deterministic hashes while supporting efficient numpy mutation.

### Options Considered

**Option A — Single model for config and runtime state**
- **Description:** one type carries both validated config and mutable arrays.
- **Impact areas:** caching correctness, memory model.
- **Pros:** fewer types.
- **Cons:** hash instability and lifecycle ambiguity.
- **Risks / Unknowns:** accidental cache invalidation bugs.
- **Perf/Resource cost:** inefficient copying or unsafe mutation.
- **Operational complexity:** high.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** pydantic model overhead for runtime arrays.

**Option B — Separate `LaserSpec` and `LaserState`**
- **Description:** immutable validated spec + mutable runtime state container.
- **Impact areas:** API clarity, caching, runtime performance.
- **Pros:** stable hashes; explicit lifecycle; array-friendly runtime.
- **Cons:** conversion step required.
- **Risks / Unknowns:** constructor consistency.
- **Perf/Resource cost:** slight init complexity, better runtime behavior.
- **Operational complexity:** moderate and clear.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** initialization utility maintenance.

### Decision
- **Chosen option:** **Option B**.
- **Trade-offs:** accept extra type to preserve cache safety and runtime ergonomics.
- **Scope of adoption:** input and state handling across pipeline setup/execution.

### Consequences
- **Positive:** predictable caching and cleaner state evolution semantics.
- **Negative / Mitigations:** mapping complexity; mitigate via explicit constructors/helpers.
- **Migration plan:** maintain `from_spec`-style initialization patterns.
- **Test strategy:** hash-stability tests for specs, mutability tests for state.
- **Monitoring & Telemetry:** track cache hit rates tied to `LaserSpec` keys.
- **Documentation:** document lifecycle from spec to initial state.

### Alternatives Considered (but not chosen)
- Immutable runtime snapshots only (too expensive for iterative physics updates).

### Open Questions
- Standardized helper APIs for initializing state across experiments.

### References
- `docs/architecture.md`
- `docs/how-to-use.md`

### Changelog
- `2026-02-07` — Proposed by abcdef-sim maintainers.
- `2026-02-07` — Accepted by abcdef-sim maintainers.

---
