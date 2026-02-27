**Title:** `Adopt variance-aware generics for phys-pipeline wrappers`

- **ADR ID:** `0006`
- **Status:** `Accepted`
- **Date:** `2026-02-07`
- **Deciders:** `abcdef-sim maintainers`
- **Area:** `abcdef-sim`
- **Related:** `docs/architecture.md`
- **Tags:** `api, typing, maintainability`

### Context
- **Problem statement.** Wrapper stages and pipeline-as-stage patterns need preserved state typing and compatible variance behavior.
- **In/Out of scope.** In scope: type-system design around stages/results/pipelines. Out of scope: runtime enforcement beyond existing checks.
- **Constraints.** Keep wrappers type-safe without excessive casts or brittle invariance errors.

### Options Considered

**Option A — Non-generic `StageResult` with invariant containers**
- **Description:** minimize generics and rely on coarse typing.
- **Impact areas:** dev ergonomics, static safety.
- **Pros:** simpler signatures.
- **Cons:** loss of end-to-end state typing.
- **Risks / Unknowns:** unsafe casts and hidden incompatibilities.
- **Perf/Resource cost:** none at runtime.
- **Operational complexity:** high debugging complexity.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** type-checker suppression pressure.

**Option B — Generic/variance-aware stage and pipeline typing**
- **Description:** use contravariant state input, covariant cfg output, and `StageResult[S]`.
- **Impact areas:** stage interfaces, wrapper composition.
- **Pros:** stronger static guarantees; safer composition.
- **Cons:** more advanced typing syntax.
- **Risks / Unknowns:** occasional mypy edge cases.
- **Perf/Resource cost:** none at runtime.
- **Operational complexity:** moderate.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** alignment with `phys-pipeline` typing contracts.

### Decision
- **Chosen option:** **Option B**.
- **Trade-offs:** added typing complexity for safer composition and fewer runtime surprises.
- **Scope of adoption:** wrappers and sequential pipeline definitions in `abcdef-sim`.

### Consequences
- **Positive:** end-to-end typed state propagation.
- **Negative / Mitigations:** steeper typing curve; mitigate with examples and documented patterns.
- **Migration plan:** keep generic signatures in wrappers and avoid invariant list pitfalls.
- **Test strategy:** strict mypy checks on wrapper composition scenarios.
- **Monitoring & Telemetry:** none beyond CI typing gates.
- **Documentation:** architecture notes on variance assumptions.

### Alternatives Considered (but not chosen)
- Runtime-only type guards with relaxed static typing.

### Open Questions
- Additional helper aliases to simplify public type signatures.

### References
- `docs/architecture.md`

### Changelog
- `2026-02-07` — Proposed by abcdef-sim maintainers.
- `2026-02-07` — Accepted by abcdef-sim maintainers.

---
