**Title:** `Separate cfg-generation cache from stage-result cache`

- **ADR ID:** `0004`
- **Status:** `Accepted`
- **Date:** `2026-02-07`
- **Deciders:** `abcdef-sim maintainers`
- **Area:** `abcdef-sim`
- **Related:** `docs/architecture.md`
- **Tags:** `performance, caching, ops`

### Context
- **Problem statement.** Cfg generation and stage execution caching have different key semantics and invalidation rules.
- **In/Out of scope.** In scope: cache layering strategy. Out of scope: exact backend implementation details.
- **Constraints.** Avoid stale outputs, keep key provenance explicit, support fast repeated cfg builds.

### Options Considered

**Option A — Unified cache for cfg and stage results**
- **Description:** one shared cache space for both concerns.
- **Impact areas:** correctness, invalidation logic.
- **Pros:** less plumbing.
- **Cons:** mixed semantics, hard invalidation boundaries.
- **Risks / Unknowns:** stale result reuse.
- **Perf/Resource cost:** potentially high miss/stale-management overhead.
- **Operational complexity:** high.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** stronger coupling of subsystems.

**Option B — Distinct caches for cfg-generation and stage-result outputs**
- **Description:** keep cfg cache independent from runtime output caching.
- **Impact areas:** correctness and operability.
- **Pros:** clear keys and lifecycle; safer invalidation.
- **Cons:** two components to maintain.
- **Risks / Unknowns:** duplicated instrumentation.
- **Perf/Resource cost:** better steady-state due to cleaner targeting.
- **Operational complexity:** moderate.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** interface contracts for both caches.

### Decision
- **Chosen option:** **Option B**.
- **Trade-offs:** accept additional components to preserve correctness.
- **Scope of adoption:** current cfg caching and planned stage-result caching.

### Consequences
- **Positive:** safer cache correctness and easier reasoning.
- **Negative / Mitigations:** extra management overhead; mitigate with clear interfaces.
- **Migration plan:** keep cfg cache live; add stage-result cache behind separate keys when ready.
- **Test strategy:** cache-key unit tests and invalidation behavior checks.
- **Monitoring & Telemetry:** separate hit/miss/eviction metrics per cache tier.
- **Documentation:** architecture docs describe two-layer cache model.

### Alternatives Considered (but not chosen)
- Per-stage bespoke caches without shared abstractions.

### Open Questions
- Final canonical key schema for stage-result cache policy/versioning.

### References
- `docs/architecture.md`

### Changelog
- `2026-02-07` — Proposed by abcdef-sim maintainers.
- `2026-02-07` — Accepted by abcdef-sim maintainers.

---
