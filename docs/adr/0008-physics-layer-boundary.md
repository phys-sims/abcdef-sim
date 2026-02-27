**Title:** `Define pure-physics layer boundary under physics/*`

- **ADR ID:** `0008`
- **Status:** `Accepted`
- **Date:** `2026-02-27`
- **Deciders:** `abcdef-sim maintainers`
- **Area:** `abcdef-sim`
- **Related:** `ADR-0007, docs/architecture.md`
- **Tags:** `architecture, physics, testing`

### Context
- **Problem statement.** We need a reviewable physics core that is isolated from pipeline/config/cache concerns while preserving current runtime behavior.
- **In/Out of scope.** In scope: package boundary for pure kernels vs adapters/orchestration. Out of scope: replacing current placeholder stage physics behavior.
- **Constraints.** Additive non-breaking change, minimal dependencies for kernels, no pipeline wiring in pure physics modules.

### Options Considered

**Option A — Keep physics logic embedded in pipeline and model layers**
- **Description:** continue existing mixed responsibility organization.
- **Impact areas:** architecture clarity, reviewability.
- **Pros:** no new package surfaces.
- **Cons:** harder isolated validation and slower physics review.
- **Risks / Unknowns:** architecture drift.
- **Perf/Resource cost:** none immediate.
- **Operational complexity:** high due to coupling.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** coupling to pipeline internals.

**Option B — Add `physics/abcd` + `physics/abcdef` pure kernels with external adapters**
- **Description:** move foundational numerics to standalone modules; keep adapters and runtime wiring elsewhere.
- **Impact areas:** package layout, tests, docs.
- **Pros:** isolated review/testing and cleaner boundaries.
- **Cons:** temporary duplication while legacy placeholders remain.
- **Risks / Unknowns:** migration path for future stage integration.
- **Perf/Resource cost:** negligible.
- **Operational complexity:** moderate with clearer ownership.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** optional validation dependency (`raytracing`) in adapters/tests only.

### Decision
- **Chosen option:** **Option B**.
- **Trade-offs:** accept temporary overlap to create a maintainable long-term architecture seam.
- **Scope of adoption:** all new foundational ABCD/ABCDEF numerics in `src/abcdef_sim/physics/*`; adapters under `src/abcdef_sim/adapters/*`.

### Consequences
- **Positive:** physics code is audit-friendly and decoupled from pipeline policy/cache details.
- **Negative / Mitigations:** duplicate representations in transition; mitigate via explicit ADR boundary and future integration plan.
- **Migration plan:** keep runtime behavior unchanged by default; adopt pure kernels in stage implementations only behind explicit future decisions.
- **Test strategy:** unit tests for kernels plus optional external reference validation.
- **Monitoring & Telemetry:** monitor regression via physics-focused test suite.
- **Documentation:** README and ADR index include new physics layout.

### Alternatives Considered (but not chosen)
- Single monolithic `physics.py` module.
- Broad OOP hierarchy for optical elements in this layer.

### Open Questions
- Which future stage(s) should first consume the pure kernels, and behind what feature-gating strategy.

### References
- `docs/architecture.md`
- `docs/adr/0007-lite-eco-0001-conventions-adoption.md`

### Changelog
- `2026-02-27` — Proposed by abcdef-sim maintainers.
- `2026-02-27` — Accepted by abcdef-sim maintainers.

---
