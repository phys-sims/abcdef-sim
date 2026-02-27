**Title:** `Propagate PolicyBag at runtime and include policy in cache provenance`

- **ADR ID:** `0005`
- **Status:** `Accepted`
- **Date:** `2026-02-07`
- **Deciders:** `abcdef-sim maintainers`
- **Area:** `abcdef-sim`
- **Related:** `docs/architecture.md, docs/how-to-use.md`
- **Tags:** `api, caching, provenance`

### Context
- **Problem statement.** Numeric and caching policies vary by run and can change outputs, so policy must be explicit at execution time.
- **In/Out of scope.** In scope: policy transport through pipeline and cache-key implications. Out of scope: final policy schema design.
- **Constraints.** Preserve reproducibility and avoid implicit stale-cache reuse.

### Options Considered

**Option A — Bake policies into stage construction**
- **Description:** policy fixed when building stages.
- **Impact areas:** pipeline API, reproducibility.
- **Pros:** simpler execution signature.
- **Cons:** rebuild required for policy changes.
- **Risks / Unknowns:** hidden provenance differences.
- **Perf/Resource cost:** repeated rebuild overhead.
- **Operational complexity:** medium/high.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** tight coupling to assembly lifecycle.

**Option B — Pass `PolicyBag` as runtime parameter**
- **Description:** pipeline forwards policy through stage processing.
- **Impact areas:** stage interfaces and cache-key strategy.
- **Pros:** explicit policy provenance and per-run overrides.
- **Cons:** stage methods accept policy args.
- **Risks / Unknowns:** requires disciplined cache-key inclusion.
- **Perf/Resource cost:** negligible call overhead.
- **Operational complexity:** moderate and transparent.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** policy hash/versioning decisions.

### Decision
- **Chosen option:** **Option B**.
- **Trade-offs:** slightly broader interfaces for stronger auditability.
- **Scope of adoption:** `SequentialPipeline.run(state, policy=...)` and all stages with policy-sensitive numerics.

### Consequences
- **Positive:** explicit and testable policy flow.
- **Negative / Mitigations:** API surface increases; mitigate via consistent stage base signatures.
- **Migration plan:** maintain policy forwarding in wrappers and include policy in affected keys.
- **Test strategy:** policy-delta tests proving output/key changes when policy changes numerics.
- **Monitoring & Telemetry:** policy-hash tagged cache metrics.
- **Documentation:** describe policy propagation and cache implications in architecture/how-to docs.

### Alternatives Considered (but not chosen)
- Global singleton policy context.

### Open Questions
- Canonical `policy_hash` implementation details and versioning lifecycle.

### References
- `docs/architecture.md`
- `docs/how-to-use.md`

### Changelog
- `2026-02-07` — Proposed by abcdef-sim maintainers.
- `2026-02-07` — Accepted by abcdef-sim maintainers.

---
