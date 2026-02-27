**Title:** `Separate immutable SystemPreset from SystemAssembler runtime build`

- **ADR ID:** `0002`
- **Status:** `Accepted`
- **Date:** `2026-02-07`
- **Deciders:** `abcdef-sim maintainers`
- **Area:** `abcdef-sim`
- **Related:** `docs/architecture.md, docs/how-to-use.md`
- **Tags:** `api, architecture, data-model`

### Context
- **Problem statement.** Presets must remain immutable/hashable, while stage assembly depends on run policy and laser grid.
- **In/Out of scope.** In scope: build responsibility boundary. Out of scope: specific stage physics.
- **Constraints.** Preserve hash purity; allow run-time policy variation without mutating presets.

### Options Considered

**Option A — Preset owns assembly**
- **Description:** `SystemPreset` builds stage cfgs/stages directly.
- **Impact areas:** data model purity, API coupling.
- **Pros:** fewer objects.
- **Cons:** mixes immutable config with mutable runtime concerns.
- **Risks / Unknowns:** broken hash provenance.
- **Perf/Resource cost:** negligible immediate difference.
- **Operational complexity:** high conceptual coupling.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** policy/cache internals leak into model layer.

**Option B — Dedicated `SystemAssembler` build step**
- **Description:** `SystemAssembler` takes `(SystemPreset, LaserSpec, PolicyBag)` and produces cfgs/stages.
- **Impact areas:** architecture boundary, testability.
- **Pros:** pure presets; policy-swappable runs; isolated testing.
- **Cons:** explicit builder object needed.
- **Risks / Unknowns:** none major.
- **Perf/Resource cost:** minor orchestration overhead.
- **Operational complexity:** clearer contracts.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** stage cfg generation interfaces.

### Decision
- **Chosen option:** **Option B**.
- **Trade-offs:** accept extra layer for better separation and reproducibility.
- **Scope of adoption:** all pipeline construction paths in `abcdef-sim`.

### Consequences
- **Positive:** clean immutable/runtime split and explicit policy flow.
- **Negative / Mitigations:** more plumbing; mitigate with helper constructors.
- **Migration plan:** continue routing build through assembler APIs.
- **Test strategy:** unit tests for cfg ordering and omega alignment invariants.
- **Monitoring & Telemetry:** build timing and cache efficacy metrics.
- **Documentation:** maintain architecture/how-to usage for assembly flow.

### Alternatives Considered (but not chosen)
- Utility free functions without a dedicated assembler abstraction.

### Open Questions
- Stage instrumentation extension points for tracing/metrics hooks.

### References
- `docs/architecture.md`
- `docs/how-to-use.md`

### Changelog
- `2026-02-07` — Proposed by abcdef-sim maintainers.
- `2026-02-07` — Accepted by abcdef-sim maintainers.

---
