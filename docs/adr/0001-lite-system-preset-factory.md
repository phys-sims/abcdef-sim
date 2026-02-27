**Title:** `Use pure-data SystemPreset with OpticSpec + factory assembly`

- **ADR ID:** `0001`
- **Status:** `Accepted`
- **Date:** `2026-02-07`
- **Deciders:** `abcdef-sim maintainers`
- **Area:** `abcdef-sim`
- **Related:** `docs/architecture.md`
- **Tags:** `api, data-model, caching`

### Context
- **Problem statement.** We need an optics layout representation that is cache-efficient, optimizer-friendly, and portable across runs. Live runtime `Optic` instances are hard to hash deterministically, difficult to serialize, and awkward for topology edits.
- **In/Out of scope.** In scope: representation of system layout and assembly boundary. Out of scope: numerical propagation implementation details.
- **Constraints.** Deterministic hashing, stable serialization, and compatibility with pipeline assembly.

### Options Considered

**Option A — Store live `Optic` instances in layout data**
- **Description:** Layout directly contains constructed optics.
- **Impact areas:** data model, cache keys, optimizer workflows.
- **Pros:** direct runtime use; fewer conversion steps.
- **Cons:** weak hashing/serialization; difficult compare/edit operations.
- **Risks / Unknowns:** mutability leaks into config plane.
- **Perf/Resource cost:** minor immediate savings, higher long-term cache inefficiency.
- **Operational complexity:** high debugging burden around identity/state.
- **Security/Privacy/Compliance:** none specific.
- **Dependencies / Externalities:** runtime object graph behavior.

**Option B — Store pure `OpticSpec` data and instantiate via `OpticFactory`**
- **Description:** Use immutable ordered specs in `SystemPreset`; construct runtime optics during assembly.
- **Impact areas:** data model, assembly API, docs.
- **Pros:** deterministic hashing; easy serialization; data-only topology search.
- **Cons:** explicit assembly step required.
- **Risks / Unknowns:** registry completeness/validation needed.
- **Perf/Resource cost:** small assembly overhead; better cache hit behavior.
- **Operational complexity:** manageable with clear factory boundary.
- **Security/Privacy/Compliance:** none specific.
- **Dependencies / Externalities:** factory registry maintenance.

### Decision
- **Chosen option:** **Option B** because it preserves immutable config semantics while enabling robust caching and topology edits.
- **Trade-offs:** accept assembly indirection to gain determinism and portability.
- **Scope of adoption:** applies to system layout representation in `abcdef-sim`; runtime stages remain separately constructed.

### Consequences
- **Positive:** stable cache keys, portable presets, cleaner optimizer integration.
- **Negative / Mitigations:** factory maintenance required; mitigate with validation tests and schema checks.
- **Migration plan:** keep layout as pure data and route all runtime construction through factory APIs.
- **Test strategy:** unit tests for factory resolution and preset hash stability.
- **Monitoring & Telemetry:** track cache hit/miss rates for cfg generation.
- **Documentation:** architecture and usage docs must describe preset-vs-runtime split.

### Alternatives Considered (but not chosen)
- Hybrid models that store both data and live objects in one structure.

### Open Questions
- Add stricter uniqueness/schema validation for `instance_name` in all load paths.

### References
- `docs/architecture.md`

### Changelog
- `2026-02-07` — Proposed by abcdef-sim maintainers.
- `2026-02-07` — Accepted by abcdef-sim maintainers.

---
