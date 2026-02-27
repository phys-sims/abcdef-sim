**Title:** `Adopt ECO-0001 conventions and units contract in abcdef-sim`

- **ADR ID:** `0007`
- **Status:** `Accepted`
- **Date:** `2026-02-27`
- **Deciders:** `abcdef-sim maintainers`
- **Area:** `abcdef-sim`
- **Related:** `ECO-0001, cpa-sim ADR-0001 style reference`
- **Tags:** `physics, conventions, units, testing`

### Context
- **Problem statement.** Physically meaningful stretcher/compressor and propagation behavior requires consistent units/signs across stages and repo boundaries.
- **In/Out of scope.** In scope: units, sign conventions, FFT scaling, and boundary contract semantics. Out of scope: choosing one specific optical layout sign convention for every external backend detail.
- **Constraints.** Interoperability with ecosystem contracts, reproducibility, explicit unit annotations.

### Options Considered

**Option A — Keep local repo-specific conventions**
- **Description:** define internal conventions ad hoc without formal external contract adoption.
- **Impact areas:** API compatibility, cross-repo validation.
- **Pros:** local flexibility.
- **Cons:** interoperability risk and ambiguous signs.
- **Risks / Unknowns:** drift and inconsistent interpretation.
- **Perf/Resource cost:** none immediate.
- **Operational complexity:** high long-term due to ambiguity.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** weaker ecosystem alignment.

**Option B — Adopt ECO-0001 contract explicitly**
- **Description:** treat ECO-0001 as authoritative baseline and document boundary conversions.
- **Impact areas:** docs, tests, future stage implementations.
- **Pros:** shared language for units/signs; easier validation.
- **Cons:** stricter implementation discipline required.
- **Risks / Unknowns:** migration effort in future physics stages.
- **Perf/Resource cost:** minimal.
- **Operational complexity:** moderate and explicit.
- **Security/Privacy/Compliance:** none.
- **Dependencies / Externalities:** coordination with ecosystem ADR updates.

### Decision
- **Chosen option:** **Option B**.
- **Trade-offs:** reduced local ambiguity in exchange for explicit convention compliance work.
- **Scope of adoption:** all internal physics calculations and documented API boundaries in `abcdef-sim`.

### Consequences
- **Positive:** consistent semantics for dispersion, grating signs, beam normalization, and FFT handling.
- **Negative / Mitigations:** additional documentation/testing burden; mitigate by convention-focused tests and docstring requirements.
- **Migration plan:** enforce conventions in new physics modules; preserve behavior in legacy placeholders until explicitly updated.
- **Test strategy:** sign-sensitive dispersion tests, grating-orientation checks, FFT round-trip/Parseval validations.
- **Monitoring & Telemetry:** add regression checks in CI for convention-sensitive tests.
- **Documentation:** keep ADR and README references to adopted contract.

### Alternatives Considered (but not chosen)
- Adopt only unit names while deferring sign/FFT conventions.

### Open Questions
- Final boundary conversion policy for user-facing I/O surfaces as more stage APIs are exposed.

### References
- ECO-0001 conventions: <https://github.com/phys-sims/cpa-architecture/blob/main/docs/adr/ECO-0001-conventions-units.md>
- Style reference: <https://github.com/phys-sims/cpa-sim/blob/main/docs/adr/ADR-0001-conventions-units.md>

### Changelog
- `2026-02-27` — Proposed by abcdef-sim maintainers.
- `2026-02-27` — Accepted by abcdef-sim maintainers.

---
