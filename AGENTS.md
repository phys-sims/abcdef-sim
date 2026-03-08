# AGENTS.md

## 1) Mission and Scope

### Mission
- Implement and harden **real optical physics behavior** for ABCD/ABCDEF simulation.
- Treat numerical correctness, physical consistency, and regression safety as first-class delivery requirements.
- Optimize architecture only in service of delivering verifiable physics outcomes.

### Current scope boundary (authoritative)
- In-scope: physics implementation in `src/abcdef_sim/physics/abcd/*` and `src/abcdef_sim/physics/abcdef/*`, integration through pipeline assembly/stages, and test coverage proving behavior.
- Transitional areas may still contain placeholders, but default expectation is to replace them with explicit, tested physics behavior when task scope calls for it.
- Any remaining placeholder must be clearly labeled in code/tests/docs with an explicit limitation and follow-up path.

## 2) Repo Map (High-Value Entry Points)

- Pipeline assembly root:
  - `src/abcdef_sim/pipeline/_assembler.py`
- Pure paraxial transfer physics (ABCD):
  - `src/abcdef_sim/physics/abcd/*`
- Structured dispersion-aware layer (ABCDEF):
  - `src/abcdef_sim/physics/abcdef/*`
- Physics-oriented tests:
  - `tests/physics/*`
- Architecture and decision records:
  - `docs/architecture.md`
  - `docs/adr/INDEX.md`

Use these locations first for impact analysis and for placing new physics tests near relevant implementations.

## 3) Working Rules

1. **Static inspection first, then physics implementation**
   - Read architecture/docs and affected modules before editing.
   - Trace data flow, units, assumptions, and call paths before changing formulas or propagation logic.

2. **Keep changes minimal, explicit, and typed**
   - Prefer focused diffs with clear physical intent.
   - Preserve/extend type hints and validation contracts.
   - Document equations/assumptions in-code when nontrivial.

3. **High-quality testing is mandatory**
   - Every physics behavior change must include or update tests.
   - Prefer layered tests: unit (formula-level), property/invariant tests, and integration/regression tests where applicable.
   - Use tolerances intentionally (`rtol`/`atol`) and justify non-obvious thresholds in test comments.

4. **Tests/docs must track behavior**
   - Update tests and docs in the same PR for runtime/contract/physics behavior changes.
   - Do not leave behavior changes undocumented.

5. **Never bypass invariants**
   - Do not weaken or bypass cache-key determinism/invariants.
   - Do not bypass policy propagation rules through assembly/stage configuration paths.
   - If a requested change appears to conflict with invariants, escalate instead of patching around them.

## 4) Validation Matrix (Pytest markers from README)

Testing is a delivery requirement, not optional cleanup. Before opening a PR, run pre-commit on all files and then select pytest/mypy lanes by risk; run broader lanes before merge for physics-impacting work.

- **Pre-PR hygiene (required for every PR)**:
  - `pre-commit run --all-files`
- **Fast gate** (required baseline for all non-trivial changes):
  - `pytest -m "not slow" -q`
- **Physics lane** (required for any optics/math/propagation change):
  - `pytest -m "physics" -q`
- **Type check lane** (required for typed interface changes and recommended otherwise):
  - `mypy`
- **Full suite** (required for cross-cutting, pipeline, caching, or release-critical changes):
  - `pytest -q`
- **Optional quick physics iteration lane**:
  - `pytest -m "physics and not slow" -q`

Minimum expectation by change type:
- Docs-only: run at least `pre-commit run --all-files` and `pytest -m "not slow" -q` when feasible.
- Non-physics code changes: `pre-commit run --all-files` + `pytest -m "not slow" -q` + `mypy`.
- Physics module/test changes: `pre-commit run --all-files` + `pytest -m "not slow" -q` + `pytest -m "physics" -q` + `mypy`.
- Cross-cutting/pipeline assembly/caching changes: `pre-commit run --all-files` + `mypy` + `pytest -q`.

## 5) PR Expectations

### Title format
- Use imperative, scoped format:
  - `<scope>: <short imperative summary>`
- Examples:
  - `physics/abcd: implement gaussian beam q propagation`
  - `pipeline: wire abcdef dispersion propagation into stage`

### Required PR checklist
- [ ] Scope and constraints reviewed against `docs/architecture.md` and relevant ADRs.
- [ ] Physics assumptions/equations and units are explicit in code/tests/docs.
- [ ] Cache-key invariants and policy propagation rules remain intact.
- [ ] `pre-commit run --all-files` was run before PR creation.
- [ ] Appropriate validation lanes were run (fast gate, physics lane, mypy, and/or full suite).
- [ ] Tests were added/updated for each behavioral physics change.
- [ ] Residual placeholders (if any) are explicitly documented with limitations.

### Required summary sections
Every PR description must include these headings with concise content:
- **What changed**
- **Why**
- **Risk**
- **Validation**

## 6) Escalation and Non-Goals

### Escalate when
- Physics requirements are ambiguous (units, sign conventions, coordinate frames, expected tolerances, or reference model).
- A request conflicts with architecture docs, ADR decisions, cache invariants, or policy propagation guarantees.
- A change requires broad refactoring beyond stated scope.
- Required validation cannot be run or yields inconsistent results that are not understood.

### Non-goals (do not invent)
- Do not invent physical assumptions without documenting and testing them.
- Do not ship numerically fragile implementations without regression coverage.
- Do not bypass failing physics tests by loosening assertions without justification.

When blocked, document the limitation and provide the smallest safe follow-up that preserves correctness.
