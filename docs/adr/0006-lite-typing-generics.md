**Title:** Typing/generics for phys-pipeline wrappers
**ADR ID:** 0006-lite
**Status:** Accepted
**Date:** 2026-02-07

**Context:** `abcdef-sim` uses `phys-pipeline` and needs the pipeline to preserve state types through wrappers (e.g., abcdef-sim stages, pipeline-as-stage). Incorrect variance or overly concrete container types cause type errors and make wrappers brittle.

**Options:**
- **A) Keep non-generic `StageResult` and invariant containers.**
  - Pros: simpler type signatures.
  - Cons: loses state typing, hampers wrappers, forces casts.
- **B) Use variance + generics in the pipeline type system.**
  - Pros: keeps state typing end-to-end; enables safe wrappers and composition.
  - Cons: requires more careful typing.

**Decision:** Use **Option B** with the following typing choices:
- `PipelineStage[S, C]` uses variance: **state is contravariant**, **cfg is covariant**.
- `StageResult[S]` is generic in the state type to preserve the state across stages.
- `SequentialPipeline` is `Generic[S]` and accepts `Sequence[PipelineStage[S, Any]]` (not `list`) to avoid invariance pitfalls.

**Consequences:**
- **Positive:** wrapper stages can accept broader state types and still return specific ones; pipeline-as-stage patterns are type-safe.
- **Future work:** align `phys-pipeline` runtime type checks with these generics.

**References:** docs/architecture.md

---
