**Title:** Adopt ECO-0001 conventions in abcdef-sim
**ADR ID:** 0007-lite
**Status:** Accepted
**Date:** 2026-02-27

**Context:** `abcdef-sim` is adding physically meaningful stretcher/compressor and propagation behavior that must be interoperable with ecosystem contracts. We need one explicit convention baseline so dispersion signs, grating geometry, units, beam parameters, and FFT transforms are consistent across stages and result surfaces.

**Decision:** Adopt **ECO-0001** as the contract for all `abcdef-sim` internal calculations and declared outputs.

1. **Dispersion understanding and sign behavior**
   - Positive chirp means `dω_inst/dt > 0`.
   - GDD/TOD follow phase-derivative sign conventions (`d²φ/dω²`, `d³φ/dω³`).
   - Dispersion-bearing components (fiber, stretcher/compressor models) must document their sign assumptions and include sign-sensitive tests.

2. **Grating conventions**
   - Use a single documented grating equation/sign convention for incidence angle, diffraction order, and compressor geometry.
   - Any backend-specific coordinate choice must be normalized to the same observable sign behavior at the `abcdef-sim` API boundary.

3. **Internal base units**
   - Internal simulation units are `fs`, `um`, `rad`.
   - Angular frequency is expressed in `rad/fs` and speed of light is `c = 0.299792458 um/fs`.
   - Contracted outputs that expose dimensional values should use explicit unit suffixes.

4. **Beam parameters**
   - Beam/envelope amplitudes use `sqrt(W)` normalization so instantaneous power is `|E|^2` in `W`.
   - Energy computations integrate over timestep scaling (`sum(|E|^2 * dt_fs * 1e-15)`).
   - Beam-radius and fluence/intensity-derived metrics must declare assumptions (e.g., Gaussian-equivalent vs numeric second moment).

5. **FFT conventions**
   - Use NumPy FFT sign convention with explicit timestep scaling.
   - Forward transform: `Ew = dt_s * FFT(Et)`.
   - Inverse transform: `Et = (1/dt_s) * IFFT(Ew)`.
   - Round-trip and Parseval-style checks are required where spectral-domain operators are introduced.

**Consequences:**
- **Positive:** compatible physics semantics across ecosystem simulators and easier cross-repo validation.
- **Tradeoff:** stricter unit/sign declarations increase implementation overhead for new stages.
- **Follow-up:** add test fixtures for dispersion sign, grating orientation, and FFT scaling invariants.

**References:**
- `cpa-architecture/docs/adr/ECO-0001-conventions-units.md`
- `deps/cpa-sim/docs/adr/ADR-0001-conventions-units.md`
