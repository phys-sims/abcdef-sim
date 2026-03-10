# Treacy Grating Phase Investigation

## Scope

This note summarizes the diagnostics added on `codex/generic-geometry-action-layer` while investigating the persistent large-beam matched double-pass Treacy error plateau:

- GDD relative error: about `3.708%`
- TOD relative error: about `5.575%`

The goal of this work was to determine whether the remaining error comes from free-space action, Treacy fold geometry, local grating transport, grating boundary phase bookkeeping, or a deeper model/reference mismatch.

## What Was Added

Main analytics and tests:

- `src/abcdef_sim/analytics/grating_boundary_debug.py`
- `tests/physics/test_grating_boundary_debug.py`
- `src/abcdef_sim/analytics/__init__.py`

The added diagnostics cover:

- local grating surface-hit state extraction
- Section IV boundary/oracle audits
- Treacy endpoint decomposition variants
- adjacent grating-pair surface-to-surface transport audits
- Treacy alignment and fold-plane checks
- actual-local incidence/matrix audits
- fitted surface-map and generating-function audits

## What Is Ruled Out

### Free-space path/action

Geometric free-space action is already in runtime and the remaining plateau does not behave like a free-space-path error.

### Treacy fold and mirror geometry

The return gratings lie on the correct planes and the fold mirror retroreflects the chief ray exactly. The current Treacy alignment logic appears correct.

### Grating separation as the main source

Sweeping separation from `25 mm` to `400 mm` leaves the relative GDD/TOD errors essentially unchanged. The weighted phase residual scales with separation, but the fitted dispersion mismatch does not.

### Flat versus sampled `A(ω), D(ω)`

Runtime transport already uses sampled `A(ω), D(ω)`. Flattening them destroys TOD but leaves the GDD plateau unchanged, so they are not the source of the `3.7%` asymptote.

### `k_center` versus `k(ω)` by itself

Sampling only the grating-side `k` makes TOD much worse. This is not the missing fix.

### Simple `F*x` family patches

The following all fail as generic fixes:

- upstream `B_pre` probes
- surface-hit `x` substitutions
- incoming-state local-kick substitutions
- paired-grating `F^2` generalizations
- constrained slope-only substitutions
- naive actual-local `F` substitutions

These are useful diagnostics, but not valid physics closures.

## What Was Confirmed

### Section IV closure

In the current analytics harness, the corrected quadratic Section IV oracle closes as:

- exact local `phi3`
- plus exact global endpoint term

This means the Section IV mismatch was partly an oracle/transcription issue, not just a runtime phase bug.

### Later-hit incidence drift is real

Later Treacy grating hits see substantial frequency-dependent actual incidence drift. For the baseline large-beam matched case:

- `g2` configured `41.485 deg`, mean actual `40.198 deg`, RMS mismatch `8.513 deg`
- `g2_return` configured `35.0 deg`, mean actual `33.923 deg`, RMS mismatch `1.818 deg`
- `g1_return` configured `41.485 deg`, mean actual `41.615 deg`, RMS mismatch `6.999 deg`

### The later-hit surface map is still thin in configured local coordinates

Numerically fitted surface maps from the traced local hit states remain effectively thin:

- `B ≈ 0`
- `E ≈ 0`

and they reconstruct surface `x` extremely well on later hits.

This means the remaining Treacy problem is not simply that the later grating encounters stop being local thin maps.

## What Failed

### Actual-local matrix substitution

Rebuilding grating phase from traced actual-local coefficients is catastrophic. It does not reduce the plateau.

### Surface-anchored transport bookkeeping

Replacing phase-side `x_after` with surface-anchored inter-hit states improves some geometric consistency metrics but does not move the GDD plateau.

### Fitted surface-map generating-function oracle

Using the fitted local surface maps directly to build a generating-function action is much worse than the current runtime when compared against Treacy analytic GDD/TOD.

For the baseline large-beam matched case:

- current runtime: GDD error `0.03708`, TOD error `0.05575`
- fitted local generator + global endpoint: GDD error `3.53739`, TOD error `1.72222`
- fitted local generator + anchored pair kernels + global endpoint: GDD error `5.37230`, TOD error `0.89875`

This strongly suggests the remaining discrepancy is not fixed by simply replacing the current local grating law with the exact traced local surface map.

## Current Interpretation

The remaining `3.7%` GDD plateau now looks less like a basic coding bug in:

- free-space action
- Treacy mirror geometry
- local grating transport
- or local surface-hit extraction

and more like a mismatch between:

- the reduced scalar/canonical phase model being used for the folded multi-hit Treacy system
- and the analytic/reference model used for comparison

In short:

- the local hit geometry is richer than the current simplified phase law
- but direct exact-local replacement also does not recover the Treacy analytic target
- so the likely remaining issue is the reduction from exact traced surface-state transport to the scalar dispersion model, or a genuine model/reference limit

## Recommended Next Steps

1. Build an independent folded-geometry Treacy reference from exact traced surface states and compare it directly against the current runtime and against the standard Treacy analytic formulas.
2. Check whether the `3.7%` plateau survives when GDD/TOD are extracted from local derivatives around `ω0` rather than only from the current weighted global polynomial fit.
3. If both agree with the current plateau, treat the residual as a likely model/reference limit rather than continuing to tune local grating patches.
