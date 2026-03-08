# raytracing-backed thick lens validation

This document records the repo's external-oracle validation story for wavelength-dependent thick
lens behavior.

## Scope

The comparison uses the shared paraxial observables available in both packages:

- wavelength-dependent effective focal length for a single dispersive thick lens
- Gaussian beam radius after a two-element dispersive doublet
- wall-clock cost for tracing one ray over multiple wavelengths

The comparison does **not** attempt to validate Martinez phase bookkeeping. The external
`raytracing` package is only used here as an oracle for paraxial spatial transfer and Gaussian beam
propagation.

## Setup

Install the optional validation dependencies:

```bash
pip install -e '.[validation]'
```

Run the canonical example:

```bash
python examples/compare_thick_lens_to_raytracing.py
```

That writes:

- `artifacts/physics/thick_lens_similarity.png`
- `artifacts/physics/wavelength_tracking_benchmarks.md`

The legacy plot-only wrapper remains available:

```bash
python scripts/generate_abcd_validation_plot.py
```

## Graphical evidence

The checked-in figure below is generated from the same example code path and shows:

- single-lens effective focal length versus wavelength
- residuals against `raytracing`
- doublet Gaussian beam-radius overlays at 810 nm and 1550 nm
- residuals for the doublet beam profiles

![Thick lens similarity](images/thick_lens_similarity.png)

## Benchmark methodology

- Wavelength counts: `1, 8, 32, 128, 512`
- Warmup runs per case: `2`
- Measured runs per case: `5`
- Metric: median wall-clock time from `time.perf_counter`
- Scenarios:
  - single thick lens, matrix build plus one fixed-ray propagation over `N` wavelengths
  - two-lens dispersive chain, sequential propagation of the same fixed ray over `N` wavelengths

## Sample benchmark table

These timings are machine-dependent and are committed only as one sample run from the maintained
validation environment.

Sample run captured on March 8, 2026 in the `wust-gnlse` conda environment:

| Scenario | Wavelengths | abcdef-sim (ms) | raytracing (ms) | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Single thick lens ray trace | 1 | 0.412 | 0.289 | 0.70x |
| Doublet chain ray trace | 1 | 0.773 | 0.430 | 0.56x |
| Single thick lens ray trace | 8 | 0.759 | 0.825 | 1.09x |
| Doublet chain ray trace | 8 | 1.405 | 0.962 | 0.68x |
| Single thick lens ray trace | 32 | 1.275 | 0.588 | 0.46x |
| Doublet chain ray trace | 32 | 4.935 | 2.090 | 0.42x |
| Single thick lens ray trace | 128 | 7.682 | 2.035 | 0.26x |
| Doublet chain ray trace | 128 | 11.610 | 6.683 | 0.58x |
| Single thick lens ray trace | 512 | 30.412 | 11.226 | 0.37x |
| Doublet chain ray trace | 512 | 70.865 | 28.994 | 0.41x |
