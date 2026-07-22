# TwoWaveformDistinguishability.jl

Computational proof of the **quartic distinguishability law** for space-based
gravitational-wave interferometry: the squared noise-weighted distance between
a two-source signal and the best-fit *single*-source template scales as

**D² ≈ (1/16) · K(u) · δ⁴**,

where δ is the parameter separation and K(u) the extrinsic curvature of the
signal manifold. The discernibility boundary follows as
δ_min(u) = (16 ρ²_thr / K(u))^{1/4}, and the 2D "zone of confusion" maps are
the exact intersection of that boundary with the hard physical parameter
bounds (positivity of amplitude/mass/time, |χ| ≤ 1, phase topology ±π).

## File structure

```text
TwoWaveformDistinguishability/
├── Project.toml            # deps, GPU weakdeps + extensions, compat
├── Manifest.toml           # version-controlled — portability guarantee
├── config.toml             # the single source of all run parameters
├── src/
│   ├── TwoWaveformDistinguishability.jl  # top module, exports
│   ├── Hardware.jl         # backend registry; CPU fallback; GPU via extensions
│   ├── Physics.jl          # Robson (2019) noise model; scalar waveform core
│   ├── Detector.jl         # TDI A/E response (fused single-pass projection)
│   ├── Bounds.jl           # physical parameter bounds, deviation boxes, capping
│   ├── Geometry.jl         # tangent basis (MGS), fused directional derivatives
│   ├── Inference.jl        # box-constrained D² optimization; KA loss kernel
│   ├── Config.jl           # validated TOML configuration (hard-fail guardrails)
│   ├── Provenance.jl       # config-hash run IDs, snapshots, safesave, metadata
│   ├── Plotting.jl         # CairoMakie figures; family-wide tick policy
│   └── Orchestrator.jl     # pipeline driver: sweeps, capped mirrored mapping
├── ext/                    # TWDCUDAExt, TWDAMDGPUExt, TWDMetalExt, TWDoneAPIExt
├── scripts/
│   ├── pipeline.jl         # CLI entry point (loads GPU package per config)
│   ├── launch_campaign.jl  # detached launcher via Base.julia_cmd()
│   ├── replot.jl           # regenerate all figures from a run's CSVs
│   └── audit_bounds.jl     # audit a run's maps against the physical bounds
├── test/
│   ├── runtests.jl         # physics validation + A/B regression + E2E
│   ├── Project.toml
│   └── fixtures/legacy/    # committed regression fixtures (legacy-code outputs)
├── benchmarks/             # BenchmarkTools scripts (own environment)
├── docs/                   # Documenter.jl sources
└── data/{logs,outputs}/    # run artifacts (git-ignored)
```

## Environment setup

```julia
using Pkg
Pkg.activate(".")           # from this directory
Pkg.instantiate()
```

`Project.toml` + `Manifest.toml` are authoritative and version-controlled.
GPU support is optional: install the package matching your hardware
(`Pkg.add("CUDA")`, `AMDGPU`, `Metal` or `oneAPI`) and the corresponding
package extension activates automatically; without one, the pipeline runs on
the multi-threaded CPU backend (`[hardware].gpu_backend = "none"` forces
this). GPU runs are pinned to a single task by the pipeline and all GPU
kernel launches are serialized library-wide — concurrent multi-task access
to GPU drivers is unsafe (observed Level Zero segfault) and buys nothing,
since the device serializes kernels anyway.

## Usage

```bash
# foreground run (progress bars on a TTY)
julia --project --threads=auto scripts/pipeline.jl --config config.toml

# detached campaign with ANSI-free logs
julia --project scripts/launch_campaign.jl

# regenerate every figure of a finished run from its CSVs (no recomputation);
# --rho R additionally rescales all maps/threshold markers to a new
# discernibility threshold using the persisted per-direction curvature
julia --project scripts/replot.jl data/outputs/run_<hash> [--rho R]

# audit a run's confusion maps against the physical bounds
julia --project scripts/audit_bounds.jl data/outputs/run_<hash>
```

Everything tunable lives in `config.toml` (grid, physics, Robson-2019 noise
coefficients, mapping resolution/refinement, `[parameter_bounds]`, optimizer,
`[safety]` memory budgets). The configuration is validated up front: unusable
values abort with a descriptive error, suspicious ones warn, and **unknown
keys warn** (typo protection). Each run lands in
`data/outputs/run_<confighash>/` with a config snapshot, `metadata.toml`
(git commit, backend, timings) and a structured, ANSI-free `run.log`; reruns
get suffixed directories and `safesave`-style backups — results are never
overwritten.

## Outputs

- **1D sweeps** (`sweeps/<name>/`): `results.csv` (per-δ D², best-fit
  parameters, convergence diagnostics, active-bound flags, multi-start gain),
  `residual_spectrum.csv`, `sweep_meta.toml` (fitted log-log slope ± stderr,
  the O(δ⁵) correction coefficients c₁/c₂ with a 10%-validity radius,
  optimizer floor level, δ*, amp_ratio), `scaling_plot.{pdf,png}` (log–log
  panel plus a D²_num/D²_theo ratio panel with the correction-fit overlay),
  `residual_plot.{pdf,png}` (d(SNR²)/df and d(D²)/df densities for channels
  A and E; the bottom panel integrates to D²). Sweeps support unequal
  amplitudes (`amp_ratio`, the A_harm law) and opt-in multi-start seeding
  (`n_starts`, seeded by `[pipeline].rng_seed`).
- **2D maps** (`maps/<name>/`): `confusion_contour.csv` (angle, capped
  boundary, `R_Math`/`R_Box`/`Prior_Limited`/`Degenerate` columns, K, g),
  `confusion_zone.{pdf,png}` with the physical prior box drawn and
  prior-limited boundary segments visually distinct from curvature-limited
  ones. K is computed on [0, π) only and mirrored (K(u) is exactly even),
  with adaptive angular refinement near boundary spikes.

## Status of components

| Component | Status |
|---|---|
| Physics / Detector / Geometry / Inference | unit-tested; A/B-locked against committed legacy fixtures |
| Robson (2019) confusion noise (Eq. 14, Table 1) | **fixed** — the pre-2026 campaign ran with an inert confusion term (coefficient transcription bug), i.e. instrumental noise only |
| Box-constrained optimization (`IPNewton`; `lbfgs_box`/`lbfgs` fallbacks) | tested, physical bounds enforced |
| 2D mapping (mirrored, prior-capped, adaptively refined) | tested end-to-end |
| GPU path (KernelAbstractions kernel + package extensions) | kernel verified ≡ CPU loop (value and gradient) on the CPU backend; device execution requires supported GPU hardware |
| Plotting (CairoMakie, no-title/tick-policy compliant) | tested; figures regenerable via `scripts/replot.jl` |
