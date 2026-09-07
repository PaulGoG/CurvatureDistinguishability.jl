# CurvatureDistinguishability.jl

[![CI](https://github.com/PaulGoG/CurvatureDistinguishability.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/PaulGoG/CurvatureDistinguishability.jl/actions/workflows/CI.yml)

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
CurvatureDistinguishability/
├── Project.toml            # deps, GPU weakdeps + extensions, compat
├── Manifest.toml           # version-controlled — portability guarantee
├── configs/
│   ├── quickstart.toml     # base: minutes-scale demonstration run (the default)
│   ├── quickstart_gpu.toml # [hardware] overlay: quickstart on the GPU path
│   ├── production_cpu.toml # base: CPU reference campaign configuration
│   ├── production_gpu.toml # [hardware] overlay: portable GPU (gpu_backend = "auto")
│   └── production_oneapi.toml # [hardware] overlay: oneAPI backend, chunked Hessian
├── src/
│   ├── CurvatureDistinguishability.jl  # top module, exports
│   ├── Backends.jl         # backend registry; CPU fallback; GPU via extensions
│   ├── Physics.jl          # Robson (2019) noise model; scalar waveform core
│   ├── Detector.jl         # TDI A/E response (fused single-pass projection)
│   ├── Residuals.jl        # residual-spectrum diagnostics (d(SNR²)/df densities)
│   ├── Bounds.jl           # physical parameter bounds, deviation boxes, capping
│   ├── Geometry.jl         # tangent basis (MGS), fused directional derivatives
│   ├── Inference.jl        # box-constrained D² optimization; KA loss kernel
│   ├── Config.jl           # validated TOML configuration (hard-fail guardrails)
│   ├── Provenance.jl       # config-hash run IDs, snapshots, safesave, metadata
│   ├── Plotting.jl         # CairoMakie figures; family-wide tick policy
│   ├── Orchestrator.jl     # pipeline driver: sweeps, capped mirrored mapping
│   └── RunFigures.jl       # figure regeneration from persisted run CSVs
├── ext/                    # CurvatureDistinguishability{CUDA,AMDGPU,Metal,oneAPI}Ext
├── scripts/
│   ├── run_pipeline.jl     # CLI entry point (loads GPU package per config)
│   ├── launch_run.jl       # detached launcher via Base.julia_cmd()
│   ├── replot.jl           # regenerate all figures from a run's CSVs
│   └── collect_plots.jl    # flat PNG browsing view of one or more runs
├── test/
│   ├── runtests.jl         # physics validation + A/B regression + E2E
│   ├── Project.toml
│   └── fixtures/reference/ # committed golden-value regression fixtures
├── bench/                  # BenchmarkTools scripts (own environment)
├── docs/                   # Documenter.jl sources
├── data/
│   ├── run_<hash>/         # provenance-stamped pipeline runs (git-ignored)
│   └── logs/               # detached-launch console logs (git-ignored)
└── plots/                  # flat regenerable PNG browsing view (git-ignored)
```

## Environment setup

```julia
using Pkg
Pkg.activate(".")           # from this directory
Pkg.instantiate()
```

To use the package as a library from another environment (unregistered;
the repository requires authenticated access while private):

```julia
Pkg.add(url = "https://github.com/PaulGoG/CurvatureDistinguishability.jl")
```

`Project.toml` + `Manifest.toml` are authoritative and version-controlled.
GPU support is optional and never a hard dependency: install the package
matching your hardware (`Pkg.add("CUDA")`, `AMDGPU`, `Metal` or `oneAPI`)
into your default (stacked) environment — the pipeline resolves it through
the load path and the corresponding package extension activates
automatically; without one, the pipeline runs on the multi-threaded CPU
backend (`[hardware].gpu_backend = "none"` forces this).
`configs/production_gpu.toml` (portable, backend auto-detection) and
`configs/production_oneapi.toml` (Intel, `hessian_chunk = 3` — the full
49-lane nested-dual kernel exceeds the Intel iGPU's kernel-argument size
limit) are thin overlays on `configs/production_cpu.toml`: each declares
`base_config = "production_cpu.toml"` and only the `[hardware]` keys that
differ, so the physics is shared by construction. The pipeline merges base
and overlay at load (sub-tables recurse; scalars and `[[sweeps]]`/`[[maps]]`
lists in the overlay replace), hashes the merged configuration into the
run ID and snapshots the merged file into the run directory. GPU runs are
pinned to a single task by the pipeline and all GPU
kernel launches are serialized library-wide — concurrent multi-task access
to GPU drivers is unsafe (observed Level Zero segfault) and buys nothing,
since the device serializes kernels anyway.

## Usage

```bash
# foreground run (progress bars on a TTY); --output-dir overrides data/.
# Without --config the minutes-scale configs/quickstart.toml runs;
# production campaigns are selected explicitly:
julia --project --threads=auto scripts/run_pipeline.jl --config configs/production_cpu.toml

# detached long run with ANSI-free logs (forwards --config/--output-dir)
julia --project scripts/launch_run.jl

# regenerate every figure of a finished run from its CSVs (no recomputation);
# --rho R additionally rescales all maps/threshold markers to a new
# discernibility threshold using the persisted per-direction curvature
julia --project scripts/replot.jl data/run_<hash> [--rho R]

# re-render all figures of one or more runs as a flat PNG browsing view
# (plots/ by default)
julia --project scripts/collect_plots.jl [dest_dir] [run_id ...]

# test suite (unit + physics validation + static QA + end-to-end)
julia --project -e 'using Pkg; Pkg.test()'

# performance benchmarks (own environment; the package resolves by path)
julia --threads=auto bench/run_benchmarks.jl

# documentation build (strict mode; own environment)
julia docs/make.jl
```

Everything tunable lives in the `configs/` scenario files (grid, physics, the full
Robson-2019 noise model — confusion and instrumental parameters alike —
mapping resolution/refinement, `[parameter_bounds]`, optimizer, `[safety]`
memory budgets). `[monitoring].enabled = true` additionally prints an
in-terminal UnicodePlots diagnostic after each completed sweep and map (TTY
sessions only; detached logs stay clean). The configuration is validated up front: unusable
values abort with a descriptive error, suspicious ones warn, and **unknown
keys warn** (typo protection). Each run lands in
`data/run_<confighash>/` with a config snapshot, `metadata.toml`
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
  `confusion_zone.{pdf,png}` with prior-limited boundary segments (wall
  color, active wall values annotated) visually distinct from
  curvature-limited ones; the uncapped mathematical contour continues past
  each wall as a dashed curve. K is computed on [0, π) only and mirrored (K(u) is exactly even),
  with adaptive angular refinement near boundary spikes.

## Status of components

| Component | Status |
|---|---|
| Physics / Detector / Geometry / Inference | unit-tested; A/B-locked against committed reference fixtures |
| Robson (2019) noise model (Eq. 12 instrumental + Eq. 14 confusion, Table 1) | active by default; `[noise].confusion_enabled = false` for instrumental-only studies |
| Box-constrained optimization (`IPNewton`; `lbfgs_box` fallback) | tested, physical bounds enforced |
| 2D mapping (mirrored, prior-capped, adaptively refined) | tested end-to-end |
| GPU path (KernelAbstractions kernel + package extensions) | production-validated on an Intel iGPU with native FP64 (cross-validated ≡ CPU at the 1e-8 level) and on CUDA/ROCm workstation hardware; every GPU failure mode encountered is fixed in code, with the operational hardening summarized in the documentation roadmap (`docs/src/roadmap.md`) |
| Plotting (CairoMakie, no-title/tick-policy compliant) | tested; figures regenerable via `scripts/replot.jl` |
