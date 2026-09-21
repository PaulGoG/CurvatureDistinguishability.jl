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
├── Project.toml        # dependencies, GPU weak dependencies, [compat]
├── activate.jl         # activates and instantiates the package environment
├── configs/            # TOML scenarios: quickstart*, production_*, campaigns/
├── src/                # library: physics, geometry, inference, orchestration, plotting
├── ext/                # GPU package extensions (CUDA, AMDGPU, Metal, oneAPI)
├── scripts/            # entry points: run_pipeline, run_supervised, launch_run, replot, collect_plots
├── test/               # unit, physics-validation, static-QA and end-to-end tests
├── bench/              # BenchmarkTools suite (own environment)
├── docs/               # Documenter.jl sources (own environment)
├── data/               # run outputs, one run_<hash>/ per run (git-ignored)
└── plots/              # flat PNG browsing view (git-ignored)
```

The full annotated tree is in [Full file tree](#full-file-tree) below.

## Environment setup

Julia ≥ 1.12 (`[compat]`), installed through `juliaup`; development and
verification run on Julia 1.13. `Manifest.toml` is not version-controlled:
each environment resolves on first activation, and every run directory
receives a copy of the resolved manifest for provenance. Each environment
ships a pure-Julia activation script that activates and instantiates it
silently; every entry point includes the script of its environment, so no
`--project` flag is needed, and `julia -i` on one opens a REPL in it:

```bash
julia -i activate.jl        # package environment
julia -i docs/activate.jl   # documentation environment (package by path)
julia -i bench/activate.jl  # benchmark environment (package by path)
julia -i test/activate.jl   # test sandbox through TestEnv.jl (installed into a shared environment on first use)
```

To use the package as a library from another environment (unregistered):

```julia
Pkg.add(url = "https://github.com/PaulGoG/CurvatureDistinguishability.jl")
```

GPU support is optional: install the vendor package matching the hardware
(`CUDA`, `AMDGPU`, `Metal` or `oneAPI`) into the default (stacked)
environment; the pipeline loads it according to `[hardware].gpu_backend` and
the matching package extension activates. Without one the pipeline runs on
the multi-threaded CPU backend (`gpu_backend = "none"` forces this).
`[hardware].require_gpu = true`, set in every shipped GPU overlay, aborts a
run before any output when no functional GPU backend resolves.
Configuration layering, run identity and the execution defaults are
described in the documentation page `docs/src/parameters.md`.

## Entry points

```bash
# main pipeline, foreground; without --config the minutes-scale configs/quickstart.toml runs
julia --threads=auto scripts/run_pipeline.jl --config configs/production_cpu.toml

# supervised run(s): hang detection, continuation in a new process, retry budget;
# re-running the same command continues what is unfinished
julia scripts/run_supervised.jl configs/production_gpu.toml configs/production_cpu.toml

# the same, detached from the terminal (console output in data/logs/)
julia scripts/launch_run.jl configs/production_gpu.toml configs/production_cpu.toml

# continue an interrupted unsupervised run instead of starting run_<id>_rN
julia --threads=auto scripts/run_pipeline.jl --config configs/production_cpu.toml --resume

# regenerate every figure of a finished run from its CSVs; --rho R rescales maps and
# threshold markers to another discernibility threshold without recomputation
julia scripts/replot.jl data/run_<hash> [--rho R]

# render all figures of one or more runs as a flat PNG view (plots/ by default)
julia scripts/collect_plots.jl [dest_dir] [run_id ...]

# test suite (unit, physics validation, static QA, end-to-end)
julia -e 'include("activate.jl"); Pkg.test()'

# benchmarks (own environment; the package resolves by path)
julia --threads=auto bench/run_benchmarks.jl

# documentation build, strict mode (own environment)
julia docs/make.jl
```

Everything tunable lives in the `configs/` scenario files (grid, physics, the full
Robson-2019 noise model — confusion and instrumental parameters alike —
mapping resolution/refinement, `[parameter_bounds]`, optimizer, `[safety]`
memory budgets). `[monitoring].enabled = true` additionally prints an
in-terminal UnicodePlots diagnostic after each completed sweep and map (TTY
sessions only; detached logs stay clean). The configuration is validated up front: unusable
values and **unknown keys** abort with an error naming the offending key,
suspicious values warn. Each run lands in
`data/run_<confighash>/` with a config snapshot, `metadata.toml`
(git commit, backend, timings), `hardware.txt`, a copy of the resolved
`Manifest.toml`, per-sweep work-item checkpoints (`checkpoint.csv`) and a
structured, ANSI-free `run.log`; reruns
get suffixed directories and `safesave`-style backups — results are never
overwritten. `scripts/run_pipeline.jl` exits with status 0 only when every
stage completed, 2 on a configuration error and 3 when one or more stages
failed (the remaining stages still run and the failures are listed in
`metadata.toml`).

## Outputs

- **1D sweeps** (`sweeps/<name>/`): `results.csv` (per-δ D², best-fit
  parameters, convergence diagnostics, active-bound flags, multi-start gain),
  `residual_spectrum.csv`, `sweep_meta.toml` (fitted log-log slope ± stderr,
  the O(δ⁵) correction coefficients c₁/c₂ — fitted on the points within
  `correction_fit_max_departure` of the law — with a 10%-validity radius,
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
  each wall as a dashed curve. Directions are sampled uniformly in the
  Fisher-normalized plane (each axis in its own σ) so that strongly
  anisotropic planes are resolved uniformly; K is computed on [0, π) only
  and mirrored (K(u) is exactly even), with adaptive angular refinement
  near boundary spikes.

## Status of components

| Component | Status |
|---|---|
| Physics / Detector / Geometry / Inference | unit-tested; A/B-locked against committed reference fixtures |
| Robson (2019) noise model (Eq. 12 instrumental + Eq. 14 confusion, Table 1) | active by default; `[noise].confusion_enabled = false` for instrumental-only studies |
| Box-constrained optimization (`IPNewton`; `lbfgs_box` fallback) | tested, physical bounds enforced |
| 2D mapping (mirrored, prior-capped, adaptively refined) | tested end-to-end |
| GPU path (KernelAbstractions kernel + package extensions) | validated against the CPU reference on CUDA, ROCm and oneAPI hardware (agreement ≤ 5×10⁻⁵ relative above the optimizer floor); see Known limitations |
| Plotting (CairoMakie, no-title/tick-policy compliant) | tested; figures regenerable via `scripts/replot.jl` |

## Known limitations

- ROCm 7.1 on gfx1100/gfx1101 (Radeon PRO W7900, RX 7700 XT): a kernel
  dispatch intermittently never completes; the process stays alive at
  near-zero CPU load with no error. Unresolved upstream; run long GPU
  campaigns on these devices under `scripts/run_supervised.jl`, which ends
  the stalled process and continues the run.
- Intel integrated GPUs driven by i915: the kernel resets the compute
  context when a single launch exceeds the engine preempt timeout (7.5 s by
  default). `hessian_chunk = 2` keeps launches well below it on the shipped
  grids; `hessian_chunk = 3` is marginal on the production grid and fails on
  longer grids, and `hessian_chunk = 0` exceeds the device's 2048-byte
  kernel-argument limit.
- Hosts with an integrated GPU beside a discrete one: restrict device
  visibility (`CUDA_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`/
  `HIP_VISIBLE_DEVICES`, `ZE_AFFINITY_MASK`) so the discrete device is
  selected; the `backend` line of `metadata.toml` names the device in use.
- The signal fills the frequency band whatever its emission time unless
  `[physics].observation_window = true`; for light systems the band below
  the frequency radiated at `t = 0` then holds decades of inspiral from
  before the observation (documentation, Waveform Physics, Finite observation).

## How to cite

Citation metadata is in `CITATION.cff`; a BibTeX entry:

```bibtex
@software{Gogita_CurvatureDistinguishability,
  author  = {Gogîță, Paul-Adrian},
  title   = {CurvatureDistinguishability.jl},
  version = {1.0.0-DEV},
  year    = {2026},
  url     = {https://github.com/PaulGoG/CurvatureDistinguishability.jl}
}
```

## Full file tree

<details>
<summary>Annotated tree</summary>

```text
CurvatureDistinguishability/
├── Project.toml            # deps, GPU weakdeps + extensions, compat
├── activate.jl             # silent activation of the package environment (included by every script)
├── configs/
│   ├── quickstart.toml     # base: minutes-scale demonstration run (the default)
│   ├── quickstart_gpu.toml # [hardware] overlay: quickstart on the GPU path
│   ├── production_cpu.toml # base: CPU reference campaign configuration
│   ├── production_gpu.toml # [hardware] overlay: GPU baseline (auto backend, chunk 2)
│   ├── quickstart_maps.toml   # overlay: nine-plane map verification on the quickstart grid
│   └── campaigns/          # multi-host campaign plans: whole-run and per-sweep chunk variants
├── src/
│   ├── CurvatureDistinguishability.jl  # top module, exports
│   ├── Backends.jl         # backend registry; CPU fallback; GPU via extensions
│   ├── Physics.jl          # Robson (2019) noise model; 1.5PN inspiral waveform core
│   ├── Detector.jl         # LISA A/E response on the analytic orbits (fused scalar core)
│   ├── Residuals.jl        # residual-spectrum diagnostics (d(SNR²)/df densities)
│   ├── Bounds.jl           # physical parameter bounds, deviation boxes, capping
│   ├── Geometry.jl         # tangent basis (MGS), fused directional derivatives
│   ├── Fitting.jl          # shared fit statistics: slope, O(δ⁵) fit, floor rule
│   ├── Inference.jl        # box-constrained D² optimization; KA loss kernel
│   ├── Config.jl           # validated TOML configuration (hard-fail guardrails)
│   ├── Provenance.jl       # config-hash run IDs, snapshots, safesave, metadata
│   ├── Plotting.jl         # CairoMakie figures; family-wide tick policy
│   ├── Heartbeat.jl        # worker liveness counter and heartbeat file
│   ├── Supervision.jl      # process watchdog: CPU/progress hang detection, retry budget
│   ├── Checkpoint.jl       # durable per-separation sweep checkpoints
│   ├── Orchestrator.jl     # pipeline driver: sweeps, capped mirrored mapping
│   ├── Campaign.jl         # supervised execution of pipeline configurations
│   └── RunFigures.jl       # figure regeneration from persisted run CSVs
├── ext/                    # CurvatureDistinguishability{CUDA,AMDGPU,Metal,oneAPI}Ext
├── scripts/
│   ├── run_pipeline.jl     # CLI entry point (loads GPU package per config)
│   ├── run_supervised.jl   # supervisor entry point (one or more configurations)
│   ├── launch_run.jl       # detached launcher of run_supervised.jl
│   ├── replot.jl           # regenerate all figures from a run's CSVs
│   └── collect_plots.jl    # flat PNG browsing view of one or more runs
├── test/
│   ├── runtests.jl         # physics validation + A/B regression + E2E
│   ├── supervision_tests.jl  # watchdog, checkpoint and continuation tests
│   ├── Project.toml
│   ├── activate.jl         # interactive test sandbox via TestEnv.jl
│   └── fixtures/
│       ├── generate_reference.jl  # regenerates the golden values from the current model
│       └── reference/      # committed golden-value regression fixtures
├── bench/                  # BenchmarkTools scripts (own environment: activate.jl)
├── docs/                   # Documenter.jl sources (own environment: activate.jl, make.jl, src/)
├── CHANGELOG.md
├── CITATION.cff
├── data/
│   ├── run_<hash>/         # provenance-stamped pipeline runs (git-ignored)
│   └── logs/               # detached-launch console logs (git-ignored)
└── plots/                  # flat regenerable PNG browsing view (git-ignored)
```

</details>
