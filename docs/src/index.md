# CurvatureDistinguishability

## Project structure

```text
CurvatureDistinguishability/
├── configs/                # validated run scenarios: quickstart (default),
│                           #   production_cpu, production_oneapi
├── src/
│   ├── Backends.jl         # backend registry; CPU fallback; GPU via extensions
│   ├── Physics.jl          # Robson (2019) noise model; scalar waveform core
│   ├── Detector.jl         # TDI A/E response (fused single-pass projection)
│   ├── Bounds.jl           # physical parameter bounds and polar capping
│   ├── Geometry.jl         # tangent basis, fused directional derivatives
│   ├── Inference.jl        # box-constrained D² optimization; KA loss kernel
│   ├── Config.jl           # validated TOML configuration
│   ├── Provenance.jl       # config-hash run IDs, snapshots, metadata
│   ├── Plotting.jl         # CairoMakie figures, family tick policy
│   ├── Orchestrator.jl     # sweeps + capped mirrored mapping driver
│   └── RunFigures.jl       # figure regeneration from persisted run CSVs
├── ext/                    # CUDA / AMDGPU / Metal / oneAPI extensions
├── scripts/                # run_pipeline.jl, launch_run.jl, replot.jl, collect_plots.jl
├── test/                   # physics validation, A/B fixtures, E2E
├── benchmarks/             # BenchmarkTools suite (own environment)
├── data/run_<hash>/        # provenance-stamped pipeline runs (git-ignored)
└── plots/                  # flat regenerable PNG browsing view (git-ignored)
```

This project is a Julia simulation pipeline that validates the theoretical
quartic scaling law (``\delta^4``) for two-waveform distinguishability and
maps the physically capped "zone of confusion" for the LISA mission.

## Theoretical intent

When data contain two closely overlapping gravitational-wave signals with a
small parameter separation ``\delta``, a single-source fit absorbs the linear
and quadratic differences; the unabsorbable residual is governed by the
extrinsic curvature ``K(u)`` of the signal manifold:

```math
D^2 \approx \frac{1}{16} K(u)\, \delta^4,
\qquad
\delta_{\mathrm{min}}(u) = \left(\frac{16\,\rho_{\mathrm{thr}}^2}{K(u)}\right)^{1/4}.
```

The published zone-of-confusion maps are the **exact intersection** of this
mathematical boundary with the hard physical parameter bounds
(``A, \mathcal{M}, t_c \ge 0``, ``|\chi| \le 1``, phase topology ``\pm\pi``) —
along degenerate directions (e.g. the equal-mass spin difference, which the
waveform cannot see) the zone is limited by the prior, not by curvature, and
the figures mark those boundary segments distinctly.

## Usage

```bash
julia --project --threads=auto scripts/run_pipeline.jl --config configs/production_cpu.toml   # foreground
julia --project scripts/launch_run.jl                                # detached
julia --project scripts/replot.jl data/run_<hash>                 # figures from CSVs
julia --project scripts/collect_plots.jl                     # PNG browsing view
```

Every physical and numerical parameter comes from a `configs/` scenario file, which is
validated up front (descriptive hard errors, warnings for suspicious values,
unknown-key typo protection). Runs land in `data/run_<confighash>/`
with a configuration snapshot, provenance metadata (git commit, backend,
timings), a structured ANSI-free `run.log`, and `safesave`-style collision
handling — results are never overwritten.

## Outputs

1. **`scaling_plot.{pdf,png}`** — log–log ``D^2(\delta)`` against the
   ``(1/16)K\delta^4`` prediction with the ``\rho^2`` threshold and
   ``\delta_{\mathrm{min}}`` marked, a shaded optimizer-floor band, the fitted
   log–log slope, and a ``D^2_{\mathrm{num}}/D^2_{\mathrm{theo}}`` ratio panel
   that exposes prefactor agreement and higher-order departures.
2. **`residual_plot.{pdf,png}`** — true density panels
   (``d(\mathrm{SNR}^2)/df`` and ``d(D^2)/df``, channels A and E, log–log)
   whose bottom-panel integral is the ``D^2`` of the scaling law; the per-bin
   noise level is drawn so the sub-noise residual is visible at a glance.
3. **`confusion_zone.{pdf,png}`** — the prior-capped discernibility zone with
   the physical bound box dashed and prior-limited boundary segments drawn
   distinctly from curvature-limited ones.

All figures are regenerable from the persisted CSVs via `scripts/replot.jl`
without recomputing any geometry.
