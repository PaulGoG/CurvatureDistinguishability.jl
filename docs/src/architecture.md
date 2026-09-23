# System Architecture

**Purpose:** exact structural map of the `CurvatureDistinguishability`
codebase for collaborators and AI agents. Statements here are kept in sync
with the code; when they disagree, the code wins and this page is the bug.

## 1. Objective and theoretical foundation

The pipeline computationally proves that the squared distance ``D^2`` between
a two-source GW signal and the best-fitting single-source model scales
quartically with the parameter separation ``\delta`` — not quadratically as a
linear Fisher analysis suggests — because the single-source manifold flexes
to absorb the ``\mathcal{O}(\delta)`` and ``\mathcal{O}(\delta^2)``
differences. The unabsorbable residual lives in the normal space and is
governed by the extrinsic curvature ``K(u)``:

```math
D^2 \approx \frac{1}{16} K(u) \delta^4 .
```

## 2. Modules

- **`Config.jl`** — parses and validates the configuration TOML into an immutable
  `PipelineSettings` (per-section parse helpers feeding a keyword
  constructor; an overlay's `base_config` is merged in first). Guardrails: descriptive hard errors for unusable input
  (bad ranges, duplicate names, base points outside the physical bounds),
  warnings for suspicious values, and hard errors on **unknown keys** (typo
  protection). Physical parameters live *only* in the TOML; the source
  carries physical constants (arm length, AU, year) and published-fit
  defaults.
- **`Physics.jl`** — the Robson et al. (2019) noise model: instrumental
  Eq. 12 plus the Eq. 14 galactic-confusion fit with Table-1 coefficients
  selected by observation time (all overridable in `[noise]`); the physical
  constants (arm length, AU, year, Gpc); and the scalar source-frame
  waveform: TaylorF2 phasing to 1.5PN (`pn_phase`, `harmonic_phase`,
  `spin_beta`), the 0.5PN harmonic amplitudes (`harmonic_amplitudes`), the
  stationary-phase time and the innermost-stable-orbit window. `WaveformParams`
  carries the scenario constants (scales, mass ratio, sky, inclination,
  polarization, constellation phases) and the precomputed response geometry.
- **`Detector.jl`** — the long-wavelength LISA response on the analytic
  Rubbo–Cornish–Poujade orbits: spacecraft positions, Michelson antenna
  patterns combined into the noise-orthogonal A and E channels
  (`channel_patterns`), the orbital Doppler phase and the finite-arm transfer
  roll-off, assembled per bin and per harmonic by the scalar core
  `channel_strain_bin` — the **single implementation** shared by the fused
  broadcast model `channel_strain`, the CPU inference loop and the GPU
  kernel. The identically zero null channel T is opt-in
  (`[physics].include_t_channel`) and exists only for diagnostic
  comparisons — it adds dead compute.
- **`Residuals.jl`** — residual-spectrum diagnostics of the sweeps: the
  decimated `d(SNR²)/df` densities of the two-source data, the best-fit
  single source and the unabsorbed residual (channels A and E, log-uniform
  windows with rms and min/max envelope) and the per-channel residual `D²`
  integrals. Host-side signal processing behind a typed signature; the
  driver only persists its table.
- **`Bounds.jl`** — hard physical parameter bounds (`[parameter_bounds]`),
  deviation-space boxes around a base point (phase treated topologically,
  ``\pm\pi``), ray–box crossing distances for the polar capping, and
  interior clamping for the constrained optimizer.
- **`Geometry.jl`** — the differential-geometry engine. One ForwardDiff
  Jacobian of the flattened response + modified Gram–Schmidt gives the
  noise-weighted orthonormal tangent basis (exactly degenerate directions
  are dropped — the 1.5PN spins enter through one scalar ``\beta``, so the
  basis has rank 5 at every mass ratio). Directional value/first/second derivatives come
  from a **single fused nested-dual evaluation**. Geometry always runs on
  CPU arrays: `ForwardDiff.jacobian` is incompatible with device arrays, and
  the map stage costs minutes, not days.
- **`Fitting.jl`** — the fit statistics shared by the sweep stage and the
  display-time regeneration: the quartic-law log-log slope with its
  standard error, the `O(δ⁵)` ratio-correction fit, optimizer-floor
  detection on the contiguous small-``\delta`` run and the production
  clean-point rule.
- **`Inference.jl`** — ``D^2`` minimization within the physical bounds.
  Optimizers: `IPNewton` (default; interior-point Newton using the exact
  ForwardDiff Hessian — fast convergence, low convergence floor),
  `Fminbox(LBFGS)`. The loss has two equivalent implementations, tested against
  each other: an allocation-free scalar CPU loop (avoids GC lock contention
  under 20+ threads) and a single KernelAbstractions kernel.
- **`Backends.jl`** — a backend probe registry populated by **package
  extensions** (`ext/CurvatureDistinguishability{CUDA,AMDGPU,Metal,oneAPI}Ext.jl`); no
  `isdefined(Main, …)` reflection. Loading e.g. `CUDA` in the session
  registers the probe; `get_best_backend()` returns the first functional
  device or the multi-threaded `CPU()` fallback. `scripts/run_pipeline.jl` loads
  the GPU package requested by `[hardware].gpu_backend` only if it is
  actually installed, with loud diagnostics.
- **`Provenance.jl`** — the effective configuration (an overlay deep-merged
  onto its `base_config`: sub-tables recurse, scalars and `[[sweeps]]`/
  `[[maps]]` lists replace, one level only), run IDs from the SHA-256 of the
  canonical serialization of its identity table — every section except the
  execution sections `[hardware]`, `[safety]` and `[monitoring]` — so the
  same physical case with the same numerical method has one identifier on
  every machine (reruns land in suffixed sibling directories; comments,
  formatting, execution settings, the base/overlay split and the wall clock
  never affect identity), config snapshots into the run directory (the
  merged table for overlays),
  `metadata.toml` (git state via DrWatson, Julia version, backend, threads,
  timings), and `safesave`-semantics backups for all output formats.
- **`Plotting.jl`** — CairoMakie figures under one publication theme
  (Computer Modern, boxed axes, no titles, no minor ticks) and one
  family-wide tick policy: integer power-of-10 log ticks restricted to the
  data range with anchors congruent mod the step; a single per-axis exponent
  for small linear values (never mixed exponents); single-denominator π
  ticks on phase axes. Figures are drawn by panel functions into a supplied
  layout position, the single figures being one-panel wrappers.
- **`Orchestrator.jl`** — the driver. Pre-flight memory estimate against
  `[safety].max_ram_gb` (refuses or downscales concurrency; the GPU branch
  additionally enforces the VRAM budget), bounded-concurrency task pools,
  TTY-gated progress bars (detached runs produce ANSI-free logs), a
  structured `run.log` via LoggingExtras, and **per-stage try/catch**: a
  failing sweep or map is logged with its backtrace and the remaining stages
  continue; failures are listed in `metadata.toml` (a degenerate sweep direction is such a failure; figures that could not be rendered are listed separately and do not fail the stage), and `run_pipeline` then
  throws a `PipelineStageError`, which `scripts/run_pipeline.jl` turns into
  exit status 3.
- **`RunFigures.jl`** — display-time figure regeneration from persisted run
  artifacts, the single implementation behind `scripts/replot.jl` and
  `scripts/collect_plots.jl`: point classification and display refits reuse
  the pipeline's own fit rules (`optimizer_floor`/`above_floor_mask`,
  `loglog_slope`, `ratio_correction_fit`), so run-time and regenerated
  figures cannot diverge. Publication composites are assembled from the
  persisted tables by `RunFigures.composite_figures` from a TOML layout
  (`scripts/compose_figures.jl`, example
  `configs/figures/quickstart_composites.toml`).

## 3. The workflows

### 1D sweeps (`[[sweeps]]`)
Two identical-amplitude sources separated by ``\delta`` along a
Fisher-normalized direction; a box-constrained single-source fit yields
``D^2_{\mathrm{num}}``. Persisted per ``\delta``: best-fit parameters,
convergence flag, iterations, gradient norm, active-bound flag. The log-log
slope is fitted over an automatically detected clean window and stored in
`sweep_meta.toml` together with the optimizer floor level. The exponent
quoted in the scaling figures is fitted over the perturbative window of the
``O(\delta^5)`` correction fit (`slope_window` in `sweep_meta.toml`), with the
slope over all clean points (`slope`) persisted alongside; display-time
regeneration recomputes the window exponent for runs that predate the key.
Guardrails warn
when the sweep direction has an amplitude component (violating the
equal-amplitude assumption of the law) or when the second source exits the
physical bounds before ``\delta_{\mathrm{max}}``.

### 2D confusion maps (`[[maps]]`)
In raw parameter coordinates the boundary radius is
``r(\varphi) = (16\rho^2/K_{\mathrm{raw}})^{1/4}`` — the Fisher-norm
rescaling cancels exactly, so no division by ``g(u,u)`` occurs. Directions
are sampled uniformly in the **Fisher-normalized plane** — each axis
measured in its own ``\sigma = g_{ii}^{-1/2}`` at the base point
(`unit_direction`) — and mapped to parameter units, because a plane whose
axes differ by five orders of magnitude in ``\sigma`` (a coalescence time
against a phase for a loud source) is a needle in parameter units that a
uniform angular grid would resolve only along its axes; the persisted
`Angle` is that sampling angle, `Dir_Cos`/`Dir_Sin` the direction in
parameter units, and the prior-limited fraction is measured in the sampling
angle. ``K`` is evaluated on ``[0, \pi)`` only and mirrored (it is exactly
even in ``u``), halving the cost and enforcing the theorem-level symmetry. Near-singular
spikes (quasi-degenerate directions) are handled by **polar prior-capping**:
``r_{\mathrm{plot}} = \min(r_{\mathrm{math}}, r_{\mathrm{box}})`` against the
deviation-space physical box, with `R_Math`, `R_Box`, `Prior_Limited` and
`Degenerate` all persisted, plus **adaptive angular refinement** where the
capped radius jumps more than `[mapping].neighbor_ratio_tol` between
neighbors. The polygon is therefore the exact zone ∩ prior-box intersection.

## 4. Notes for future development

1. Keep the scalar core (`channel_strain_bin` over the `Physics` kernels)
   the single source of physics truth — the loop, broadcast and kernel paths all call
   them, and the test suite asserts their equivalence.
2. New physical parameters go: configuration TOML → `Config.jl` validation →
   `WaveformParams` field → scalar core. Never a bare kwarg default
   duplicated across modules.
3. The GPU path supports the 2-channel (A, E) configuration; parameters
   cross the kernel boundary as isbits `NTuple`s so `ForwardDiff.Dual`
   gradients compile to device code. The classic failure mode (broadcasting
   with `Ref(p)` over a heap `Vector{Dual}`) is designed out. Under AD the
   kernel uses the **lanes layout**: Dual arithmetic runs inside the kernel,
   but each scalar lane (value + partials, recursively — 7 for gradients,
   49 for Hessians) is stored directly into a plain `Float64` matrix and
   reduced with standard per-column sums; device arrays never carry Dual
   eltypes (some GPU runtimes reject them), and no intermediate lane tuple
   is materialized (large tuples trip `gpu_malloc` on some compilers).
   Two measures are enforced in code, not left to configuration: kernel
   launches are serialized through a library-wide lock and the orchestrator
   pins a GPU run to a single task (vendor runtimes are not safe under
   concurrent multi-task access, and the device serializes kernels anyway),
   and device output buffers are cached and reused (allocating them per
   evaluation destabilises long sessions).
4. Changing `[noise]`, the optimizer, or the mapping algorithm invalidates
   comparisons with earlier runs — the config snapshot plus `metadata.toml`
   in every run directory is the provenance chain; rely on it.
