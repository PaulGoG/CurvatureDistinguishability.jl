# System Architecture

This page is a structural map of the `CurvatureDistinguishability` codebase:
modules, data flow, GPU path and extension points. Where this page and the
code disagree, the code is right and the page needs correcting.

## 1. Objective and theoretical foundation

The pipeline verifies numerically that the squared distance ``D^2`` between
a two-source GW signal and the best-fitting single-source model scales
quartically with the parameter separation ``\delta``. The squared distance
between the two signals themselves is quadratic in ``\delta``, but a
single-source fit absorbs the first-order difference and the tangential part
of the second-order one. The unabsorbable residual lives in the normal space
and is governed by the extrinsic curvature ``K(u)``:

```math
D^2 \approx \frac{1}{16} K(u) \delta^4 .
```

## 2. Modules

- **`Config.jl`**: parses and validates the configuration TOML into an
  immutable `PipelineSettings` (per-section parse helpers feeding a keyword
  constructor; an overlay's `base_config` is merged in first). Guardrails are
  descriptive hard errors for unusable input (bad ranges, duplicate names,
  base points outside the physical bounds), warnings for suspicious values,
  and hard errors on unknown keys (typo protection). Physical parameters live
  only in the TOML; the source carries physical constants (arm length, AU,
  year) and published-fit defaults.
- **`Physics.jl`**: the Robson et al. (2019) noise model, the physical
  constants (arm length, AU, year, Gpc) and the scalar source-frame waveform.
  The noise model is the per-channel instrumental Eq. 12 plus the Eq. 14
  galactic-confusion fit, a two-channel sensitivity-level fit brought to the
  channel level by the Eq. 9 response, with Table-1 coefficients selected by
  observation time (all overridable in `[noise]`). The waveform comprises
  TaylorF2 phasing to the post-Newtonian order of
  `[physics].phase_pn_order`, 1.5 to 4.5 (`phase_coefficients`, `pn_phase`,
  `harmonic_phase`), the 0.5PN harmonic amplitudes
  (`harmonic_amplitudes`), the stationary-phase time and the
  innermost-stable-orbit window. `WaveformParams` carries the
  scenario constants (scales, mass ratio, sky, inclination, polarization,
  constellation phases) and the precomputed response geometry.
- **`Detector.jl`**: the long-wavelength LISA response on the analytic
  Rubbo–Cornish–Poujade orbits: spacecraft positions, Michelson antenna
  patterns combined into the noise-orthogonal A and E channels
  (`channel_patterns`), the orbital Doppler phase and the finite-arm transfer
  roll-off. These are assembled per bin and per harmonic by the scalar core
  `channel_strain_bin`, the single implementation shared by the fused
  broadcast model `channel_strain`, the CPU inference loop and the GPU
  kernel. The identically zero null channel T is opt-in
  (`[physics].include_t_channel`) and exists only for diagnostic
  comparisons.
- **`Residuals.jl`**: residual-spectrum diagnostics of the sweeps, namely the
  decimated `d(SNR²)/df` densities of the two-source data, the best-fit
  single source and the unabsorbed residual (channels A and E, log-uniform
  windows with mean and min/max envelope) and the per-channel residual `D²`
  integrals. Host-side signal processing behind a typed signature; the
  driver only persists its table.
- **`Bounds.jl`**: hard physical parameter bounds (`[parameter_bounds]`),
  deviation-space boxes around a base point (phase periodic with period
  ``4\pi``, deviations capped at ``\pm\pi``), ray–box crossing distances for
  the polar capping, and interior clamping for the constrained optimizer.
- **`Geometry.jl`**: the differential-geometry engine. One ForwardDiff
  Jacobian of the flattened response followed by modified Gram–Schmidt with
  one re-orthogonalization pass gives the noise-weighted orthonormal tangent
  basis. Exactly degenerate directions are dropped: at 1.5PN order the spins
  enter through one scalar ``\beta``, so the basis has rank 5 at every mass
  ratio; from 2PN on the higher-order spin terms constrain it
  ([Waveform Physics](physics.md), §2). Directional value/first/second
  derivatives come from a single fused nested-dual evaluation. Geometry always
  runs on CPU arrays: `ForwardDiff.jacobian` is incompatible with device
  arrays, and the map stage costs minutes, not days.
  `whitened_jacobian_factor` returns the triangular factor of the
  noise-whitened Jacobian by a blocked QR factorization, for the whitened fit.
- **`Fitting.jl`**: the fit statistics shared by the sweep stage and the
  display-time regeneration, namely the quartic-law log-log slope with its
  standard error, the `O(δ⁵)` ratio-correction fit with its validity radius
  (`validity_radius`), optimizer-floor detection on the contiguous
  small-``\delta`` run and the production clean-point rule.
- **`Inference.jl`**: ``D^2`` minimization within the physical bounds. The
  default optimizer is `IPNewton`, an interior-point Newton method using the
  exact ForwardDiff Hessian (fast convergence, low convergence floor);
  `Fminbox(LBFGS)` is the alternative. With
  `[pipeline].fit_coordinates = "whitened"` IPNewton runs in the coordinates
  `z` of `θ = θ₀ + T z`, where `T` (`whitening_transform`) makes the
  Gauss–Newton Hessian the identity on every constrained direction; the box on
  `θ` becomes six linear constraints. This keeps the fit converging when a
  direction is constrained only weakly, as the spins are at phase orders above
  1.5. `T` is computed once, at the start, so the whitened fit relies on a
  start near the optimum, as the sweeps provide; from a distant start it can
  end higher than the fit in the parameters. The loss has two equivalent implementations, tested against each
  other: an allocation-free scalar CPU loop (avoids GC lock contention under
  20+ threads) and a single KernelAbstractions kernel.
- **`Backends.jl`**: a backend probe registry populated by package
  extensions
  (`ext/CurvatureDistinguishability{CUDA,AMDGPU,Metal,oneAPI}Ext.jl`); no
  `isdefined(Main, …)` reflection. Loading e.g. `CUDA` in the session
  registers the probe; `get_best_backend()` returns the first functional
  device or the multi-threaded `CPU()` fallback. The Metal probe declines:
  Apple GPUs have no double-precision type and the device path is double
  precision only. `scripts/run_pipeline.jl` loads the GPU package requested
  by `[hardware].gpu_backend` only if it is installed, and warns otherwise.
- **`Provenance.jl`**: the effective configuration, an overlay deep-merged
  onto its `base_config` (sub-tables recurse, scalars and `[[sweeps]]`/
  `[[maps]]` lists replace, one level only). Run IDs are the SHA-256 of the
  canonical serialization of the identity table, which holds every section
  except the execution sections `[hardware]`, `[safety]`, `[monitoring]` and
  `[supervision]`, so the same physical case with the same numerical method
  has one identifier on every machine. Reruns land in suffixed sibling
  directories; comments, formatting, execution settings, the base/overlay
  split and the wall clock never affect identity. The module also writes
  config snapshots into the run directory (the merged table for overlays),
  `metadata.toml` (git state via DrWatson, Julia version, backend, threads,
  timings) and `safesave`-semantics backups for all output formats. It also
  takes the run-directory locks (`acquire_run_lock`) and records each
  attempt's environment (`record_attempt!`).
- **`Plotting.jl`**: CairoMakie figures under one publication theme
  (Computer Modern, boxed axes, no titles, no minor ticks) and one
  family-wide tick policy: integer power-of-10 log ticks restricted to the
  data range with anchors congruent mod the step; a single per-axis exponent
  for small linear values (never mixed exponents); single-denominator π
  ticks on phase axes. Figures are drawn by panel functions into a supplied
  layout position, the single figures being one-panel wrappers.
- **`ZoneSampling.jl`**: the sampling geometry of the maps: directions in a
  plane whose axes are measured in their Fisher scales, adaptive angular
  refinement where the capped radius of neighbouring directions jumps, the
  vertices where the zone boundary crosses the prior box or passes its
  corners, and the mirroring of the half-circle onto the full circle with
  the prior cap.
- **`Orchestrator.jl`**: the driver. A pre-flight memory estimate against
  `[safety].max_ram_gb` (refuses or downscales concurrency; the GPU branch
  additionally enforces the VRAM budget), bounded-concurrency task pools,
  TTY-gated progress bars (detached runs produce ANSI-free logs) and a
  structured `run.log` via LoggingExtras. Each stage runs under its own
  try/catch: a failing sweep or map is logged with its backtrace and the
  remaining stages continue. Failures are listed in `metadata.toml`; a
  degenerate sweep direction is such a failure, while figures that could not
  be rendered are listed separately and do not fail the stage.
  `run_pipeline` then throws a `PipelineStageError`, which
  `scripts/run_pipeline.jl` turns into exit status 3.
- **`RunFigures.jl`**: display-time figure regeneration from persisted run
  artifacts, the single implementation behind `scripts/replot.jl` and
  `scripts/collect_plots.jl`. Point classification and display refits reuse
  the pipeline's own fit rules (`optimizer_floor`/`above_floor_mask`,
  `loglog_slope`, `ratio_correction_fit`), so run-time and regenerated
  figures cannot diverge. Publication composites are assembled from the
  persisted tables by `RunFigures.composite_figures` from a TOML layout
  (`scripts/compose_figures.jl`, example
  `configs/figures/quickstart_composites.toml`).

## 3. Workflows

### One-dimensional sweeps (`[[sweeps]]`)
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
A sweep direction with an amplitude component is rejected (the amplitude of
the second source is set by `amp_ratio`); the sweep warns when the second
source exits the physical bounds before ``\delta_{\mathrm{max}}``.

### Two-dimensional confusion maps (`[[maps]]`)
In raw parameter coordinates the boundary radius is
``r(\varphi) = (16\rho^2/K_{\mathrm{raw}})^{1/4}``; the Fisher-norm
rescaling cancels exactly, so no division by ``g(u,u)`` occurs. Directions
are sampled uniformly in the Fisher-normalized plane, in which each axis is
measured in its own ``\sigma = g_{ii}^{-1/2}`` at the base point
(`unit_direction`), and mapped to parameter units. The normalization matters
because a plane whose axes differ by four to five orders of magnitude in
``\sigma`` (a coalescence time against a phase for a loud source) is a
needle in parameter units that a uniform angular grid would resolve only
along its axes. The persisted `Angle` is that sampling angle,
`Dir_Cos`/`Dir_Sin` the direction in parameter units, and the prior-limited
fraction is measured in the sampling angle. ``K`` is evaluated on
``[0, \pi)`` only and mirrored (it is exactly even in ``u``), halving the
cost and enforcing the symmetry. Near-singular spikes (quasi-degenerate
directions) are handled by polar prior-capping:
``r_{\mathrm{plot}} = \min(r_{\mathrm{math}}, r_{\mathrm{box}})`` against the
deviation-space physical box, with `R_Math`, `R_Box`, `Prior_Limited` and
`Degenerate` all persisted. Adaptive angular refinement is added where the
capped radius jumps more than `[mapping].neighbor_ratio_tol` between
neighbors. Every vertex of the polygon lies on the boundary of the zone ∩
prior-box intersection.

## 4. Notes for future development

1. Keep the scalar core (`channel_strain_bin` over the `Physics` kernels)
   the only implementation of the physics: the loop, broadcast and kernel paths
   all call them, and the test suite asserts their equivalence.
2. New physical parameters go: configuration TOML → `Config.jl` validation →
   `WaveformParams` field → scalar core. Never a bare kwarg default
   duplicated across modules.
3. The GPU path supports the 2-channel (A, E) configuration; parameters
   cross the kernel boundary as isbits `NTuple`s so `ForwardDiff.Dual`
   gradients compile to device code. The classic failure mode (broadcasting
   with `Ref(p)` over a heap `Vector{Dual}`) is designed out. Under AD the
   kernel uses the lanes layout. Dual arithmetic runs inside the kernel, but
   each scalar lane (value + partials, recursively; 7 for gradients, 49 for
   Hessians) is stored directly into a plain `Float64` matrix and reduced
   with standard per-column sums. Device arrays never carry Dual eltypes
   (some GPU runtimes reject them), and no intermediate lane tuple is
   materialized (large tuples trip `gpu_malloc` on some compilers).
   Two measures are enforced in code, not left to configuration. Kernel
   launches are serialized through a library-wide lock and the orchestrator
   pins a GPU run to a single task, because vendor runtimes are not safe
   under concurrent multi-task access and the device serializes kernels
   anyway. Device output buffers are cached and reused, because allocating
   them per evaluation destabilizes long sessions.
4. Changing `[noise]`, the optimizer, or the mapping algorithm invalidates
   comparisons with earlier runs. The config snapshot plus `metadata.toml`
   in every run directory is the provenance chain.
