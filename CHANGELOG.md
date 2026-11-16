# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `[physics].phase_pn_order` selects the post-Newtonian order of the TaylorF2
  phase: 1.5 (the default), 2, 2.5, 3, 3.5, 4 or 4.5. The coefficients are
  those of aligned-spin black holes, complete through 3.5PN including the
  spin–spin tail term of Cho, Porto & Yang (2022) that the usual ready-to-use
  forms omit; 4PN with its spin–orbit tail, next-to-next-to-leading spin–spin
  and quartic-in-spin terms; 4.5PN with its non-spinning term only
  (`Physics.phase_coefficients`, `Physics.PhaseCoefficients`,
  `Physics.PHASE_PN_ORDERS`, `Physics.spin_sigma`; new methods of `pn_phase`
  and `harmonic_phase`). Orders 1.5 and 2 reproduce the earlier phase bit for
  bit, and run identifiers of existing configurations do not change. From 2PN
  on, the spin combination that is flat at 1.5PN is constrained; from 2.5PN on
  for unequal masses also at zero spins (Waveform Physics, §2).
  `configs/pn_orders/` holds overlays of the quickstart configuration and of
  four spin maps of the production configuration at every order.
- `[pipeline].fit_coordinates = "whitened"` fits in the coordinates in which
  the Gauss–Newton Hessian is the identity, from a blocked QR factorization of
  the noise-whitened Jacobian (`Geometry.whitened_jacobian_factor`,
  `Inference.whitening_transform`); the box on the parameters becomes six
  linear constraints. At 2PN order and above the fit in the parameters stalls
  on the weakly constrained spin combination; the whitened fit converges. The
  default remains `"parameters"`.
- `Fitting.validity_radius`, `Fitting.angular_fraction`,
  `Provenance.acquire_run_lock`, `Provenance.record_attempt!`,
  `Orchestrator.run_log_logger`, `Backends.GPU_PACKAGES`.
- Test groups: `Pkg.test(test_args = ["physics"])` runs one group.

### Changed

- The prior-limited and degenerate fractions annotated on the confusion-zone
  maps are fractions of the sampling angle, as the annotation and the
  documentation state. They were fractions of the vertex count, which adaptive
  refinement concentrates near the prior walls: on a coarse map the two differ
  by an order of magnitude, on the production spin map by less than a tenth of
  a percent. Persisted tables and log lines are unchanged.
- `scripts/replot.jl` and `scripts/collect_plots.jl` exit with status 1 when a
  case failed or a run directory does not exist.
- Documentation: the conventions document is cited at its revision 20; the
  instrumental noise terms are written out in the form implemented (Eqs. 10–11
  of the arXiv version of Robson et al., with two frequency-dependent factors
  that the journal version omits); the polarization angle is referred to the
  line-of-nodes vectors of the amplitudes, a quarter turn from the wave-frame
  vectors of the conventions document; the descriptions of the sweeps follow
  the production results; the executable example sweeps separations above the
  optimizer floor. Docstrings and comments are corrected where they had
  drifted from the code, and public physics functions state their units.
- Continuous integration: the formatter check runs as its own workflow with
  the formatter major fixed, the test matrix reads the Julia floor from
  `[compat]`, and the test environment admits JET 0.13.

### Fixed

- The Metal extension declines instead of offering a backend that cannot hold
  `Float64` arrays: `gpu_backend = "auto"` on an Apple GPU now falls back to
  the CPU.
- `run_pipeline(...; run_dir)` refuses a directory that holds the snapshot of
  another configuration.
- `InterruptException` and `OutOfMemoryError` raised inside a work-item task
  end the run, as documented, instead of being recorded as a stage failure.
- A configuration value of the wrong type (a number where a string, a table or
  an array of tables is expected; a Boolean, an infinity or a NaN where a
  number is expected) is a configuration error naming the key, and a finite
  interval for the periodic `coalescence_phase` in `[parameter_bounds]` is
  refused.
- The `clean` mask passed to `scaling_figure` and `scaling_panel!` selects the
  floor-band points; it was ignored in favour of a recomputed rule.

## [2.1.2] - 2026-11-09

### Changed

- Confusion-zone maps are drawn in physical units with the unit on each axis:
  chirp mass in solar masses, coalescence time in seconds, phase in radians,
  luminosity distance in Gpc (`Plotting.display_scale`, from the run's
  `mass_scale`, `time_scale` and `distance_scale`). Computation and persisted
  contour tables stay in the rescaled coordinates; numerical results are
  unchanged.

## [2.1.1] - 2026-10-22

### Changed

- `[compat]` admits KernelAbstractions 0.10 alongside 0.9, in the package
  and in the test and benchmark environments, and DocStringExtensions 0.10
  alongside 0.9. Under KernelAbstractions 0.10 the CPU backend launches
  kernels through the same interface as the GPU backends, so the static
  analysis of the test suite no longer sees the three known kernel-call
  artifacts of 0.9; the suite expects the count of the resolved version.
  Numerical results are unchanged.
- Documentation pages, README and docstrings reworded in plain prose; the
  Scientific Context page is rewritten and the parameter guide is retitled
  Run Parameters. Error and warning messages are punctuated with colons and
  semicolons instead of dashes. No change in behaviour or results.

## [2.1.0] - 2026-10-14

### Added

- `metadata.toml` records `package_version`; `Provenance.package_version`
  returns it and `Provenance.check_package_version` refuses a run directory
  recorded by another version.
- `ParameterBounds` carries a `deviation_cap` per periodic axis, separate
  from the period. The four-argument constructor sets it to half the period,
  as before.

### Changed

- A run is continued (`resume`, `run_dir`, the supervisor) only by the
  package version that produced it. The run identifier still hashes the
  configuration alone; `resolve_run_dir` skips a sibling recorded by another
  version, or by a version before 2.1.0, with a warning and starts a new
  one, and `run_pipeline(...; run_dir)` throws. Sweep checkpoints carry the
  version in their signature.

### Fixed

- The coalescence phase is declared with its true period `4π`: harmonic `k`
  carries `−(k/2)φ_c`, so a `2π` shift reverses the odd harmonics except at
  equal mass. The confusion maps keep the deviation cap `|Δφ_c| ≤ π`, half
  the period of the dominant harmonic; no map or table changes.
- The parameter guide described the mass–spin and seed-binary zones as
  capped at the prior walls; in the production configuration no direction
  of either map is prior-limited.

## [2.0.4] - 2026-10-13

### Changed

- `[compat]` admits CSV.jl 1.1 alongside 0.10, in the package and in the
  test environment. The persisted run tables and the reference fixtures
  hold numeric and Boolean columns only, which the 1.x parser returns with
  unchanged element types, so no read or write site changes; the suite
  passes with CSV.jl 1.1.0 resolved.

## [2.0.3] - 2026-09-27

### Changed

- Figure labels follow the manuscript notation: `D²_opt` and `D²_th` for
  the optimized and predicted distances (legend and ratio strip), `D_*²`
  for the threshold line, `φ_c` for the coalescence phase, "exponent" for
  the fitted log-log slope, `dD²/df` for the residual density.
- Legends are set at the size of the axis labels; in-axis annotations are
  slightly larger.
- Annotations quote values to three significant digits with trailing zeros
  kept (`δ* = 316`, `D²_opt = 30.0`); a leading minus and a colon are set
  tight (`c₂ = −3.4×10⁻⁸`, `Exponent: 3.995`).
- The prior-wall note of a confusion map gives the capped fraction of the
  sampling angle to one decimal and names the vanishing-curvature fraction
  only when it reaches 0.1 %.
- The README and documentation landing figures are re-rendered with these
  labels.

## [2.0.2] - 2026-09-26

### Changed

- The LISA conventions reference is the published science ground segment
  conventions document (Baghi et al. 2026, arXiv:2603.22377) in place of the
  Rosetta Stone draft; equation numbers in the docstrings follow it.

## [2.0.1] - 2026-09-26

### Changed

- The README and documentation landing figures are two sweeps of the
  production configuration (six-dimensional diagonal, chirp-mass /
  coalescence-time) instead of the quickstart sweep and spin map.
- Waveform Physics documentation: the frequency grid is the analysis band,
  the observation window the span of the observation.

## [2.0.0] - 2026-09-25

The numerical results of this version differ from those of every 1.x
release (see Fixed); the public interface is unchanged.

### Fixed

- The Galactic confusion term entered the noise PSD at the wrong level.
  Robson et al. (2019) calibrate their Eq. 14 fit on the two-channel
  sensitivity curve, S_n = P_n/R + S_c (their Eq. 1), whereas the
  instrumental Eq. 12 term is the PSD of one channel, against which the
  explicit channel response is weighted. `analytic_noise_psd` now adds
  R(f)·S_c(f) with the Eq. 9 response R(f) = (3/10)/(1 + 0.6 (f/f★)²), so
  that the per-channel PSD divided by R(f) is the published sensitivity
  curve and the sky- and polarization-averaged A + E signal-to-noise ratio
  coincides with the one computed from it. Below 4 mHz the confusion term
  was over-weighted by up to 3.3. On the shipped base points the squared
  signal-to-noise ratios rise by 9 % (the two light sources) to a factor
  2.9 (the 8×10⁵ M☉ binaries), Fisher norms by factors between 1.0 and 3.0
  depending on the axis, and every D², δ_min and zone extent computed with
  earlier versions changes accordingly; the fitted exponents of the quartic
  law and the run timings do not. Run identifiers hash the configuration,
  not the code, and are unchanged: results of earlier versions under the
  same identifier are told apart by the commit in their `metadata.toml`.

### Added

- `sky_averaged_response(f, arm_length)`: Robson et al. (2019) Eq. 9.
- Closure test of the noise model against the published sensitivity curve
  (Eq. 13 + Eq. 14).

### Changed

- The README and documentation landing figures are regenerated from the
  quickstart run under the corrected noise weighting.

## [1.1.0] - 2026-09-23

### Added

- Perturbative-window exponent of the quartic law (`slope_window`,
  `slope_window_err` in `sweep_meta.toml`): the log–log slope fitted over the
  points admitted to the O(δ⁵) correction fit. The scaling figure quotes it;
  the slope over all clean points is persisted alongside.
- `scripts/replot.jl --refit`: refit slope, window exponent and correction
  coefficients from the persisted results table.
- Panel drawers `scaling_panel!`, `residual_panel!` and `zone_panel!`, which
  draw a figure into a supplied grid position; the single figures are
  one-panel wrappers around them.
- Composite builders `composite_scaling_figure`, `composite_zone_figure` and
  `composite_residual_figure`: panels on a grid with a shared legend; a
  single-column layout draws the panels at full size, wider grids the compact
  variant; optional panel labels (none by default).
- Layout-driven composition from persisted run tables: `composite_figures`
  reads a TOML layout, with the panel loaders `sweep_panel_data`,
  `zone_panel_data` and `residual_panel_data`.
- `scripts/compose_figures.jl` (with `--print-width PT`, which scales every
  PDF to the manuscript's text width) and the example layout
  `configs/figures/quickstart_composites.toml`.
- `save_figure(...; pt_per_unit)` and `canvas_width`.
- README and documentation landing page: the quickstart scaling law and the
  spin-plane zone of confusion as figures (`docs/src/assets/`).

### Changed

- The residual-spectrum legend is one row, the two channel groups side by
  side, in the single figure and in the composites.
- `[compat]` admits CUDA.jl 6 alongside 5 (the H200 campaign ran on the
  CUDACore 6 series).
- The GPU-path agreement quoted in the README is the campaign-wide figure
  (3×10⁻⁴ or better above the optimizer floor).

### Fixed

- Residual-figure axis ranges: the top-panel limits and ticks follow the signal
  means, and the envelopes no longer extend below the frame in either panel.

## [1.0.0] - 2026-09-22

First public release.

### Added

- Frequency-domain inspiral model: TaylorF2 phasing to 1.5PN with spin–orbit
  coupling, (2,2) harmonic plus the 0.5PN amplitude harmonics at unequal
  mass, inspiral taper at the innermost stable orbit.
- Optional finite-observation window (`[physics].observation_window`): every
  harmonic is weighted at its stationary-phase emission time, so that signal
  emitted before the observation starts does not contribute.
- LISA response on analytic orbits in the A and E channel combinations; the
  Robson–Cornish–Liu (2019) instrument and galactic-confusion noise model.
- Signal-manifold geometry: tangent basis, Fisher norm and extrinsic
  curvature K(u) along a direction, by nested forward-mode differentiation.
- Box-constrained fits of a single-source template to two-source data
  (interior-point Newton, L-BFGS fallback), multi-start with reproducible
  streams, optimizer-floor detection, log–log slope and O(δ⁵) correction fits.
- Two-dimensional zones of confusion: Fisher-normalized direction sampling,
  adaptive angular refinement, exact intersection with the physical prior box.
- KernelAbstractions loss kernel with CUDA, AMDGPU, oneAPI and Metal package
  extensions; chunked Hessian evaluation (`hessian_chunk`).
- Validated TOML configuration with base/overlay layering; configuration-hashed
  run directories with config, manifest, hardware and git provenance.
- Supervised runs: heartbeat and CPU-time hang detection, work-item
  checkpoints, continuation of interrupted runs, retry budget
  (`scripts/run_supervised.jl`, `[supervision]`).
- Publication figures (CairoMakie) and their regeneration from persisted
  tables (`scripts/replot.jl`, `scripts/collect_plots.jl`).

[Unreleased]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.1.2...HEAD
[2.1.2]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.1.1...v2.1.2
[2.1.1]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.1.0...v2.1.1
[2.1.0]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.0.4...v2.1.0
[2.0.4]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.0.3...v2.0.4
[2.0.3]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.0.2...v2.0.3
[2.0.2]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.0.1...v2.0.2
[2.0.1]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.0.0...v2.0.1
[2.0.0]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/releases/tag/v1.0.0
