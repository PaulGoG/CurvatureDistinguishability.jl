# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Perturbative-window exponent of the quartic law (`slope_window`,
  `slope_window_err` in `sweep_meta.toml`): the log–log slope fitted over the
  points admitted to the O(δ⁵) correction fit. The scaling figure quotes it;
  the slope over all clean points is persisted alongside.
- `scripts/replot.jl --refit`: refit slope, window exponent and correction
  coefficients from the persisted results table.

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

[Unreleased]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/releases/tag/v1.0.0
