# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v1.1.0...v2.0.0
[1.1.0]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/PaulGoG/CurvatureDistinguishability.jl/releases/tag/v1.0.0
