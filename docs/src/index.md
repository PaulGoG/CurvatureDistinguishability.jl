# CurvatureDistinguishability

A Julia pipeline that validates the quartic distinguishability law for
overlapping gravitational-wave signals and maps the physically capped
"zone of confusion" for a LISA-like detector.

When data contain two closely overlapping signals with a small parameter
separation ``\delta``, a single-source fit absorbs the first-order difference
and the tangential part of the second-order one. What it cannot absorb is
governed by the extrinsic curvature ``K(u)`` of the signal manifold:

```math
D^2 \approx \frac{1}{16} K(u)\, \delta^4,
\qquad
\delta_{\mathrm{min}}(u) = \left(\frac{16\,\rho_{\mathrm{thr}}^2}{K(u)}\right)^{1/4}.
```

The zone-of-confusion maps are the intersection of this
mathematical boundary with the hard physical parameter bounds
(``D_L, \mathcal{M}, t_c \ge 0``, ``|\chi| \le 1``, phase deviations
within ``\pm\pi``). Along degenerate directions, such as the spin combination
the default 1.5PN phase cannot see, the zone is limited by the prior rather
than by curvature, and the figures mark those boundary segments distinctly.

Both figures below are sweeps of the production configuration
(`configs/production_cpu.toml`, a one-year observation on 1.57 × 10⁶
frequency bins): the six-dimensional diagonal direction of a distant
moderate-mass binary, and the chirp-mass/coalescence-time direction of a
massive aligned-spin binary.

![Quartic law along the six-dimensional diagonal direction](assets/production_scaling_diagonal.png)

![Quartic law along the chirp-mass/coalescence-time direction](assets/production_scaling_mass_time.png)

The repository README is the operational reference (file structure,
environment setup, usage, component status). This documentation covers:

- [System Architecture](architecture.md): module inventory, data flow,
  GPU path and extension points.
- [Scientific Context](science.md): the source-confusion problem and the
  geometric formulation.
- [Waveform Physics](physics.md): the inspiral model (TaylorF2 phase from
  1.5PN, the default, to 4.5PN), constellation response and noise model.
- [Run Parameters](parameters.md): the physical rationale of the
  production configuration's sweeps and maps.
- [Executable Example](generated/quartic_law.md): the law on a miniature
  grid, end to end.
- [Roadmap](roadmap.md): planned extensions with design sketches, and GPU
  operational notes.
- [API Reference](api.md): every public symbol.
- [References](references.md).

The quickest start is `julia --threads=auto scripts/run_pipeline.jl`, which
runs the minutes-scale `configs/quickstart.toml` scenario end to end and
shows the scaling law, the optimizer floor and the prior-limited spin zone.
