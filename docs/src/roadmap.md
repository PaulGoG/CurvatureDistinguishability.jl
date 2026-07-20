# Roadmap — deferred features

Design sketches for features deliberately **not** implemented yet (user
decision, 2026-07), so future work starts warm. None of these block the
production campaign.

## 1. 2PN spin–spin phase term (lifts the exact χ_a degeneracy)

**Why.** At equal masses the current 1.5PN spin–orbit model depends on the
spins only through ``\chi_{\rm eff} = (\chi_1+\chi_2)/2``; the anti-symmetric
combination ``\chi_a`` is an exact flat direction (tangent rank 5), so the
spin-plane confusion zones are prior-limited along it. The 2PN phase carries a
spin1–spin2 coupling ``\sigma \propto \chi_1\chi_2`` which breaks the pure
``\chi_{\rm eff}`` dependence and turns those boundary segments
curvature-limited.

**How.**
- Extend `strain_bin` (src/Physics.jl) with the 2PN term in the bracket:
  ``[1 - 4\beta v^3 + (\ldots)\,\sigma v^4]`` using the standard TaylorF2 2PN
  coefficient (transcribe from a primary reference — e.g. Buonanno, Iyer,
  Ochsner et al. 2009, Table/Eqs. for ``\psi_4`` — with the same care as the
  Robson Table 1 transcription; add the transcription to the test suite).
- Config-gate it: `[physics] spin_spin_2pn = false` (default off) threaded
  through `WaveformParams`, so all existing fixtures/tests and the published
  model stay bit-identical when disabled.
- Consequences when enabled: tangent rank becomes 6 (Gram–Schmidt keeps the
  6th vector), K values change everywhere, spin maps gain genuine structure —
  the paper would need a model-variant note and regenerated figures. The A/B
  fixture tests must assert the *disabled* path only.

## 2. Noise-realization study (distribution of the distance statistic)

**Why.** The pipeline currently validates the noiseless-baseline geometry.
The paper's likelihood interpretation (``\Delta \ln L_{\max} = D^2/2``)
invites the follow-up question: how is ``\hat D^2`` distributed under real
noise, and how sharp is the confusion boundary operationally?

**How.**
- New pipeline stage (config-gated, e.g. `[noise_study] n_realizations`,
  seeded from `[pipeline].rng_seed`): draw Gaussian noise in the whitened
  space (i.i.d. unit normals per real/imaginary frequency-channel component,
  matching the discrete inner product), add to the two-source injection,
  re-optimize per realization.
- Compare the empirical ``\hat D^2`` distribution against the non-central
  ``\chi^2`` expectation with non-centrality ``D^2_{\rm noiseless}`` and
  effective dimension set by the residual normal space; persist per-realization
  results and a QQ-style figure.
- This is the largest deferred item (new science section in the paper).

## 3. Datacenter-GPU campaign notes

The lanes kernel (value + per-partial Float64 stores, see
src/Inference.jl `loss_bins_lanes!`) is vendor-portable by construction; the
consumer-GPU bottleneck is FP64 throughput (1/64 rate on recent consumer
NVIDIA parts — only ~2× end-to-end campaign gain over a good CPU). On
FP64-strong hardware (A100/H100/MI200-class, 10–60× more FP64) the sweep
stage becomes GPU-dominated; the map stage remains CPU by design
(`ForwardDiff.jacobian` on host) and would then dominate — parallelize maps
across CPU cores concurrently with GPU sweeps if the campaign time matters.
Validate with `benchmarks/` + a single-δ solve before committing a queue
allocation.

**Compiler-limit escape hatch (validated on the Meteor Lake iGPU under FP64
emulation):** if a backend's compiler rejects the full 49-lane Hessian kernel
(observed: IGC `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` — the emulation
instruction blow-up exceeds an internal limit; value and 7-lane gradient
kernels build fine), evaluate the Hessian in chunks: passing
`ForwardDiff.HessianConfig(loss, x, ForwardDiff.Chunk{3}())` to
`ForwardDiff.hessian` splits the outer duals so each kernel launch carries
(1+3)·(1+6) = 28 lanes. Measured on the iGPU: chunk-3 Hessian in 0.91 s at
the production grid with machine-epsilon agreement (2e-16) against the CPU
reference — making a full production-size IPNewton solve ≈ 16 s on the
emulated iGPU versus 161 s on 22 CPU threads. If GPU sweeps become a
production path, plumb an optional `hessian_chunk` through
`calculate_numerical_distance`'s `h!`.

## 4. Float32 + compensated summation (only with a validated error model)

FP32 doubles consumer-GPU throughput ~64× for this kernel, but ``D^2`` spans
``10^{-21}``–``10^{-3}`` relative to the signal norm: naive FP32 is unusable.
A viable scheme needs Kahan/Neumaier compensation in the bin reduction *and*
an error model validated against FP64 on the target grid (the fixture harness
is the right tool). Pursue only if a concrete GPU-bound campaign demands it.
