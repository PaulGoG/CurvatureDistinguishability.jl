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
Validate with `bench/` + a single-δ solve before committing a queue
allocation.

**Production-hardening now enforced in code (2026-07-21 lessons):** GPU
kernel launches are serialized through a library-wide lock and GPU runs are
pinned to one task by `plan_resources` (concurrent multi-task Level Zero
access segfaulted); device output buffers are cached across evaluations
(per-call allocation churn triggered a long-session oneAPI
"freed reference" failure — the cache reduces device allocations by ~10³;
if the driver bug still bites in very long sessions, split the campaign
into shorter per-sweep processes, which the per-stage guardrails and
config-hash run directories make lossless).

**Compiler-limit fallback (implemented; validated on the Meteor Lake iGPU
under FP64 emulation):** if a backend's compiler rejects the full 49-lane
Hessian kernel (observed: IGC `ZE_RESULT_ERROR_MODULE_BUILD_FAILURE` — the
emulation instruction blow-up exceeds an internal limit; value and 7-lane
gradient kernels build fine), the Hessian is evaluated in chunks: the
config key `[hardware].hessian_chunk` (0 = full chunk) threads
`ForwardDiff.HessianConfig(loss, x, ForwardDiff.Chunk{c}())` through
`calculate_numerical_distance`'s `h!`, so each kernel launch carries
(1+c)·(1+6) lanes. Measured on the iGPU at chunk 3: Hessian in 0.91 s at
the production grid with machine-epsilon agreement (2e-16) against the CPU
reference — a full production-size IPNewton solve ≈ 16 s on the emulated
iGPU versus 161 s on 22 CPU threads. The chunked path is exercised in the
test suite.

## 4. Float32 + compensated summation (only with a validated error model)

FP32 doubles consumer-GPU throughput ~64× for this kernel, but ``D^2`` spans
``10^{-21}``–``10^{-3}`` relative to the signal norm: naive FP32 is unusable.
A viable scheme needs Kahan/Neumaier compensation in the bin reduction *and*
an error model validated against FP64 on the target grid (the fixture harness
is the right tool). Pursue only if a concrete GPU-bound campaign demands it.

## 5. Derivative-free optimizer fallback (only when differentiability breaks)

Deliberately **not** part of v1.0. The current loss is a C^∞ pure-Julia
least-squares objective in six dimensions with exact ForwardDiff gradients
and Hessians; the exact-Hessian interior-point Newton locates minima
precisely enough to resolve ``D^2 \sim 10^{-19}`` in 16–40 iterations. A
simplex-type method converges linearly, stalls near parameter accuracy
``\sqrt{\varepsilon} \approx 10^{-8}``, and would raise the optimizer floor
by roughly eight to ten decades while costing *more* wall time (hundreds of
1.57M-bin evaluations vs ~30 Hessian steps). Robustness is already covered
elsewhere: AD correctness is A/B-locked against the committed fixtures, and
basin robustness is handled by seeded multi-start.

**Trigger condition:** an objective the AD cannot penetrate — production
waveform families called through external C libraries (LALSuite-style), or
non-smooth statistics from the noise-realization study (item 2).

**Chosen candidates (in order):** `PRIMA.jl` **BOBYQA** — registered,
actively maintained modern reimplementation of Powell's methods, natively
bound-constrained, quadratic-model-based (far superior to Nelder–Mead on
smooth-ish low-dimensional problems); and, for a zero-new-dependency sanity
cross-check available today, `Fminbox(NelderMead())` from Optim.jl (already
in the dependency tree). Wire either through the existing
`[pipeline].optimizer` validation and the `calculate_numerical_distance`
dispatch; expect and document a raised floor in the scaling figures.
