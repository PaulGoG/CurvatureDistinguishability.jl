# Roadmap — deferred features

Design sketches for features deliberately **not** implemented yet (user
decision, 2026-07), so future work starts warm. None of these block the
production campaign.

**Scope decision (2026-08):** the current publication is complete with the
1.5PN waveform model and the noiseless-baseline validation. Every item on
this page is post-publication follow-up work; items 4 and 5 are closed
and will not be implemented.

## 1. 2PN spin–spin phase term

**Why.** The 1.5PN spin–orbit phase depends on the spins only through
``\beta = [(113 - 76\eta)\chi_s + 113\,\delta_m\chi_a]/12``, so the spin
combination with ``\mathrm{d}\beta = 0`` (``\chi_a`` at equal mass, a
tilted line otherwise) is an exact flat direction at every mass ratio
(tangent rank 5), and the spin-plane confusion zones are prior-limited
along it. The 2PN phase carries a spin1–spin2 coupling
``\sigma \propto \chi_1\chi_2`` (plus ``\chi_i^2`` self-terms) which adds
genuinely new spin structure and turns those boundary segments
curvature-limited.

**How.**
- Extend `pn_phase` (src/Physics.jl) with the 2PN term of the bracket,
  ``(15293365/508032 + 27145\eta/504 + 3085\eta^2/72 - 10\sigma)\,v^4``, with
  ``\sigma`` transcribed from a primary reference (Poisson & Will 1995;
  Arun et al. 2009 for the aligned-spin self-terms) with the same care as
  the Robson Table 1 transcription; add the transcription to the test suite.
- Config-gate it: `[physics] spin_spin_2pn = false` (default off) threaded
  through `WaveformParams`, so all existing fixtures/tests and the v1.0
  model stay bit-identical when disabled. The term is a toggle, not a
  replacement: the 1.5PN model remains the default and stays selectable.
- Consequences when enabled: K values change everywhere and spin maps gain
  genuine structure — the paper would need a model-variant note and
  regenerated figures. The A/B fixture tests must assert the *disabled*
  path only.
- Display: `Plotting.zone_figure` draws a same-unit zone whose
  principal-axis aspect reaches `NEEDLE_ASPECT` in the frame of its null
  direction (abscissa along it, ordinate transverse; introduced for the
  1.5PN spin needles, whose transverse width is otherwise invisible).
  Because the term is a toggle, the 1.5PN needles remain and the frame
  stays; with the term enabled the spin zone closes, the criterion stops
  firing and the parameter frame returns by itself — verify this on the
  first 2PN spin maps.

## 1b. Inspiral–merger–ringdown realism (IMRPhenomD)

**Why.** The LDC massive-binary catalogues (Sangria, Spritz, Yorsh) are
generated with IMRPhenomD/HM; the inspiral-only model windows the signal
out at ISCO and discards the merger SNR, which dominates for
``M \gtrsim 10^{6}\,M_\odot`` sources in the LISA band. A frequency-domain
phenomenological approximant is the natural next step for realism.

**How.**
- Port the IMRPhenomD amplitude and phase (Husa et al. 2016; Khan et al.
  2016) as a second scalar core selectable through `[physics].approximant`,
  keeping the six-parameter state vector (``\eta`` fixed per scenario).
- The GPU kernels take the coefficient set as kernel arguments; the
  IMRPhenomD coefficient struct exceeds the 2048-byte argument limit of the
  Intel iGPU compiler, so the coefficients must be recomputed per bin from
  ``(\mathcal{M}, \eta, \chi_1, \chi_2)`` inside the kernel (they are cheap
  polynomials) rather than passed in.
- Validation: SNR and phase agreement against an independent
  implementation (LAL through PythonCall) on the fixture grid.

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

**Compiler-limit fallback (implemented; validated on the Meteor Lake
iGPU):** if a backend's compiler rejects the full 49-lane Hessian kernel,
the Hessian is evaluated in chunks: the config key
`[hardware].hessian_chunk` (0 = full chunk) threads
`ForwardDiff.HessianConfig(loss, x, ForwardDiff.Chunk{c}())` through
`calculate_numerical_distance`'s `h!`, so each kernel launch carries (1+c)²
lanes and a Hessian costs ⌈6/c⌉² launches (ForwardDiff applies the chunk to
the outer and the inner dual: c = 3 gives 16 lanes × 4 launches, c = 2 gives
9 × 9, c = 1 gives 4 × 36). The failure mechanism was isolated on the iGPU:
the nested-dual parameter tuple enters the kernel as an argument of
6 × 49 × 8 = 2352 bytes, exceeding the device's 2048-byte kernel-argument
limit — IGC reports "Total size of kernel arguments exceeds limit" and fails
the module build (`ZE_RESULT_ERROR_MODULE_BUILD_FAILURE`). Value (1-lane)
and gradient (7-lane) kernels fit comfortably; chunk 3 carries 16 lanes =
768 bytes. Meteor Lake Xe-LPG executes FP64 natively
(`ZE_DEVICE_MODULE_FLAG_FP64` is reported without emulation flags). On the
production grid a 16-lane launch takes about 4.3 s on the Meteor Lake iGPU,
against the 7.5 s i915 preempt timeout; 9 lanes (chunk 2) take about 0.9 s,
which is why chunk 2 is the shipped GPU baseline. Agreement with the CPU
reference is at machine epsilon (2×10⁻¹⁶) for every chunk. The chunked path
is exercised in the test suite.

## 4. Float32 + compensated summation (closed — will not be implemented)

FP32 would raise consumer-GPU throughput ~64× for this kernel, but ``D^2``
spans ``10^{-21}``–``10^{-3}`` relative to the signal norm, so naive FP32 is
unusable and a viable scheme would need compensated bin reductions plus an
error model validated against FP64 on the target grid. Closed: the
pipeline stays FP64 throughout.

## 5. Derivative-free optimizer fallback (closed — will not be implemented)

**Closed by decision (2026-08):** no derivative-free fallback will be
implemented. The analysis below is retained for the record: the loss is
smooth with exact AD derivatives, the fallback would raise the optimizer
floor by many decades at higher cost, and the trigger condition (an
AD-impenetrable objective) is not on the project's path.

Original analysis — deliberately **not** part of v1.0. The current loss is a C^∞ pure-Julia
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

## 6. Public-release checklist (repository is private until then)

Deferred deliberately while Actions minutes are billed and the manuscript
is under revision; every item is mechanical.

- CI: restore the `push`/`pull_request`/tag triggers and the macOS,
  Windows and pre-release Linux legs removed from `.github/workflows/CI.yml`
  for the private phase.
- Dependency automation: CompatHelper (weekly `[compat]` bumps) and
  TagBot (release tags) workflows; the Dependabot GitHub-Actions ecosystem
  stays.
- Documentation: `deploydocs` to GitHub Pages with `prettyurls` restored
  to the CI-conditional form in `docs/make.jl`.
- Citation: DOI (Zenodo archive of the tagged release) in `CITATION.cff`
  and the README.
- Registration in the General registry after the v1.0.0 tag.
