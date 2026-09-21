# Roadmap — planned extensions

This page lists the extensions planned beyond the current model, in order of
priority, each with the design sketch it would follow. None of them is
implemented; the pipeline ships the 1.5PN waveform model with the
noiseless-baseline validation. The last section documents the operational
behaviour of the GPU backends.

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
  genuine structure. The A/B fixture tests must assert the *disabled*
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
The likelihood interpretation (``\Delta \ln L_{\max} = D^2/2``) invites the
follow-up question: how is ``\hat D^2`` distributed under real noise, and how
sharp is the confusion boundary operationally?

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

## 3. GPU backends: operational notes

**Device sizing.** The lanes kernel (value + per-partial Float64 stores, see
src/Inference.jl `loss_bins_lanes!`) is vendor-portable by construction; the
consumer-GPU bottleneck is FP64 throughput (1/64 rate on recent consumer
NVIDIA parts — only ~2× end-to-end campaign gain over a good CPU). On
FP64-strong hardware (A100/H100/MI200-class, 10–60× more FP64) the sweep
stage becomes GPU-dominated; the map stage remains CPU by design
(`ForwardDiff.jacobian` on host) and then dominates — parallelize maps
across CPU cores concurrently with GPU sweeps if the campaign time matters.
Validate with `bench/` + a single-δ solve before committing a queue
allocation.

**Launch serialization and device-buffer cache.** Kernel launches are
serialized through a library-wide lock and `plan_resources` pins a GPU run to
a single task, because the vendor runtimes are not safe under concurrent
multi-task access. Device output buffers are cached per backend, element type
and shape, because allocating them per evaluation destabilises long sessions;
the cache reduces device allocations by ~10³. A run interrupted at any point
continues from its completed stages and work-item checkpoints
(`scripts/run_pipeline.jl --resume`).

**Chunked Hessian.** If a backend's compiler rejects the full 49-lane Hessian
kernel, the Hessian is evaluated in chunks: the config key
`[hardware].hessian_chunk` (0 = full chunk) threads
`ForwardDiff.HessianConfig(loss, x, ForwardDiff.Chunk{c}())` through
`calculate_numerical_distance`'s `h!`, so each kernel launch carries (1+c)²
lanes and a Hessian costs ⌈6/c⌉² launches (ForwardDiff applies the chunk to
the outer and the inner dual: c = 3 gives 16 lanes × 4 launches, c = 2 gives
9 × 9, c = 1 gives 4 × 36). The mechanism on the Meteor Lake iGPU:
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

Long GPU campaigns should run under `scripts/run_supervised.jl`, which detects
a stalled device from the worker's heartbeat and CPU time and continues the run
in a new process (see Complex Run Parameters, Supervised runs).
