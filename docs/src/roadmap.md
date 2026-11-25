# Roadmap

This page lists the extensions planned beyond the current model, in order of
priority, each with the design sketch it would follow. None of them is
implemented. The last section documents the operational behaviour of the
GPU backends.

## 1. Inspiral–merger–ringdown realism (IMRPhenomD)

The LDC massive-binary catalogues (Sangria, Spritz, Yorsh) are
generated with IMRPhenomD/HM; the inspiral-only model windows the signal
out at ISCO and discards the merger SNR, which dominates for
``M \gtrsim 10^{6}\,M_\odot`` sources in the LISA band. A frequency-domain
phenomenological approximant is the natural next step for realism.

Implementation:
- Port the IMRPhenomD amplitude and phase ([Husa2016](@cite);
  [Khan2016](@cite)) as a second scalar core selectable through
  `[physics].approximant`, keeping the six-parameter state vector
  (``\eta`` fixed per scenario).
- The GPU kernels take the coefficient set as kernel arguments; the
  IMRPhenomD coefficient struct exceeds the 2048-byte argument limit of the
  Intel iGPU compiler, so the coefficients must be recomputed per bin from
  ``(\mathcal{M}, \eta, \chi_1, \chi_2)`` inside the kernel (they are cheap
  polynomials) rather than passed in.
- Validation: SNR and phase agreement against an independent
  implementation (LAL through PythonCall) on the fixture grid.

## 2. Noise-realization study (distribution of the distance statistic)

The pipeline validates the noiseless geometry. With the likelihood
interpretation ``\Delta \ln \mathcal{L}_{\max} = D^2/2``, the open question is
how ``\hat D^2`` is distributed under noise and how sharp the confusion
boundary then is.

Implementation:
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

### Device sizing

The lanes kernel (value + per-partial Float64 stores, see
src/Inference.jl `loss_bins_lanes!`) is vendor-portable; the
consumer-GPU bottleneck is FP64 throughput, 1/64 of the FP32 rate on recent
consumer NVIDIA parts. On FP64-strong hardware (A100/H100/MI200-class, 10–60× more FP64) the
sweep stage becomes GPU-dominated; the map stage remains CPU by design
(`ForwardDiff.jacobian` on host) and then dominates. If the campaign time
matters, parallelize the maps across CPU cores concurrently with the GPU
sweeps. Validate with `bench/` + a single-δ solve before committing a queue
allocation.

### Launch serialization and device-buffer cache

Kernel launches are serialized through a library-wide lock and
`plan_resources` pins the sweep stage of a GPU run to a single task, because the vendor
runtimes are not safe under concurrent multi-task access. Device output
buffers are cached per backend, element type
and shape, because allocating them per evaluation destabilizes long sessions;
a sweep evaluates the loss 10³ to 10⁴ times on one cached buffer. A run interrupted at any point
continues from its completed stages and work-item checkpoints
(`scripts/run_pipeline.jl --resume`).

### Chunked Hessian

If a backend's compiler rejects the full 49-lane Hessian kernel, the Hessian
is evaluated in chunks: the config key
`[hardware].hessian_chunk` (0 = full chunk) threads
`ForwardDiff.HessianConfig(loss, x, ForwardDiff.Chunk{c}())` through
`calculate_numerical_distance`'s `h!`, so each kernel launch carries (1+c)²
lanes and a Hessian costs ⌈6/c⌉² launches (ForwardDiff applies the chunk to
the outer and the inner dual: c = 3 gives 16 lanes × 4 launches, c = 2 gives
9 × 9, c = 1 gives 4 × 36). On the Meteor Lake iGPU the nested-dual
parameter tuple enters the kernel as an argument of
6 × 49 × 8 = 2352 bytes, exceeding the device's 2048-byte kernel-argument
limit. IGC reports "Total size of kernel arguments exceeds limit" and fails
the module build (`ZE_RESULT_ERROR_MODULE_BUILD_FAILURE`). Value (1-lane)
and gradient (7-lane) kernels fit; chunk 3 carries 16 lanes =
768 bytes. Meteor Lake Xe-LPG executes FP64 natively
(`ZE_DEVICE_MODULE_FLAG_FP64` is reported without emulation flags). On the
production grid a 16-lane launch takes about 4.3 s on the Meteor Lake iGPU,
against the 7.5 s i915 preempt timeout; 9 lanes (chunk 2) take about 0.9 s,
which is why chunk 2 is the shipped GPU baseline. Per-evaluation losses and
Hessians agree with the CPU reference to machine epsilon (2×10⁻¹⁶) for every
chunk. The chunked path
is exercised in the test suite.

Long GPU campaigns should run under `scripts/run_supervised.jl`, which detects
a stalled device from the worker's heartbeat and CPU time and continues the run
in a new process (see Run Parameters, Supervised runs).
