# GPU Postmortem — every failure, its root cause, and the fix

Chronological record of all six GPU-related failures encountered while
bringing this pipeline to a working GPU path (2026-07), written so that
future development does not rediscover them. Each entry: **symptom → root
cause → fix (and where it lives) → general principle.** The first four were
found during development and testing; the last two struck during the
production campaign and were fixed in code afterwards.

## 1. The decorative GPU layer (legacy, pre-remediation)

- **Symptom:** GPU "support" existed in the original codebase but no GPU
  computation ever ran; `force_cpu = true` hid this, and the 2026-06
  campaign burned 470 CPU-hours.
- **Root cause:** four independent blockers — GPU packages absent from the
  environment (`try using CUDA catch end` silently failed), detection via
  `isdefined(Main, :CUDA)` reflection, zero actual kernels, and a compute
  path broadcasting `tdi_modulation.(freqs, Ref(p))` where `p` is a heap
  `Vector{ForwardDiff.Dual}` — a non-isbits capture no GPU compiler accepts.
- **Fix:** full rewrite — package extensions (`ext/TWD*Ext.jl`) with a probe
  registry in `Hardware.jl`; a single KernelAbstractions kernel over a
  scalar physics core; parameters cross the kernel boundary as isbits
  `NTuple`s (src/Inference.jl `loss_bins!`).
- **Principle:** GPU-compatibility claims are worthless untested; isbits
  discipline at the kernel boundary is non-negotiable, and optional GPU
  deps belong in `[weakdeps]` + extensions, never behind `Main` reflection.

## 2. Dual-eltype device reduction rejected (`ZE_RESULT_ERROR_INVALID_SIZE`)

- **Symptom:** GPU loss *value* worked, but `ForwardDiff.gradient` through
  the kernel crashed: Level Zero rejected the reduction with
  `ZE_RESULT_ERROR_INVALID_SIZE`.
- **Root cause:** the output buffer was a device array with a 56-byte
  `Dual{…,Float64,6}` eltype; exotic-eltype allocations/reductions are not
  reliably supported by GPU runtimes (observed on oneAPI; other backends
  may differ, but none guarantee it).
- **Fix:** the **lanes layout** (src/Inference.jl `loss_bins_lanes!` +
  `flatten_dual`/`rebuild_dual`): Dual arithmetic stays *inside* the kernel
  (scalar Duals are isbits and compile fine), but outputs are stored as
  plain `Float64` lanes (value + partials, recursively) in an `n × M`
  matrix, reduced with `M` standard per-column sums, and the scalar Dual is
  reassembled on the host. CPU-backend equivalence of this path is
  regression-tested to machine epsilon.
- **Principle:** device arrays should carry primitive eltypes only; keep
  AD types inside kernels, never in device memory layouts or reductions.

## 3. Lane-tuple materialization → `gpu_malloc` InvalidIRError

- **Symptom:** with the lanes layout, gradients (7 lanes) worked but
  Hessians (49 lanes) failed to compile: `InvalidIRError … call to
  gpu_malloc` (a heap allocation in device code).
- **Root cause:** building the full `NTuple{49,Float64}` via recursive tuple
  splatting before storing it exceeded what the IGC compiler would
  stack-allocate; it fell back to heap allocation, which is illegal in
  device code.
- **Fix:** direct recursive per-scalar stores (`_store_dual!` /
  `_store_parts!`): each lane is written to the output matrix as it is
  produced — no intermediate wide tuple ever exists.
- **Principle:** avoid materializing large tuples in kernels; write-through
  recursion compiles to straight-line stores on every backend.

## 4. 49-lane Hessian exceeds an IGC module-build limit (FP64 emulation)

- **Symptom:** after fix #3 the Hessian kernel produced *valid IR* but the
  Intel graphics compiler failed the module build
  (`ZE_RESULT_ERROR_MODULE_BUILD_FAILURE`) on the Meteor Lake iGPU.
- **Root cause:** FP64 *emulation* multiplies the instruction count per
  double operation; 49 emulated-FP64 lanes blow an internal IGC limit.
  Value (1 lane) and gradient (7 lanes) kernels build fine.
- **Fix / escape hatch:** chunked Hessian evaluation — `[hardware]
  .hessian_chunk = 3` threads `ForwardDiff.HessianConfig(…, Chunk{3}())`
  through the optimizer's `h!`, so each launch carries (1+3)·(1+6) = 28
  lanes. Validated on the iGPU: 0.91 s per production-size Hessian,
  machine-epsilon agreement (2×10⁻¹⁶) with the CPU reference. On
  native-FP64 hardware the full 49-lane kernel is expected to build; try
  `hessian_chunk = 0` first, fall back to 3.
- **Principle:** compiler limits are backend- and precision-dependent;
  provide a chunking knob rather than assuming the widest kernel compiles
  everywhere.

## 5. Level Zero segfault under concurrent task access (production, campaign night)

- **Symptom:** the first GPU campaign crashed ~2 minutes in:
  `julia terminated by signal SEGV` inside the driver, under the sweep
  stage's multi-task pool. The serial benchmark had run the identical
  kernels for hours without issue.
- **Root cause:** several Julia tasks (on different OS threads) driving the
  Level Zero runtime concurrently — driver state is not safe under this
  pattern.
- **Fix (in code, not config):** all GPU kernel launches serialize through
  the library-wide `Inference.GPU_LOCK`, and `Orchestrator.plan_resources`
  pins any GPU run to a single task with a logged explanation. This costs
  nothing: the device serializes kernels regardless, so GPU-side task
  concurrency only added crash surface.
- **Principle:** treat GPU drivers as single-consumer resources unless a
  backend documents otherwise; enforce it in the library so no
  configuration can reintroduce the crash. **Monitoring corollary:** the
  failure was silent for hours because the log-watch filter matched
  `ERROR`-style lines but not `terminated by signal` — failure filters must
  cover every terminal signature, and silence must never be read as
  progress.
- **Validation (2026-07-22):** with the `GPU_LOCK` + buffer cache in place,
  a deliberate stress test ran 10 concurrent optimizations on the
  FP64-emulated iGPU with the single-task pin *bypassed* (22 Julia threads,
  concurrent `to_backend` allocations + concurrent solves). It completed in
  95 s with **no segfault and zero caught failures**, every `D²` matching
  theory to ratio 1.000. So the library-wide lock — not the task pin — is
  what actually makes concurrent GPU access safe; the `plan_resources`
  single-task pin is now belt-and-suspenders (kept because GPU kernels
  serialize on the device regardless, so multi-task concurrency buys no GPU
  throughput — only CPU-side optimizer overlap across δ-points, which is
  minor). The pin can be relaxed if that overlap is ever wanted; the lock
  keeps it safe.

## 6. oneAPI "freed reference" in a long session (production, last sweep)

- **Symptom:** ~2.5 h into the GPU campaign, the final sweep failed with
  `ArgumentError: Attempt to copy a freed reference` inside oneAPI's array
  machinery; the four completed sweeps were preserved by the pipeline's
  per-stage guardrail.
- **Root cause:** device output buffers were allocated fresh on *every*
  loss evaluation — 10³–10⁴ device allocations per sweep. The accumulated
  allocation/free churn eventually tripped a oneAPI.jl memory-management
  bug (a device buffer freed while still referenced).
- **Fix (mitigation in code + operational fallback):** device buffers are
  now cached and reused (`Inference.DEVICE_BUFFER_CACHE`, guarded by the
  GPU lock), removing ~99.9% of the churn. A fresh-process retry of the
  failed sweep succeeded immediately and is the documented fallback: for
  very long GPU campaigns, split into per-sweep processes — the
  config-hash run directories and per-stage guardrails make this lossless.
- **Principle:** minimize device allocation churn (cache and reuse); design
  campaigns so any stage can be re-run in a fresh process without losing
  completed work.

## Operational summary for future GPU work

1. Kernel boundary: isbits parameters, primitive-eltype device arrays,
   no wide tuples, lanes layout for AD.
2. Concurrency: one task, one thread per GPU; the library enforces it.
3. Compilers: expect per-backend limits; `hessian_chunk` is the knob.
4. Long sessions: buffer cache is on by default; prefer per-sweep processes
   for multi-hour GPU campaigns.
5. FP64 emulation (Arc iGPUs): set `OverrideDefaultFP64Settings=1
   IGC_EnableDPEmulation=1`; expect ~10⁻¹⁰ relative value fidelity and a
   raised small-separation floor — cross-validated against CPU at the
   10⁻⁸ level in production.
6. Monitoring: watch for `SEGV`/`terminated by signal`/`freed reference`
   alongside conventional error strings.
