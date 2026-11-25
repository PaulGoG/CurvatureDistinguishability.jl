"""
Pipeline driver: resource planning against the [safety] budget, the 1D
sweep and 2D confusion-mapping stages, structured logging, opt-in
monitoring and per-stage error isolation.
"""
module Orchestrator

using DocStringExtensions: TYPEDSIGNATURES
using Printf: @sprintf
using Dates: Dates, now
using TOML: TOML
using CSV: CSV
using DataFrames: DataFrame
using Random: Xoshiro
using SHA: sha256
using ProgressMeter: Progress, ProgressUnknown, finish!, next!
using Logging: Logging, global_logger, with_logger
using LoggingExtras: FormatLogger, MinLevelLogger, TeeLogger
using KernelAbstractions: KernelAbstractions

using ..Backends
using ..Physics
using ..Physics: N_PARAMS
using ..Bounds: SPIN_INDICES
using ..Detector
using ..Residuals
using ..Geometry
using ..Inference
using ..Bounds
using ..Config
using ..Provenance
using ..Provenance: acquire_run_lock, record_attempt!, continuation_config
using ..Heartbeat: tick!, note_item!, start_heartbeat, stop_heartbeat!, HEARTBEAT_FILE
using ..Checkpoint:
    CheckpointRow, SweepCheckpoint, open_checkpoint, record_item!, CHECKPOINT_FILE
using ..Plotting
using ..Plotting: display_scale, map_diagnostic_panel, sweep_diagnostic_panel
using ..Fitting:
    MIN_FIT_POINTS, above_floor_mask, angular_fraction, loglog_slope, optimizer_floor,
    perturbative_mask, ratio_correction_fit, validity_radius
using ..ZoneSampling:
    unit_direction, refine_directions!, insert_crossover_vertices!,
    insert_box_corner_vertices!, mirror_to_full_circle

export run_pipeline
public ResourceBudgetError,
    PipelineStageError, DegenerateDirectionError, DEFAULT_CONFIG,
    run_log_logger

"""
Configuration the pipeline scripts run without an explicit `--config`: the
minutes-scale quickstart scenario, relative to the project root, so a bare
invocation never launches a campaign.
"""
const DEFAULT_CONFIG = joinpath("configs", "quickstart.toml")

"""
    ResourceBudgetError(msg)

Raised by [`plan_resources`](@ref) when the `[safety]` memory budget cannot
accommodate even a single task of the configured grid; the message names
the budget to raise or the grid to reduce.
"""
struct ResourceBudgetError <: Exception
    msg::String
end
Base.showerror(io::IO, e::ResourceBudgetError) = print(io, "ResourceBudgetError: ", e.msg)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

"""
    PipelineStageError(run_dir, failed_stages)

Thrown by [`run_pipeline`](@ref) once the run has finished when one or more
stages failed: `metadata.toml` lists the failures and `run.log` carries their
backtraces. The other stages have run unless the device stopped responding.
"""
struct PipelineStageError <: Exception
    run_dir::String
    failed_stages::Vector{String}
end

function Base.showerror(io::IO, err::PipelineStageError)
    print(io, "PipelineStageError: stage(s) ", join(err.failed_stages, ", "),
        " failed; backtraces in ", joinpath(err.run_dir, "run.log"))
end

"""
    DegenerateDirectionError(sweep, g_uu, threshold)

The Fisher norm `g(u,u)` of a sweep direction lies below
`[pipeline.sweep_settings].g_uu_degenerate`: the two sources cannot be
separated along it and no separation in Fisher units exists. The sweep is
recorded as a failed stage.
"""
struct DegenerateDirectionError <: Exception
    sweep::String
    g_uu::Float64
    threshold::Float64
end

function Base.showerror(io::IO, err::DegenerateDirectionError)
    print(io, "DegenerateDirectionError: sweep '", err.sweep, "' has g(u,u) = ",
        err.g_uu, " below the degeneracy threshold ", err.threshold)
end

# prior-limited boundary fraction above which the spin-plane note is logged
const SPIN_PRIOR_NOTE_FRACTION = 0.25
# simultaneously live whitened-vector buffers per nested-dual evaluation in
# the plan_resources memory model (value + two derivative work arrays)
const NESTED_DUAL_EVAL_BUFFERS = 3

format_time(seconds) = @sprintf(
    "%02d:%02d:%02d",
    divrem(divrem(floor(Int, seconds), 60)[1], 60)...,
    mod(floor(Int, seconds), 60)
)

"""
Timestamped, ANSI-free line to a stage log file, mirrored to the console.
"""
function logline(io::IO, msg::AbstractString)
    stamped = "[" * Dates.format(now(), "yyyy-mm-dd HH:MM:SS") * "] " * msg
    tick!()
    println(stdout, msg)
    flush(stdout)
    println(io, stamped)
    flush(io)
    return nothing
end

progress_enabled() = stderr isa Base.TTY

"""
Bounded-concurrency parallel foreach: at most `ntasks` of the spawned tasks
run simultaneously. Exceptions propagate to the caller (the stage guards
catch them there).
"""
function parallel_foreach(f, n::Int, ntasks::Int)
    sem = Base.Semaphore(max(1, ntasks))
    @sync for i in 1:n
        Threads.@spawn begin
            Base.acquire(sem)
            try
                f(i)
            finally
                Base.release(sem)
            end
        end
    end
    return nothing
end

# InterruptException and OutOfMemoryError end the run, also when a work-item
# task raised them and `@sync` delivered them wrapped
is_fatal(err) = err isa Union{InterruptException,OutOfMemoryError}
is_fatal(err::TaskFailedException) = is_fatal(err.task.result)
is_fatal(err::CompositeException) = any(is_fatal, err.exceptions)

"""
$(TYPEDSIGNATURES)

Pre-flight memory estimate against the `[safety]` budget: refuses to start
when even a single task exceeds it, downscales concurrency otherwise, and
applies the VRAM budget on GPU backends. Returns `(sweep_tasks, map_tasks,
estimated_gb)`: one sweep task on GPU backends; the mapping stage, which
launches no kernels, keeps the CPU concurrency.
"""
function plan_resources(cfg::PipelineSettings, n_bins::Int, n_ch::Int, backend)
    flatlen = 2 * n_ch * n_bins
    fixed =
        flatlen * N_PARAMS * 8 +               # tangent-basis Jacobian
        flatlen * 8 * (N_PARAMS + 1) * NESTED_DUAL_EVAL_BUFFERS +
        N_PARAMS * n_ch * n_bins * 16 +        # orthonormal basis storage
        4 * n_bins * 8                         # grid + PSD
    per_task = (2 + 3 * n_ch) * n_bins * 16 # per-δ data streams and temporaries
    budget = cfg.max_ram_gb * 2^30

    if fixed + per_task > budget
        throw(
            ResourceBudgetError(
                "Estimated memory for a single task ($(round(fixed / 2^30, digits = 2)) GB fixed + " *
                "$(round(per_task / 2^30, digits = 2)) GB/task) exceeds [safety].max_ram_gb = " *
                "$(cfg.max_ram_gb) GB. Reduce the frequency grid ([grid]) or raise the budget.",
            ),
        )
    end

    threads = min(Threads.nthreads(), cfg.max_threads)
    while threads > 1 && fixed + threads * per_task > budget
        threads -= 1
    end
    threads < min(Threads.nthreads(), cfg.max_threads) &&
        @warn "Concurrency downscaled to $threads tasks to respect [safety].max_ram_gb = $(cfg.max_ram_gb) GB."

    map_tasks = threads
    sweep_tasks = threads
    if !(backend isa KernelAbstractions.CPU)
        # One task per GPU sweep: vendor runtimes are not safe under concurrent
        # multi-task access, the library lock serializes kernel launches and the
        # device serializes kernels, so further tasks add no throughput. The
        # mapping stage never dispatches kernels and keeps its CPU concurrency.
        sweep_tasks = 1
        map_tasks > 1 &&
            @info "GPU backend: one sweep task; the 2D mapping stage keeps " *
                  "$map_tasks tasks."
        vram_budget = max(0.0, cfg.max_vram_gb - cfg.os_vram_overhead_gb) * 2^30
        need = n_bins * cfg.bytes_per_bin_per_task_gpu
        need <= vram_budget ||
            throw(
                ResourceBudgetError(
                    "VRAM budget ($(round(vram_budget / 2^30, digits = 2)) GB usable) is below " *
                    "the footprint of one GPU task " *
                    "($(round(need / 2^30, digits = 2)) GB at " *
                    "$(cfg.bytes_per_bin_per_task_gpu) bytes/bin). Reduce the grid or raise " *
                    "[safety].max_vram_gb.",
                ),
            )
    end

    return sweep_tasks, map_tasks, (fixed + map_tasks * per_task) / 2^30
end

"""
$(TYPEDSIGNATURES)

Inter-stage memory maintenance for long campaigns (`[hardware]
.gc_between_stages`): flushes the device-buffer cache on GPU backends, runs
a full host garbage collection (device arrays are freed by their host
finalizers, so only a collection returns device memory between stages),
and asks the backend to reclaim pooled device memory where an API exists.
"""
function maintain_memory!(backend, enabled::Bool)
    enabled || return nothing
    if !(backend isa KernelAbstractions.CPU)
        Inference.clear_device_buffers!()
    end
    GC.gc(true)
    Backends.reclaim_device_memory!(backend)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Thread-safe progress hook for a stage's work-item loop: returns a
zero-argument closure that counts completed items and emits an `@info` line
into the structured run log every `fraction` of `total` items (never at the
final item, since completion has its own log line). `fraction <= 0` disables the
hook (`[monitoring].progress_log_fraction = 0`). Long stages are otherwise
silent in `run.log` for hours, which detached campaign monitoring
(`tail -f`) cannot distinguish from a hang.
"""
function stage_progress_hook(label::String, total::Int, fraction::Real)
    fraction > 0 || return () -> nothing
    done = Threads.Atomic{Int}(0)
    step = max(1, ceil(Int, fraction * total))
    return () -> begin
        d = Threads.atomic_add!(done, 1) + 1
        d % step == 0 && d < total &&
            @info "$label: $d/$total work items complete."
        return nothing
    end
end

"""
$(TYPEDSIGNATURES)

Whether `backend` still executes work: a small upload, a synchronization and a
read-back. After a context reset or a lost device every one of them
throws, and so would every remaining GPU stage of the process. Always `true`
on the CPU backend.
"""
function device_responds(backend)
    backend isa KernelAbstractions.CPU && return true
    try
        probe = to_backend(ones(Float64, 16), backend)
        KernelAbstractions.synchronize(backend)
        return sum(Array(probe)) == 16.0
    catch err
        err isa Union{InterruptException,OutOfMemoryError} && rethrow()
        @debug "device probe failed" exception = err
        return false
    end
end

"""
Immutable per-run context threaded through the sweep and map stages:
validated configuration, output base directory, frequency grid and PSD
(host and device copies), the planned stage concurrencies and the number
of the process launch working on the run directory
(`attempt`, above 1 when an interrupted run is continued). Parametric over
the backend and device-array types so every per-δ solve closure captures
concrete fields.
"""
struct RunContext{B,FD,SD}
    cfg::PipelineSettings
    out_base::String
    freqs::Vector{Float64}
    Sn::Vector{Float64}
    freqs_dev::FD
    Sn_dev::SD
    df::Float64
    backend::B
    sweep_tasks::Int
    map_tasks::Int
    attempt::Int
end

# -----------------------------------------------------------------------------
# Stage 1: one-dimensional parameter sweeps (δ⁴ validation)
# -----------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Execute one 1D separation sweep: directional geometry (K(u), g(u,u)),
per-δ box-constrained optimization with optional multi-start
([`optimize_separations`](@ref)), floor detection and slope/correction
fits ([`fit_sweep_law`](@ref)), and persistence of the results table,
residual spectrum, metadata and figures into the run directory
([`persist_sweep_results`](@ref)).
"""
function run_sweep(sweep::SweepSpec, idx::Int, total::Int, ctx::RunContext)
    cfg = ctx.cfg
    wp = cfg.wp
    name = sweep.name
    theta0 = sweep.theta_0
    u_raw = sweep.u_dir
    rho_sq = sweep.rho_thresh^2
    amp_ratio = sweep.amp_ratio                              # A₂/A₁
    amp_prefactor = (2amp_ratio / (1 + amp_ratio))^2         # (A_harm/A)², = 1 at amp_ratio = 1

    out_dir = joinpath(ctx.out_base, "sweeps", name)
    mkpath(out_dir)
    log_io = open(joinpath(out_dir, "sweep.log"), "a")
    t0 = time()
    try
        logline(log_io, "─"^78)
        logline(log_io, "[Sweep $idx/$total] $name")
        logline(log_io, "  base parameters : $theta0")
        logline(log_io, "  direction       : $u_raw")
        logline(log_io, "  rho^2 threshold : $rho_sq")
        amp_ratio != 1.0 && logline(
            log_io,
            @sprintf("  amplitude ratio : A₂/A₁ = %.4g (A_harm prefactor %.5f)",
                amp_ratio, amp_prefactor),
        )

        logline(log_io, "  [1/3] manifold geometry (K(u), g(u,u))")
        K_u, g_uu =
            compute_extrinsic_curvature(theta0, u_raw, ctx.freqs, ctx.Sn, ctx.df, wp)
        logline(log_io, @sprintf("        raw Fisher norm g(u,u)   : %.6e", g_uu))
        if g_uu < cfg.g_uu_degenerate
            logline(
                log_io,
                "  [ERROR] degenerate direction: g(u,u) below $(cfg.g_uu_degenerate)",
            )
            throw(DegenerateDirectionError(name, g_uu, cfg.g_uu_degenerate))
        end
        norm_scale = 1.0 / sqrt(g_uu)
        u_norm = u_raw .* norm_scale
        K_norm = K_u * norm_scale^4
        # D² = (p/16) K δ⁴ = ρ² with the amplitude prefactor p = (A_harm/A)²
        delta_min = (16.0 * rho_sq / (amp_prefactor * K_norm))^(1 / 4)
        logline(log_io, @sprintf("        normalized K(u)          : %.6e", K_norm))
        logline(log_io, @sprintf("        delta_min                : %.6e", delta_min))

        # absolute separation grid in Fisher-normalized units (δ = 1 is one
        # Fisher σ along u). It reaches below the optimizer floor of the fits
        # by design (floor points are detected, struck through in the figure
        # and excluded from the fits) and must bracket δ_min for the
        # threshold crossing to lie inside the sweep.
        deltas = 10 .^ range(cfg.min_log_delta, cfg.max_log_delta, length = cfg.n_deltas)
        logline(
            log_io,
            @sprintf("        separation grid          : [%.3e, %.3e] = [%.2e, %.2e] δ_min",
                deltas[1], deltas[end], deltas[1] / delta_min, deltas[end] / delta_min),
        )
        if !(deltas[1] < delta_min < deltas[end])
            logline(
                log_io,
                "  [WARN] the separation grid does not bracket δ_min; the threshold " *
                "crossing lies outside the sweep.",
            )
            @warn "Sweep '$name': the separation grid [$(deltas[1]), $(deltas[end])] does " *
                  "not bracket δ_min = $delta_min; the threshold crossing lies outside the sweep."
        end

        theta_far = second_source(theta0, u_norm, maximum(deltas), amp_ratio)
        for i in 1:N_PARAMS
            cfg.bounds.periodic[i] && continue
            if !(cfg.bounds.lower[i] <= theta_far[i] <= cfg.bounds.upper[i])
                @warn "Sweep '$name': the second source leaves the physical bounds along " *
                      "component $i before δ_max (θ₂[$i] reaches $(theta_far[i])). Large-δ points " *
                      "describe an unphysical companion."
                break
            end
        end

        n = cfg.n_deltas
        logline(
            log_io,
            "  [2/3] optimization sweep ($(n) separations, " *
            "$(ctx.sweep_tasks) concurrent, optimizer = $(cfg.optimizer)" *
            (cfg.n_starts > 1 ? ", $(cfg.n_starts) starts" : "") * ")",
        )
        checkpoint = open_checkpoint(joinpath(out_dir, CHECKPOINT_FILE),
            sweep_signature(sweep, deltas, u_norm, cfg))
        isempty(checkpoint.rows) || logline(log_io,
            "        continuing: $(length(checkpoint.rows))/$n separations in the checkpoint",
        )
        fits = optimize_separations(sweep, deltas, u_norm, "Sweep '$name' ($idx/$total)",
            ctx, checkpoint)
        law = fit_sweep_law(deltas, fits.D2_num, K_norm, amp_prefactor, cfg, log_io)

        if cfg.n_starts > 1 && maximum(fits.multi_start_gain) > cfg.secondary_minimum_gain
            @warn "Sweep '$name': multi-start found a lower minimum than the canonical " *
                  "start for $(count(>(cfg.secondary_minimum_gain), fits.multi_start_gain))/$n separations (max gain " *
                  "$(round(maximum(fits.multi_start_gain), digits = 2))): evidence of secondary minima."
        end
        any(fits.at_bound) &&
            @warn "Sweep '$name': the best fit sits on a physical bound for " *
                  "$(count(fits.at_bound))/$n separations (see AtBound column)."
        all(fits.converged) || @warn "Sweep '$name': optimizer did not converge for " *
              "$(count(!, fits.converged))/$n separations (see Converged column)."

        if cfg.monitoring_enabled && progress_enabled()
            println(
                stdout,
                sweep_diagnostic_panel(deltas, fits.D2_num, law.D2_theo, law.clean,
                    law.slope, law.slope_err),
            )
        end

        logline(log_io, "  [3/3] persisting results and figures")
        persist_sweep_results(out_dir, sweep, deltas, u_norm,
            (; K_norm, g_uu, delta_min), fits, law, ctx)

        elapsed = format_time(time() - t0)
        logline(log_io, "  [done] sweep '$name' completed in $elapsed")
        # stage timing in run.log, which detached campaigns follow
        @info "Sweep '$name' ($idx/$total) completed in $elapsed."
    finally
        close(log_io)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Per-separation box-constrained fits of a sweep: for every δ in `deltas`
the two-source data (second source at `θ₀ + δ u_norm` with the sweep's
amplitude ratio) is fitted from the degenerate-source guess and, with
`n_starts > 1`, from seeded perturbations of it, keeping the lowest
distance. Runs `ctx.sweep_tasks` fits concurrently; `label` names the
stage in the run-log progress lines. Separations already in `checkpoint` are
taken from it; every other one is recorded there as soon as it is fitted, so
an interrupted sweep loses the items in flight only. Returns the per-δ
distances, best-fit parameters and optimizer diagnostics as a NamedTuple.
"""
function optimize_separations(sweep::SweepSpec, deltas::AbstractVector,
    u_norm::AbstractVector, label::String, ctx::RunContext,
    checkpoint::SweepCheckpoint)
    cfg = ctx.cfg
    wp = cfg.wp
    name = sweep.name
    theta0 = sweep.theta_0
    amp_ratio = sweep.amp_ratio
    n = length(deltas)
    D2_num = zeros(n)
    best_fits = zeros(n, N_PARAMS)
    # Vector{Bool}, not BitVector: elements are written from concurrent tasks and
    # neighboring bits of a BitVector share one machine word
    converged = fill(false, n)
    iteration_counts = zeros(Int, n)
    gradient_norms = zeros(n)
    at_bound = fill(false, n)
    multi_start_gain = ones(n)

    prog = Progress(n; desc = "  optimizing: ", enabled = progress_enabled())
    note_progress = stage_progress_hook(label, n, cfg.progress_log_fraction)
    store!(row::CheckpointRow) = begin
        i = row.index
        D2_num[i] = row.D2
        best_fits[i, :] .= row.best_fit
        converged[i] = row.converged
        iteration_counts[i] = row.iterations
        gradient_norms[i] = row.g_norm
        at_bound[i] = row.at_bound
        multi_start_gain[i] = row.multi_start_gain
    end
    finished = copy(checkpoint.rows)
    ch1 = channel_strain(theta0, ctx.freqs, wp)
    parallel_foreach(n, ctx.sweep_tasks) do i
        if haskey(finished, i)
            store!(finished[i])
            next!(prog)
            note_progress()
            return nothing
        end
        t_item = time()
        d = deltas[i]
        p2 = second_source(theta0, u_norm, d, amp_ratio)
        ch2 = channel_strain(p2, ctx.freqs, wp)
        data = map((a, b) -> a .+ b, ch1, ch2)
        if !(ctx.backend isa KernelAbstractions.CPU)
            data = map(a -> to_backend(a, ctx.backend), data)
        end
        # degenerate effective source: amplitude (1+q)A at the weighted
        # midpoint, with q = A₂/A₁ the amplitude ratio, i.e. luminosity
        # distance D/(1+q)
        guess = theta0 .+ (amp_ratio / (1 + amp_ratio) * d) .* u_norm
        guess[1] = theta0[1] / (1 + amp_ratio)

        freqs_active =
            ctx.backend isa KernelAbstractions.CPU ? ctx.freqs : ctx.freqs_dev
        Sn_active = ctx.backend isa KernelAbstractions.CPU ? ctx.Sn : ctx.Sn_dev
        solve(x0) =
            calculate_numerical_distance(data, x0, freqs_active, Sn_active, ctx.df;
                g_tol = cfg.g_tol,
                f_reltol = cfg.f_reltol,
                iterations = cfg.max_iterations,
                backend = ctx.backend,
                optimizer = cfg.optimizer,
                bounds = cfg.bounds,
                hessian_chunk = cfg.hessian_chunk,
                coordinates = cfg.fit_coordinates,
                wp = wp)
        dist, best, res = solve(guess)
        dist_canonical = dist
        for k in 2:cfg.n_starts
            rng = multi_start_stream(cfg.rng_seed, name, i, k)
            pert =
                guess .+ (cfg.multi_start_parallel_scale * d * randn(rng)) .* u_norm .+
                cfg.multi_start_transverse_scale .* randn(rng, N_PARAMS)
            dist_k, best_k, res_k = solve(clamp_interior(pert, cfg.bounds))
            if dist_k < dist
                dist, best, res = dist_k, best_k, res_k
            end
        end
        diagnostics = optimization_diagnostics(res, best, cfg.bounds)
        row = CheckpointRow(i, dist, collect(Float64, best), diagnostics.converged,
            diagnostics.iterations, diagnostics.g_norm, diagnostics.at_bound,
            dist > 0 ? dist_canonical / dist : 1.0, time() - t_item, ctx.attempt)
        record_item!(checkpoint, row)
        store!(row)
        note_item!()
        next!(prog)
        note_progress()
        return nothing
    end
    finish!(prog)
    return (; D2_num, best_fits, converged, iteration_counts, gradient_norms, at_bound,
        multi_start_gain)
end

"""
$(TYPEDSIGNATURES)

Identifier of the fitting problem behind a sweep checkpoint: SHA-256 of the
sweep definition, the separation grid, the normalized direction, the
optimizer and grid settings and the package `version` that computes it. Rows
written under another signature are refused.
"""
function sweep_signature(sweep::SweepSpec, deltas::AbstractVector,
    u_norm::AbstractVector, cfg::PipelineSettings;
    version::AbstractString = package_version())
    parts = (sweep.name, sweep.theta_0, sweep.amp_ratio, collect(deltas), collect(u_norm),
        cfg.optimizer, cfg.fit_coordinates, cfg.n_starts, cfg.rng_seed, cfg.g_tol,
        cfg.f_reltol,
        cfg.max_iterations, cfg.multi_start_parallel_scale,
        cfg.multi_start_transverse_scale, cfg.hessian_chunk, cfg.T_obs, cfg.f_min,
        cfg.f_max, version)
    text = sprint() do io
        for part in parts
            show(io, part)
            print(io, '|')
        end
    end
    return first(bytes2hex(sha256(text)), 16)
end

"""
$(TYPEDSIGNATURES)

Random stream of multi-start `k` at separation index `i` of the sweep
`name`, seeded from the SHA-256 of the master seed and the work-item
coordinates: every task draws an independent stream that depends on
neither thread scheduling nor the Julia release (`Base.hash` of strings
and tuples is not stable across releases and would silently change the
perturbations recorded under one `rng_seed`).
"""
function multi_start_stream(seed::Integer, name::AbstractString, i::Integer, k::Integer)
    digest = sha256(string(seed, '|', name, '|', i, '|', k))
    return Xoshiro(first(reinterpret(UInt64, digest)))
end

"""
$(TYPEDSIGNATURES)

Compare the fitted distances with the quartic law: the theoretical `D²`
(with the amplitude-ratio prefactor), the numerical/theoretical ratio, the
detected optimizer floor with its clean-point mask, the log-log slope
over the clean window and over the perturbative window of the correction fit
(`slope_window`), and the `O(δ⁵)` ratio-correction fit with its
validity radius. The fit summary is logged to `log_io`.
"""
function fit_sweep_law(deltas::AbstractVector, D2_num::AbstractVector, K_norm::Real,
    amp_prefactor::Real, cfg::PipelineSettings, log_io::IO)
    n = length(deltas)
    D2_theo = (amp_prefactor / 16.0) .* K_norm .* deltas .^ 4
    ratio = D2_num ./ D2_theo
    # the slope and ratio fits use only points strictly above the
    # detected optimizer floor (see optimizer_floor/above_floor_mask)
    floor_level = optimizer_floor(D2_num, ratio, cfg.floor_detection_ratio)
    clean = above_floor_mask(D2_num, floor_level)
    slope, slope_err =
        count(clean) >= MIN_FIT_POINTS ?
        loglog_slope(deltas[clean], D2_num[clean]) : (NaN, NaN)
    logline(
        log_io,
        @sprintf(
            "        fit points (above optimizer floor) %d/%d; fitted log-log slope %.4f ± %.4f",
            count(clean), n, slope, slope_err),
    )
    # the correction expansion is fitted only where the ratio is still
    # perturbatively close to unity (Fitting.perturbative_mask)
    perturbative = perturbative_mask(ratio, clean, cfg.correction_fit_max_departure)
    # the exponent of the leading-order law is fitted over the same
    # perturbative window as the O(δ⁵) correction: points beyond the
    # crossover to resolution are not described by the law they would be
    # fitted to
    slope_window, slope_window_err =
        count(perturbative) >= MIN_FIT_POINTS ?
        loglog_slope(deltas[perturbative], D2_num[perturbative]) : (NaN, NaN)
    c1, c1_err, c2 = ratio_correction_fit(deltas[perturbative], ratio[perturbative])
    logline(
        log_io,
        @sprintf("        O(δ⁵) fit window: %d/%d clean points within %.0f%% of the law",
            count(perturbative), count(clean), 100cfg.correction_fit_max_departure),
    )
    isfinite(slope_window) && logline(
        log_io,
        @sprintf(
            "        fitted log-log slope over the perturbative window %.4f ± %.4f (%d points)",
            slope_window, slope_window_err, count(perturbative)),
    )
    delta_valid = validity_radius(c1, c1_err, c2, cfg.correction_validity_fraction)
    isfinite(c1) &&
        logline(
            log_io,
            @sprintf(
                "        O(δ⁵) fit: ratio ≈ 1 + c₁δ + c₂δ² with c₁ = %.4g ± %.2g, c₂ = %.4g (%.0f%%-validity δ ≈ %.3g)",
                c1, c1_err, c2, 100cfg.correction_validity_fraction, delta_valid),
        )
    return (; D2_theo, ratio, floor_level, clean, slope, slope_err,
        slope_window, slope_window_err, c1, c1_err, c2, delta_valid)
end

"""
$(TYPEDSIGNATURES)

Persist a completed sweep into `out_dir`: the per-δ results table, the
residual spectrum at the primary evaluation point δ* (the clean separation
inside the fitted validity window nearest `2δ_min`) and, when it differs, the
threshold companion, the `sweep_meta.toml` record, and the scaling and
residual figures (a figure failure is logged and never loses the numerical
results). `geometry` carries `K_norm`, `g_uu` and `delta_min`; `fits` and
`law` are the outputs of [`optimize_separations`](@ref) and
[`fit_sweep_law`](@ref).
"""
function persist_sweep_results(out_dir::AbstractString, sweep::SweepSpec,
    deltas::AbstractVector, u_norm::AbstractVector, geometry, fits, law,
    ctx::RunContext)
    cfg = ctx.cfg
    wp = cfg.wp
    name = sweep.name
    theta0 = sweep.theta_0
    amp_ratio = sweep.amp_ratio
    rho_sq = sweep.rho_thresh^2
    n = length(deltas)
    (; K_norm, g_uu, delta_min) = geometry
    (; D2_num, best_fits) = fits
    (; D2_theo, clean, floor_level, slope, slope_err, slope_window, slope_window_err,
        c1, c1_err, c2, delta_valid) = law

    results_table = DataFrame(Delta = collect(deltas), D2_Numerical = D2_num,
        D2_Theoretical = D2_theo, K_u_Norm = fill(K_norm, n),
        BestFit_LuminosityDistance = best_fits[:, 1],
        BestFit_ChirpMass = best_fits[:, 2],
        BestFit_CoalescenceTime = best_fits[:, 3],
        BestFit_CoalescencePhase = best_fits[:, 4],
        BestFit_Spin1 = best_fits[:, 5], BestFit_Spin2 = best_fits[:, 6],
        Converged = collect(fits.converged), Iterations = fits.iteration_counts,
        GradNorm = fits.gradient_norms, AtBound = collect(fits.at_bound),
        Starts = fill(cfg.n_starts, n), MultiStartGain = fits.multi_start_gain)
    CSV.write(backup_existing!(joinpath(out_dir, "results.csv")), results_table)

    # Residual-spectrum evaluation points. δ*: the clean separation with
    # δ ≤ delta_valid nearest 2 δ_min (D² ≈ 16 ρ²), the top of the window the
    # quartic law describes; its residual is the normal projection the law
    # integrates. Threshold companion: the separation nearest D² = ρ².
    pos = findall(>(0), D2_num)
    idx_thr = isempty(pos) ? n : pos[argmin(abs.(log10.(D2_num[pos] ./ rho_sq)))]
    in_validity = isfinite(delta_valid) ? (deltas .<= delta_valid) : trues(n)
    valid_pos = findall(clean .& in_validity .& (D2_num .> 0))
    idx_star =
        isempty(valid_pos) ? idx_thr :
        valid_pos[argmin(abs.(log10.(deltas[valid_pos] ./ (2delta_min))))]
    d_star = deltas[idx_star]
    spec, meta = residual_spectrum(theta0, u_norm, amp_ratio, d_star,
        best_fits[idx_star, :], ctx.freqs, ctx.Sn, ctx.df, wp;
        n_windows = cfg.residual_spectrum_windows)
    CSV.write(backup_existing!(joinpath(out_dir, "residual_spectrum.csv")), spec)
    spec_thr = nothing
    thr_meta = Dict{String,Any}()
    if idx_thr != idx_star
        spec_thr, meta_thr = residual_spectrum(theta0, u_norm, amp_ratio,
            deltas[idx_thr], best_fits[idx_thr, :], ctx.freqs, ctx.Sn, ctx.df,
            wp; n_windows = cfg.residual_spectrum_windows)
        CSV.write(
            backup_existing!(joinpath(out_dir, "residual_spectrum_threshold.csv")),
            spec_thr)
        thr_meta = Dict{String,Any}("delta_thr" => deltas[idx_thr],
            "d2_num_thr" => D2_num[idx_thr], "d2_theo_thr" => D2_theo[idx_thr],
            "residual_thr_d2_integral_A" => meta_thr.int_A,
            "residual_thr_d2_integral_E" => meta_thr.int_E)
    end
    open(backup_existing!(joinpath(out_dir, "sweep_meta.toml")), "w") do io
        TOML.print(
            io,
            Dict(
                "name" => name, "delta_star" => d_star,
                "d2_num_star" => D2_num[idx_star], "d2_theo_star" => D2_theo[idx_star],
                "K_u_norm" => K_norm, "g_uu_raw" => g_uu, "rho_sq" => rho_sq,
                "df" => ctx.df, "f_min" => cfg.f_min, "f_max" => cfg.f_max,
                "delta_min" => delta_min, "amp_ratio" => amp_ratio,
                "slope" => slope, "slope_err" => slope_err,
                "slope_window" => isnan(slope_window) ? -1.0 : slope_window,
                "slope_window_err" => isnan(slope_window_err) ? -1.0 : slope_window_err,
                "c1" => c1, "c1_err" => c1_err, "c2" => c2,
                "delta_valid" => delta_valid,
                "floor_level" => isnan(floor_level) ? -1.0 : floor_level,
                "residual_d2_integral_A" => meta.int_A,
                "residual_d2_integral_E" => meta.int_E,
                thr_meta...),
        )
    end

    try
        # the legend quotes the perturbative-window exponent; the all-clean
        # slope stays in sweep_meta.toml
        fig = scaling_figure(collect(deltas), D2_num, D2_theo;
            rho_sq = rho_sq, delta_min = delta_min,
            slope = isfinite(slope_window) ? slope_window : slope,
            slope_err = isfinite(slope_window) ? slope_window_err : slope_err,
            clean = collect(clean),
            floor_level = floor_level, c1 = c1, c2 = c2)
        save_figure(fig, joinpath(out_dir, "scaling_plot"))
        rfig = residual_figure(
            spec,
            ResidualFigureMeta(d_star, ctx.df, D2_num[idx_star], D2_theo[idx_star]),
        )
        save_figure(rfig, joinpath(out_dir, "residual_plot"))
        if spec_thr !== nothing
            rfig_thr = residual_figure(
                spec_thr,
                ResidualFigureMeta(deltas[idx_thr], ctx.df,
                    D2_num[idx_thr], D2_theo[idx_thr]);
                delta_symbol = "\\delta_{\\mathrm{thr}}",
            )
            save_figure(rfig_thr, joinpath(out_dir, "residual_plot_threshold"))
        end
    catch err
        err isa Union{InterruptException,OutOfMemoryError} && rethrow()
        @warn "Sweep '$name': figure generation failed; numerical results are saved." exception =
            (err, catch_backtrace())
        note_figure_failure!(ctx.out_base, stage_key(:sweep, name))
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Stage 2: two-dimensional confusion mapping (mirrored polar sampling, prior capping)
# -----------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Execute one 2D confusion map: tangent basis at the base point, mirrored
angular sweep with adaptive refinement ([`refine_directions!`](@ref)),
exact prior-wall and box-corner vertices
([`insert_crossover_vertices!`](@ref), [`insert_box_corner_vertices!`](@ref)),
mirroring with prior capping ([`mirror_to_full_circle`](@ref)), and
persistence of the contour table and zone figure into the run directory
([`persist_map_results`](@ref)).
"""
function run_map(map_spec::MapSpec, idx::Int, total::Int, ctx::RunContext)
    cfg = ctx.cfg
    wp = cfg.wp
    name = map_spec.name
    px = map_spec.param_x
    py = map_spec.param_y
    rho_sq = map_spec.rho_thresh^2
    theta0 = map_spec.theta_0
    n_angles = map_spec.n_angles

    out_dir = joinpath(ctx.out_base, "maps", name)
    mkpath(out_dir)
    log_io = open(joinpath(out_dir, "mapping.log"), "a")
    t0 = time()
    try
        box = deviation_box(cfg.bounds, theta0, px, py)
        logline(log_io, "─"^78)
        logline(log_io, "[Map $idx/$total] $name")
        logline(log_io, "  plane      : parameter $px vs parameter $py")
        logline(log_io, "  rho^2      : $rho_sq")
        logline(
            log_io,
            "  prior box  : Δx ∈ [$(box[1]), $(box[2])], Δy ∈ [$(box[3]), $(box[4])]",
        )
        logline(log_io, "  [1/2] tangent basis (Jacobian at base point)")
        basis = compute_tangent_basis(theta0, ctx.freqs, ctx.Sn, ctx.df, wp)
        logline(log_io, "        basis rank: $(length(basis)) of $(length(theta0))")
        # Directions are sampled uniformly in the Fisher-normalized plane
        # (each axis measured in its own σ at the base point) and mapped to
        # parameter units: a plane whose axes differ by five orders of
        # magnitude in σ (a coalescence time against a phase for a loud
        # source) is a needle in parameter units that a uniform angular grid
        # would resolve only along its axes.
        sigma_x, sigma_y =
            axis_fisher_scales(theta0, px, py, basis, ctx, cfg.g_uu_degenerate)
        logline(
            log_io,
            @sprintf("        axis Fisher σ: %.3e × %.3e (sampling anisotropy %.2e)",
                sigma_x, sigma_y, max(sigma_x, sigma_y) / min(sigma_x, sigma_y)),
        )
        direction = phi -> unit_direction(phi, sigma_x, sigma_y)
        # boundary radius in the normalized plane (constant for an ellipse
        # aligned with the σ scales): the refinement criterion lives here
        normalized_radius(phi) =
            (cs = direction(phi); hypot(cs[1] / sigma_x, cs[2] / sigma_y))

        n_base_directions = n_angles ÷ 2
        # milestones count the base directions; directions added by
        # refinement and bisection log no further milestones
        note_progress = stage_progress_hook(
            "Map '$name' ($idx/$total): angular sweep", n_base_directions,
            cfg.progress_log_fraction)
        prog = ProgressUnknown(desc = "  mapping: ", enabled = progress_enabled())

        # batched (K, g) evaluation of half-circle directions; every helper
        # below samples the plane through this closure only
        function eval_angles(phis::Vector{Float64})
            out = Vector{NTuple{2,Float64}}(undef, length(phis))
            parallel_foreach(length(phis), ctx.map_tasks) do i
                dir = zeros(N_PARAMS)
                dir[px], dir[py] = direction(phis[i])
                K, g = compute_extrinsic_curvature_from_basis(theta0, dir, basis,
                    ctx.freqs, ctx.Sn, ctx.df, wp)
                out[i] = (K, g)
                note_item!()
                next!(prog)
                note_progress()
            end
            return out
        end

        logline(
            log_io,
            "  [2/2] mirrored angular sweep ($n_base_directions base directions " *
            "on [0, π), adaptive refinement tol $(cfg.neighbor_ratio_tol), " *
            "$(cfg.max_refine_levels) levels; $(ctx.map_tasks) concurrent)",
        )
        phis = [(k - 1) * π / n_base_directions for k in 1:n_base_directions]
        curvature_norm_pairs = eval_angles(phis)
        entries = [
            (phi = phis[i], K = curvature_norm_pairs[i][1],
                g = curvature_norm_pairs[i][2],
                wall = 0x00) for i in 1:n_base_directions
        ]

        r_math_of(K) = boundary_radius(K, rho_sq)
        r_box_of(phi) = ray_box_crossing(direction(phi)..., box...)
        # capped radii of a half-circle direction and of its mirror image in
        # the normalized plane: the prior box need not be symmetric, so both
        # halves drive refinement
        function r_cap_pair(e)
            scale = normalized_radius(e.phi)
            return (min(r_math_of(e.K), r_box_of(e.phi)) * scale,
                min(r_math_of(e.K), r_box_of(e.phi + π)) * scale)
        end

        added = refine_directions!(entries, eval_angles, r_cap_pair;
            max_levels = cfg.max_refine_levels, ratio_tol = cfg.neighbor_ratio_tol)
        logline(
            log_io,
            "        refinement added $added directions " *
            "($(length(entries)) on the half-circle)",
        )

        if cfg.corner_bisect_iters > 0
            capped_at = (alpha, K) -> begin
                rb = r_box_of(alpha) # cos/sin of the full-circle angle
                isfinite(rb) && r_math_of(K) >= rb
            end
            n_brackets, n_vertices = insert_crossover_vertices!(entries, eval_angles,
                capped_at, cfg.corner_bisect_iters)
            n_brackets > 0 && logline(
                log_io,
                @sprintf(
                    "        corner bisection: %d crossover(s) located, %d exact corner vertex(es) inserted (%d iterations)",
                    n_brackets, n_vertices, cfg.corner_bisect_iters),
            )
            n_corners = insert_box_corner_vertices!(entries, eval_angles, capped_at, box,
                (cx, cy) -> atan(cy / sigma_y, cx / sigma_x))
            n_corners > 0 && logline(
                log_io,
                @sprintf("        box-corner vertices: %d inserted", n_corners),
            )
        end
        finish!(prog)

        polar = mirror_to_full_circle(entries, box, r_math_of, cfg.g_uu_degenerate;
            direction = direction)
        r_cap = polar.r_cap
        n_unbounded, polygon_cap = cap_unbounded_radii!(r_cap, cfg.unbounded_cap_factor)
        n_unbounded > 0 &&
            @warn "Map '$name': $n_unbounded directions are unbounded " *
                  "(no curvature limit and no finite physical bound); capping them at " *
                  "$polygon_cap for the polygon. Consider adding [parameter_bounds]."

        prior_frac = count(polar.prior_limited) / length(polar.prior_limited)
        degen_frac = count(polar.degenerate) / length(polar.degenerate)
        logline(
            log_io,
            @sprintf("        prior-limited directions : %.1f%%", 100prior_frac),
        )
        degen_frac > 0 &&
            logline(
                log_io,
                @sprintf("        degenerate directions    : %.1f%%", 100degen_frac),
            )
        if px in SPIN_INDICES && py in SPIN_INDICES &&
           prior_frac > SPIN_PRIOR_NOTE_FRACTION && cfg.wp.phase_pn_order == 1.5
            logline(
                log_io,
                "  [note] the 1.5PN phase depends on the spins only through β; " *
                "along the combination with dβ = 0 (χ_a at equal mass, a tilted " *
                "line otherwise) the manifold is exactly flat, so the zone there " *
                "is limited by the physical spin prior [-1, 1], not by curvature.",
            )
        end
        if px in SPIN_INDICES && py in SPIN_INDICES &&
           prior_frac > SPIN_PRIOR_NOTE_FRACTION && cfg.wp.phase_pn_order == 2
            logline(
                log_io,
                "  [note] at 2PN order the spins enter the phase through β and σ; " *
                "a spin zone limited by the prior marks a base point on or near " *
                "the line where their gradients are parallel (zero spins; equal " *
                "spins at equal mass), on which one spin combination is flat to " *
                "first order.",
            )
        end
        if px in SPIN_INDICES && py in SPIN_INDICES &&
           prior_frac > SPIN_PRIOR_NOTE_FRACTION && cfg.wp.phase_pn_order >= 2.5
            logline(
                log_io,
                "  [note] above 2PN order the spin gradients are independent except " *
                "at equal mass with equal spins, where χ_a is flat to first order; " *
                "elsewhere a spin zone limited by the prior means that the spin " *
                "combination fixed only by the higher-order terms is constrained " *
                "too weakly for the zone to close inside [-1, 1].",
            )
        end

        if cfg.monitoring_enabled && progress_enabled()
            println(stdout, map_diagnostic_panel(polar.angle, r_cap, prior_frac))
        end

        # the figure quotes fractions of the sampling angle, not of the vertex count
        prior_angle_frac = angular_fraction(polar.angle, polar.prior_limited)
        degen_angle_frac = angular_fraction(polar.angle, polar.degenerate)
        persist_map_results(out_dir, map_spec, box, polar, prior_angle_frac,
            degen_angle_frac, display_scale(wp))

        elapsed = format_time(time() - t0)
        logline(log_io, "  [done] map '$name' completed in $elapsed")
        # structured run.log record (see the sweep counterpart)
        @info "Map '$name' ($idx/$total) completed in $elapsed " *
              "(prior-limited $(round(100prior_frac, digits = 1))%)."
    finally
        close(log_io)
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Fisher scales `(σ_x, σ_y) = (g_xx^{-1/2}, g_yy^{-1/2})` of the two axes of
a map plane at `theta0`, from the directional Fisher norms of the unit
axis vectors (two curvature evaluations). An axis whose Fisher norm falls
below `g_uu_degenerate` (a null direction) gets the scale 1, so the
normalization stays finite.
"""
function axis_fisher_scales(theta0::AbstractVector, px::Int, py::Int, basis::Vector,
    ctx::RunContext, g_uu_degenerate::Real)
    scales = ntuple(2) do k
        dir = zeros(N_PARAMS)
        dir[k == 1 ? px : py] = 1.0
        _, g = compute_extrinsic_curvature_from_basis(theta0, dir, basis, ctx.freqs,
            ctx.Sn, ctx.df, ctx.cfg.wp)
        (isfinite(g) && g > g_uu_degenerate) ? 1 / sqrt(g) : 1.0
    end
    return scales
end

"""
$(TYPEDSIGNATURES)

Persist a completed map into `out_dir`: the contour table
(`confusion_contour.csv`) from the capped polar table `polar`
([`mirror_to_full_circle`](@ref), unbounded directions already capped for
the polygon) and the zone figure in the display units of `scale` (a figure
failure is logged and never loses the numerical results).
"""
function persist_map_results(out_dir::AbstractString, map_spec::MapSpec,
    box::NTuple{4,Float64}, polar, prior_frac::Real, degen_frac::Real,
    scale::NTuple{6,Float64})
    name = map_spec.name
    px = map_spec.param_x
    py = map_spec.param_y
    X = polar.r_cap .* polar.dir_cos
    Y = polar.r_cap .* polar.dir_sin
    contour_table = DataFrame(Angle = polar.angle, X_Bound = X, Y_Bound = Y,
        Dir_Cos = polar.dir_cos, Dir_Sin = polar.dir_sin,
        R_Capped = polar.r_cap, R_Math = polar.r_math, R_Box = polar.r_box,
        Prior_Limited = collect(polar.prior_limited),
        Degenerate = collect(polar.degenerate),
        K_Raw = polar.K_raw, G_uu = polar.g_uu)
    CSV.write(backup_existing!(joinpath(out_dir, "confusion_contour.csv")), contour_table)

    try
        fig = zone_figure(X, Y, collect(polar.prior_limited);
            px = px, py = py, box = box,
            prior_frac = prior_frac, degenerate_frac = degen_frac,
            x_math = polar.r_math .* polar.dir_cos,
            y_math = polar.r_math .* polar.dir_sin, scale = scale)
        save_figure(fig, joinpath(out_dir, "confusion_zone"))
    catch err
        err isa Union{InterruptException,OutOfMemoryError} && rethrow()
        @warn "Map '$name': figure generation failed; numerical results are saved." exception =
            (err, catch_backtrace())
        # out_dir is <run>/maps/<name>
        note_figure_failure!(dirname(dirname(out_dir)), stage_key(:map, name))
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Pipeline entry point
# -----------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Logger of a run-log file: one line per record, `[yyyy-mm-dd HH:MM:SS] LEVEL
message`, followed by the exception of the record and its backtrace when it
carries one; records below `Info` are dropped.
"""
function run_log_logger(io::IO)
    formatter = FormatLogger(io) do stream, record
        println(stream, "[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ",
            uppercase(string(record.level)), " ", record.message)
        # the exception of a failed stage and its backtrace belong in the file
        for (key, value) in record.kwargs
            key === :exception || continue
            err, trace = value isa Tuple ? value : (value, nothing)
            trace === nothing ? showerror(stream, err) : showerror(stream, err, trace)
            println(stream)
        end
    end
    return MinLevelLogger(formatter, Logging.Info)
end

"""
$(TYPEDSIGNATURES)

Run the pipeline: validate the TOML configuration (hard errors on unusable
input), allocate hardware within the `[safety]` memory budget, snapshot the
configuration and provenance metadata into a configuration-hashed run
directory, then execute the sweep and mapping stages. Each sweep and map is
guarded individually: a failing stage is logged with its backtrace and the
remaining stages continue, unless the device no longer responds, which ends the
attempt. `output_dir` is resolved relative to `project_root`.

Returns the run directory when every stage completed. When stages failed, the
run is finished first (metadata written, log closed) and a
[`PipelineStageError`](@ref) naming them is thrown. `InterruptException` and
`OutOfMemoryError` are never absorbed by the stage guards. A second process
that targets the same run directory stops with an `ArgumentError`.

Keywords:

  - `resume = false`: continue the unfinished run of this configuration
    (`Provenance.resolve_run_dir`) instead of creating a new sibling
    directory. Stages recorded as complete are skipped, a sweep continues from
    its work-item checkpoint, stages abandoned by a supervisor are skipped, and
    the logs are appended to. A run that is already complete is returned as is.
    Only runs produced by the same package version are continued.
  - `run_dir = nothing`: use exactly this run directory (created when absent);
    implies `resume`. An `ArgumentError` is thrown when the directory was
    produced by another package version (`Provenance.check_package_version`)
    or holds the snapshot of a different configuration.
  - `heartbeat_period_s = 0`: when positive, publish `heartbeat.toml` in the
    run directory at this period for a supervising process.
"""
function run_pipeline(config_path::AbstractString, project_root::AbstractString,
    output_dir::AbstractString; resume::Bool = false,
    run_dir::Union{Nothing,AbstractString} = nothing, heartbeat_period_s::Real = 0)
    cfg = load_and_validate_config(config_path)
    # a required GPU is checked before any output exists: the probe is
    # idempotent and repeated inside _run_pipeline
    cfg.require_gpu &&
        get_best_backend(prefer = cfg.gpu_backend) isa KernelAbstractions.CPU &&
        throw(
            ArgumentError(
                "[hardware].require_gpu = true but no functional GPU backend resolved for " *
                "gpu_backend = \"$(cfg.gpu_backend)\" (the vendor package must be loaded from " *
                "the default environment and its device functional); no run directory was created.",
            ),
        )
    base_dir = joinpath(project_root, output_dir)
    if run_dir !== nothing
        out_base = String(run_dir)
        check_package_version(out_base)
        snapshot = joinpath(out_base, "config.toml")
        (
            isfile(snapshot) &&
            continuation_config(effective_config(snapshot)) !=
            continuation_config(effective_config(config_path))
        ) && throw(
            ArgumentError(
                "$out_base holds the run of another configuration: its config.toml " *
                "differs from $config_path in more than thread, memory, monitoring " *
                "or supervision settings",
            ),
        )
        mkpath(out_base)
    elseif resume
        out_base, state = resolve_run_dir(base_dir, config_path)
        if state === :complete
            @info "Run already complete: $out_base"
            return out_base
        end
    else
        out_base = unique_run_dir(base_dir, run_id_from_config(config_path))
    end
    lock = acquire_run_lock(out_base)
    failures = try
        isfile(joinpath(out_base, "config.toml")) || snapshot_config(config_path, out_base)
        isfile(joinpath(out_base, "Manifest.toml")) || snapshot_manifest(out_base)

        log_stream = open(joinpath(out_base, "run.log"), "a")
        tee = TeeLogger(global_logger(), run_log_logger(log_stream))
        beat =
            heartbeat_period_s > 0 ?
            start_heartbeat(joinpath(out_base, HEARTBEAT_FILE), heartbeat_period_s) :
            nothing
        try
            with_logger(tee) do
                _run_pipeline(cfg, project_root, out_base, config_path)
            end
        finally
            beat === nothing || stop_heartbeat!(beat)
            close(log_stream)
        end
    finally
        close(lock)
    end
    isempty(failures) || throw(PipelineStageError(out_base, failures))
    return out_base
end

function _run_pipeline(cfg::PipelineSettings, project_root::AbstractString,
    out_base::AbstractString, config_path::AbstractString)
    start_time = time()
    df = 1.0 / cfg.T_obs
    freqs = collect(cfg.f_min:df:cfg.f_max)
    Sn = analytic_noise_psd.(freqs; noise = cfg.noise)
    n_ch = n_channels(cfg.wp)

    backend = get_best_backend(prefer = cfg.gpu_backend)
    cfg.gpu_backend === :none &&
        @info "[hardware].gpu_backend = \"none\": GPU detection bypassed, running on the CPU backend."
    if !(backend isa KernelAbstractions.CPU) && n_channels(cfg.wp) == 3
        throw(
            ArgumentError(
                "GPU backends support the 2-channel (A, E) configuration only; " *
                "set [physics].include_t_channel = false (T is identically zero).",
            ),
        )
    end

    sweep_tasks, map_tasks, est_gb = plan_resources(cfg, length(freqs), n_ch, backend)

    freqs_dev = backend isa KernelAbstractions.CPU ? freqs : to_backend(freqs, backend)
    Sn_dev = backend isa KernelAbstractions.CPU ? Sn : to_backend(Sn, backend)
    previous = read_run_metadata(out_base)
    attempt = Int(get(previous, "attempts", 0)) + 1
    ctx = RunContext(cfg, out_base, freqs, Sn, freqs_dev, Sn_dev, df, backend,
        sweep_tasks, map_tasks, attempt)
    elapsed_before = Float64(get(previous, "elapsed_seconds", 0.0))

    write_run_metadata(out_base;
        attempts = attempt,
        started = get(previous, "started", string(now())),
        run_id = basename(out_base), git = git_state(project_root),
        package_version = package_version(),
        config_file = basename(config_path),
        base_config = string(get(TOML.parsefile(config_path), "base_config", "")),
        julia_version = string(VERSION), hostname = gethostname(),
        backend = backend_name(backend),
        cpu_model = Backends.cpu_model(),
        cpu_threads = Sys.CPU_THREADS,
        total_memory_gb = round(Sys.total_memory() / 2^30, digits = 1),
        julia_threads = Threads.nthreads(), active_tasks = sweep_tasks,
        map_tasks = map_tasks,
        n_frequency_bins = length(freqs), channels = n_ch,
        estimated_ram_gb = round(est_gb, digits = 2),
        optimizer = string(cfg.optimizer),
        n_starts = cfg.n_starts, rng_seed = cfg.rng_seed,
        confusion_noise = cfg.noise.confusion_enabled)
    record_attempt!(out_base, attempt; git = git_state(project_root),
        backend = backend_name(backend))
    isfile(joinpath(out_base, "hardware.txt")) || write_hardware_fingerprint(out_base;
        device_report = Backends.device_fingerprint(backend))

    println("=" ^ 78)
    println("  CurvatureDistinguishability Pipeline")
    println("=" ^ 78)
    @info "Run directory: $out_base"
    @info "Grid: $(length(freqs)) bins ($(cfg.f_min) – $(cfg.f_max) Hz, df = $df); channels: $n_ch"
    @info "Backend: $(backend_name(backend)); concurrency: $sweep_tasks sweep / " *
          "$map_tasks map tasks; estimated peak RAM $(round(est_gb, digits = 2)) GB " *
          "(budget $(cfg.max_ram_gb) GB)"
    @info "Noise: per-channel Robson Eq.12 + confusion $(cfg.noise.confusion_enabled ? "Eq.14 × Eq.9 response" : "disabled")"
    @info "Optimizer: $(cfg.optimizer) in $(cfg.fit_coordinates) coordinates " *
          "with physical bounds"

    attempt > 1 && @info "Continuing the run: attempt $attempt."
    failures = String[]
    done = Set(completed_stages(out_base))
    abandoned = Set(abandoned_stages(out_base))
    skip_stage(key) =
        key in done ? (@info "Stage '$key' is complete; skipped."; true) :
        key in abandoned ?
        (@warn "Stage '$key' was abandoned by the supervisor; skipped."; true) :
        false
    device_lost = false
    if cfg.run_1d_sweeps && !isempty(cfg.sweeps)
        @info ">>> 1D parameter sweeps ($(length(cfg.sweeps)) configurations)"
        for (i, s) in enumerate(cfg.sweeps)
            key = stage_key(:sweep, s.name)
            skip_stage(key) && continue
            try
                run_sweep(s, i, length(cfg.sweeps), ctx)
                mark_stage_complete!(out_base, key)
            catch err
                is_fatal(err) && rethrow()
                push!(failures, s.name)
                @error "Sweep '$(s.name)' failed; continuing with remaining stages." exception =
                    (err, catch_backtrace())
                # a reset or wedged device context fails every later launch of
                # this process: end the attempt instead of failing the rest
                if !device_responds(ctx.backend)
                    device_lost = true
                    @error "The device no longer responds; the remaining stages " *
                           "need a new process."
                    break
                end
            end
            maintain_memory!(ctx.backend, cfg.gc_between_stages)
        end
    end
    if cfg.run_2d_mapping && !isempty(cfg.maps) && !device_lost
        @info ">>> 2D confusion mapping ($(length(cfg.maps)) configurations)"
        ctx.map_tasks != ctx.sweep_tasks &&
            @info "2D mapping: $(ctx.map_tasks) concurrent tasks."
        for (i, m) in enumerate(cfg.maps)
            key = stage_key(:map, m.name)
            skip_stage(key) && continue
            try
                run_map(m, i, length(cfg.maps), ctx)
                mark_stage_complete!(out_base, key)
            catch err
                is_fatal(err) && rethrow()
                push!(failures, m.name)
                @error "Map '$(m.name)' failed; continuing with remaining stages." exception =
                    (err, catch_backtrace())
            end
            maintain_memory!(ctx.backend, cfg.gc_between_stages)
        end
    end

    elapsed = time() - start_time
    # elapsed_seconds accumulates over the attempts of a continued run
    write_run_metadata(out_base; finished = string(now()),
        elapsed_seconds = round(elapsed_before + elapsed, digits = 1),
        failed_stages = join(failures, ","))
    println("=" ^ 78)
    unrendered = String.(get(read_run_metadata(out_base), "figure_failures", String[]))
    isempty(unrendered) ||
        @warn "Figures of $(join(unrendered, ", ")) were not rendered; rebuild them with " *
              "scripts/replot.jl once the cause in run.log is resolved."
    if isempty(failures)
        @info "Pipeline complete in $(format_time(elapsed)); results in $out_base"
    else
        @warn "Pipeline finished in $(format_time(elapsed)) with failed stages: " *
              "$(join(failures, ", ")); see run.log for backtraces. Results in $out_base"
    end
    println("=" ^ 78)
    return failures
end

end # module
