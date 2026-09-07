"""
Pipeline driver: resource planning against the [safety] budget, the 1D
sweep and 2D confusion-mapping stages, structured logging, opt-in
monitoring and per-stage guardrails.
"""
module Orchestrator

using DocStringExtensions: TYPEDSIGNATURES
using Printf: @sprintf
using Dates: Dates, now
using TOML: TOML
using CSV: CSV
using DataFrames: DataFrame
using Random: Xoshiro
using ProgressMeter: Progress, ProgressUnknown, finish!, next!
using Logging: Logging, global_logger, with_logger
using LoggingExtras: FormatLogger, MinLevelLogger, TeeLogger
using KernelAbstractions: KernelAbstractions

using ..Backends
using ..Physics
using ..Physics: N_PARAMS
using ..Detector
using ..Residuals
using ..Geometry
using ..Inference
using ..Bounds
using ..Config
using ..Provenance
using ..Plotting
using ..Plotting: map_diagnostic_panel, sweep_diagnostic_panel
using ..Fitting:
    MIN_FIT_POINTS, above_floor_mask, loglog_slope, optimizer_floor,
    ratio_correction_fit

export run_pipeline

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

# angular tolerance below which two map directions are the same vertex
const ANGLE_DEDUPE_TOL = 1e-10
# prior-limited boundary fraction above which the spin-plane χ_eff
# degeneracy advisory is logged
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
    println(stdout, msg)
    flush(stdout)
    println(io, stamped)
    flush(io)
    return nothing
end

progress_enabled() = stderr isa Base.TTY

"""
Bounded-concurrency parallel foreach: at most `ntasks` of the spawned tasks
run simultaneously. Exceptions propagate to the caller (per-stage guardrails
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

"""
$(TYPEDSIGNATURES)

Pre-flight memory estimate against the `[safety]` budget: refuses to start
when even a single task exceeds it, downscales concurrency otherwise, and
applies the VRAM budget on GPU backends. Resource planning is stage-aware:
the returned `(sweep_tasks, map_tasks, estimated_gb)` distinguishes the
kernel-dispatch sweep stage (single-task on GPU backends) from the 2D
mapping stage, which evaluates host-side automatic differentiation on every
backend and therefore always keeps the multi-threaded CPU concurrency.
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
        error(
            "Estimated memory for a single task ($(round(fixed / 2^30, digits = 2)) GB fixed + " *
            "$(round(per_task / 2^30, digits = 2)) GB/task) exceeds [safety].max_ram_gb = " *
            "$(cfg.max_ram_gb) GB. Reduce the frequency grid ([grid]) or raise the budget explicitly.",
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
        # GPU sweep execution is single-task by design: GPU drivers are not
        # reliably safe under concurrent multi-task access (observed Level
        # Zero segmentation fault), the library-level GPU lock serializes
        # kernel launches, and the device itself serializes kernels —
        # additional tasks provide no throughput. The pin applies to the
        # sweep stage only: 2D mapping never dispatches kernels, and pinning
        # it single-threaded quadrupled its wall time in production.
        sweep_tasks = 1
        map_tasks > 1 &&
            @info "GPU backend active: sweep-stage concurrency pinned to 1 task " *
                  "(kernels serialize on the device; concurrent driver access is " *
                  "unsafe); the CPU-only 2D mapping stage keeps $map_tasks tasks."
        vram_budget = max(0.0, cfg.max_vram_gb - cfg.os_vram_overhead_gb) * 2^30
        need = n_bins * cfg.bytes_per_bin_per_task_gpu
        need <= vram_budget ||
            error(
                "VRAM budget ($(round(vram_budget / 2^30, digits = 2)) GB usable) is below " *
                "the footprint of one GPU task " *
                "($(round(need / 2^30, digits = 2)) GB at " *
                "$(cfg.bytes_per_bin_per_task_gpu) bytes/bin). Reduce the grid or raise " *
                "[safety].max_vram_gb.",
            )
    end

    return sweep_tasks, map_tasks, (fixed + map_tasks * per_task) / 2^30
end

"""
$(TYPEDSIGNATURES)

Inter-stage memory maintenance for long campaigns (`[hardware]
.gc_between_stages`): flushes the device-buffer cache on GPU backends, runs
a full host garbage collection (device arrays are freed by their host
finalizers, so an explicit collection is what actually returns device
memory between stages), and asks the backend to reclaim pooled device
memory where an API exists.
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
final item — completion has its own log line). `fraction <= 0` disables the
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
Immutable per-run context threaded through the sweep and map stages:
validated configuration, output base directory, frequency grid and PSD
(host and device copies), and the planned stage concurrencies (the sweep
stage is pinned to one task on GPU backends; the CPU-only mapping stage is
not). Parametric over the backend and device-array types so every per-δ
solve closure captures concrete fields.
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
end

# -----------------------------------------------------------------------------
# Module 1: 1D parameter sweeps (δ⁴ validation)
# -----------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Execute one 1D separation sweep: directional geometry (K(u), g(u,u)),
per-δ box-constrained optimization with optional multi-start, floor
detection and slope/correction fits, and persistence of the results table,
residual spectrum, metadata and figures into the run directory.
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
    log_io = open(joinpath(out_dir, "sweep.log"), "w")
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
                amp_ratio, amp_prefactor)
        )

        logline(log_io, "  [1/3] manifold geometry (K(u), g(u,u))")
        K_u, g_uu =
            compute_extrinsic_curvature(theta0, u_raw, ctx.freqs, ctx.Sn, ctx.df, wp)
        logline(log_io, @sprintf("        raw Fisher norm g(u,u)   : %.6e", g_uu))
        if g_uu < cfg.g_uu_degenerate
            logline(
                log_io,
                "  [WARN] direction is degenerate (g(u,u) below " *
                "$(cfg.g_uu_degenerate)); the sources cannot be separated " *
                "along it. Skipping sweep.",
            )
            return nothing
        end
        norm_scale = 1.0 / sqrt(g_uu)
        u_norm = u_raw .* norm_scale
        K_norm = K_u * norm_scale^4
        delta_min = (16.0 * rho_sq / K_norm)^(1 / 4)
        logline(log_io, @sprintf("        normalized K(u)          : %.6e", K_norm))
        logline(log_io, @sprintf("        delta_min                : %.6e", delta_min))

        deltas = 10 .^ range(cfg.min_log_delta, cfg.max_log_delta, length = cfg.n_deltas)

        # verify the second source stays within the physical bounds at δ_max
        theta_far = theta0 .+ maximum(deltas) .* u_norm
        theta_far[1] = amp_ratio * theta0[1]
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
        D2_num = zeros(n)
        best_fits = zeros(n, N_PARAMS)
        converged = falses(n)
        iteration_counts = zeros(Int, n)
        gradient_norms = zeros(n)
        at_bound = falses(n)
        multi_start_gain = ones(n)

        logline(
            log_io,
            "  [2/3] optimization sweep ($(n) separations, " *
            "$(ctx.sweep_tasks) concurrent, optimizer = $(cfg.optimizer)" *
            (cfg.n_starts > 1 ? ", $(cfg.n_starts) starts" : "") * ")",
        )
        prog = Progress(n; desc = "  optimizing: ", enabled = progress_enabled())
        note_progress = stage_progress_hook(
            "Sweep '$name' ($idx/$total)", n, cfg.progress_log_fraction)
        parallel_foreach(n, ctx.sweep_tasks) do i
            d = deltas[i]
            p2 = theta0 .+ d .* u_norm
            p2[1] = amp_ratio * theta0[1]
            h1 = scaled_waveform_model(theta0, ctx.freqs, wp)
            h2 = scaled_waveform_model(p2, ctx.freqs, wp)
            ch1 = project_to_tdi(h1, ctx.freqs, theta0, wp)
            ch2 = project_to_tdi(h2, ctx.freqs, p2, wp)
            data = map((a, b) -> a .+ b, ch1, ch2)
            if !(ctx.backend isa KernelAbstractions.CPU)
                data = map(a -> to_backend(a, ctx.backend), data)
            end
            # degenerate effective source: amplitude (1+q)A at the weighted
            # midpoint, with q = A₂/A₁ the amplitude ratio
            guess = theta0 .+ (amp_ratio / (1 + amp_ratio) * d) .* u_norm
            guess[1] = (1 + amp_ratio) * theta0[1]

            freqs_active =
                ctx.backend isa KernelAbstractions.CPU ? ctx.freqs : ctx.freqs_dev
            Sn_active = ctx.backend isa KernelAbstractions.CPU ? ctx.Sn : ctx.Sn_dev
            solve(x0) =
                calculate_numerical_distance(data, x0, freqs_active, Sn_active, ctx.df;
                    g_tol = cfg.g_tol,
                    iterations = cfg.max_iterations,
                    backend = ctx.backend,
                    optimizer = cfg.optimizer,
                    bounds = cfg.bounds,
                    hessian_chunk = cfg.hessian_chunk,
                    wp = wp)
            dist, best, res = solve(guess)
            dist_canonical = dist
            for k in 2:cfg.n_starts
                # deterministic per-task stream: independent of thread scheduling
                rng = Xoshiro(hash((cfg.rng_seed, name, i, k)))
                pert =
                    guess .+ (cfg.multi_start_parallel_scale * d * randn(rng)) .* u_norm .+
                    cfg.multi_start_transverse_scale .* randn(rng, N_PARAMS)
                dist_k, best_k, res_k = solve(clamp_interior(pert, cfg.bounds))
                if dist_k < dist
                    dist, best, res = dist_k, best_k, res_k
                end
            end
            diagnostics = optimization_diagnostics(res, best, cfg.bounds)
            D2_num[i] = dist
            multi_start_gain[i] = dist > 0 ? dist_canonical / dist : 1.0
            best_fits[i, :] .= best
            converged[i] = diagnostics.converged
            iteration_counts[i] = diagnostics.iterations
            gradient_norms[i] = diagnostics.g_norm
            at_bound[i] = diagnostics.at_bound
            next!(prog)
            note_progress()
        end
        finish!(prog)

        D2_theo = (amp_prefactor / 16.0) .* K_norm .* deltas .^ 4
        ratio = D2_num ./ D2_theo
        # the slope and ratio fits use only points strictly above the
        # bootstrapped optimizer floor (see optimizer_floor/above_floor_mask)
        floor_level = optimizer_floor(D2_num, ratio, cfg.floor_detection_ratio)
        clean = above_floor_mask(D2_num, floor_level)
        slope, slope_err =
            count(clean) >= MIN_FIT_POINTS ?
            loglog_slope(deltas[clean], D2_num[clean]) : (NaN, NaN)
        logline(
            log_io,
            @sprintf(
                "        fit points (above optimizer floor) %d/%d; fitted log-log slope %.4f ± %.4f",
                count(clean), n, slope, slope_err)
        )
        c1, c1_err, c2 = ratio_correction_fit(deltas[clean], ratio[clean])
        delta_valid =
            (isfinite(c1) && abs(c1) > 1e-12) ?
            cfg.correction_validity_fraction / abs(c1) : Inf
        isfinite(c1) &&
            logline(
                log_io,
                @sprintf(
                    "        O(δ⁵) fit: ratio ≈ 1 + c₁δ + c₂δ² with c₁ = %.4g ± %.2g, c₂ = %.4g (%.0f%%-validity δ ≈ %.3g)",
                    c1, c1_err, c2, 100cfg.correction_validity_fraction, delta_valid)
            )
        if cfg.n_starts > 1 && maximum(multi_start_gain) > cfg.secondary_minimum_gain
            @warn "Sweep '$name': multi-start found a lower minimum than the canonical " *
                  "start for $(count(>(cfg.secondary_minimum_gain), multi_start_gain))/$n separations (max gain " *
                  "$(round(maximum(multi_start_gain), digits = 2))) — evidence of secondary minima."
        end
        any(at_bound) &&
            @warn "Sweep '$name': the best fit sits on a physical bound for " *
                  "$(count(at_bound))/$n separations (see AtBound column)."
        all(converged) || @warn "Sweep '$name': optimizer did not converge for " *
              "$(count(!, converged))/$n separations (see Converged column)."

        if cfg.monitoring_enabled && progress_enabled()
            println(
                stdout,
                sweep_diagnostic_panel(deltas, D2_num, D2_theo, clean,
                    slope, slope_err),
            )
        end

        logline(log_io, "  [3/3] persisting results and figures")
        df_res = DataFrame(Delta = collect(deltas), D2_Numerical = D2_num,
            D2_Theoretical = D2_theo, K_u_Norm = fill(K_norm, n),
            BestFit_Amplitude = best_fits[:, 1], BestFit_ChirpMass = best_fits[:, 2],
            BestFit_CoalescenceTime = best_fits[:, 3],
            BestFit_CoalescencePhase = best_fits[:, 4],
            BestFit_Spin1 = best_fits[:, 5], BestFit_Spin2 = best_fits[:, 6],
            Converged = collect(converged), Iterations = iteration_counts,
            GradNorm = gradient_norms, AtBound = collect(at_bound),
            Starts = fill(cfg.n_starts, n), MultiStartGain = multi_start_gain)
        CSV.write(backup_existing!(joinpath(out_dir, "results.csv")), df_res)

        # Residual-spectrum evaluation points. Primary δ*: the largest clean
        # separation still inside the fitted validity window of the
        # leading-order law (δ ≤ delta_valid), so the plotted residual is the
        # normal projection the quartic law integrates. Rule-of-thumb
        # companion: the separation nearest the discernibility threshold on
        # the swept range (the old rule). The companion is emitted only when
        # the two points differ — for on-law directions the threshold point
        # is itself inside the validity window and one panel suffices.
        pos = findall(>(0), D2_num)
        idx_thr = isempty(pos) ? n : pos[argmin(abs.(log10.(D2_num[pos] ./ rho_sq)))]
        in_validity = isfinite(delta_valid) ? (deltas .<= delta_valid) : trues(n)
        valid_pos = findall(clean .& in_validity .& (D2_num .> 0))
        idx_star = isempty(valid_pos) ? idx_thr : last(valid_pos)
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
        open(joinpath(out_dir, "sweep_meta.toml"), "w") do io
            TOML.print(
                io,
                Dict(
                    "name" => name, "delta_star" => d_star,
                    "d2_num_star" => D2_num[idx_star], "d2_theo_star" => D2_theo[idx_star],
                    "K_u_norm" => K_norm, "g_uu_raw" => g_uu, "rho_sq" => rho_sq,
                    "df" => ctx.df, "f_min" => cfg.f_min, "f_max" => cfg.f_max,
                    "delta_min" => delta_min, "amp_ratio" => amp_ratio,
                    "slope" => slope, "slope_err" => slope_err,
                    "c1" => c1, "c1_err" => c1_err, "c2" => c2,
                    "delta_valid" => delta_valid,
                    "floor_level" => isnan(floor_level) ? -1.0 : floor_level,
                    "residual_d2_integral_A" => meta.int_A,
                    "residual_d2_integral_E" => meta.int_E,
                    thr_meta...),
            )
        end

        try
            fig = scaling_figure(collect(deltas), D2_num, D2_theo;
                rho_sq = rho_sq, delta_min = delta_min, slope = slope,
                slope_err = slope_err, clean = collect(clean),
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
            @warn "Sweep '$name': figure generation failed; numerical results are saved." exception =
                (err, catch_backtrace())
        end

        elapsed = format_time(time() - t0)
        logline(log_io, "  [done] sweep '$name' completed in $elapsed")
        # structured run.log record (detached campaigns tail run.log, which
        # otherwise carries no per-item timing)
        @info "Sweep '$name' ($idx/$total) completed in $elapsed."
    finally
        close(log_io)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Module 2: 2D confusion mapping (mirrored polar sampling, prior capping)
# -----------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Execute one 2D confusion map: tangent basis at the base point, mirrored
angular sweep with adaptive refinement, exact prior-wall and box-corner
vertices, prior capping, and persistence of the contour table and zone
figure into the run directory.
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
    log_io = open(joinpath(out_dir, "mapping.log"), "w")
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

        n_base_directions = n_angles ÷ 2
        # milestones count against the base direction budget; refinement and
        # bisection add a small unknown surplus that the hook simply absorbs
        note_progress = stage_progress_hook(
            "Map '$name' ($idx/$total): angular sweep", n_base_directions,
            cfg.progress_log_fraction)

        function eval_angles(phis::Vector{Float64}, prog)
            out = Vector{NTuple{2,Float64}}(undef, length(phis))
            parallel_foreach(length(phis), ctx.map_tasks) do i
                dir = zeros(N_PARAMS)
                dir[px] = cos(phis[i])
                dir[py] = sin(phis[i])
                K, g = compute_extrinsic_curvature_from_basis(theta0, dir, basis,
                    ctx.freqs, ctx.Sn, ctx.df, wp)
                out[i] = (K, g)
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
        prog = ProgressUnknown(desc = "  mapping: ", enabled = progress_enabled())
        phis = [(k - 1) * π / n_base_directions for k in 1:n_base_directions]
        curvature_norm_pairs = eval_angles(phis, prog)
        entries = [
            (phi = phis[i], K = curvature_norm_pairs[i][1], g = curvature_norm_pairs[i][2]) for i in 1:n_base_directions
        ]

        r_math_of(K) = K > K_UNDERFLOW ? (16.0 * rho_sq / K)^(1 / 4) : Inf
        r_box_of(phi) = ray_box_crossing(cos(phi), sin(phi), box...)
        r_cap_of(e) = min(r_math_of(e.K), r_box_of(e.phi))

        added = 0
        for _ in 1:cfg.max_refine_levels
            sort!(entries, by = e -> e.phi)
            rcaps = [r_cap_of(e) for e in entries]
            mids = Float64[]
            for i in 1:length(entries)
                j = mod1(i + 1, length(entries))
                gap = (j == 1 ? π + entries[1].phi : entries[j].phi) - entries[i].phi
                r1, r2 = rcaps[i], rcaps[j]
                (isfinite(r1) && isfinite(r2)) || continue
                ratio = max(r1, r2) / max(min(r1, r2), K_UNDERFLOW)
                ratio > cfg.neighbor_ratio_tol && push!(mids, entries[i].phi + gap / 2)
            end
            isempty(mids) && break
            curvature_new = eval_angles(mids, prog)
            append!(
                entries,
                [
                    (phi = mids[i], K = curvature_new[i][1], g = curvature_new[i][2])
                    for i in 1:length(mids)
                ],
            )
            added += length(mids)
        end
        sort!(entries, by = e -> e.phi)
        logline(
            log_io,
            "        refinement added $added directions " *
            "($(length(entries)) on the half-circle)",
        )

        # Exact corner vertices at capping crossovers. The boundary polygon
        # chords over the direction where the mathematical contour pierces a
        # prior wall (r_math = r_box), chamfering the zone's corners — and the
        # neighbor-ratio refinement cannot see it, because the capped radius
        # saturates at r_box on the wall side. Scan the full circle (the box
        # need not be mirror-symmetric), bracket every capped/uncapped
        # transition between consecutive directions, and bisect each bracket
        # on the capping predicate (robust to r_math = Inf on degenerate
        # directions, where a sign-based bisection would hit Inf - Inf). All
        # active brackets are evaluated as one batch per iteration.
        if cfg.corner_bisect_iters > 0
            capped_at = (alpha, K) -> begin
                rb = r_box_of(alpha) # cos/sin of the full-circle angle
                isfinite(rb) && r_math_of(K) >= rb
            end
            n_entries = length(entries)
            alphas = vcat([e.phi for e in entries], [e.phi + π for e in entries])
            K_full_circle = vcat([e.K for e in entries], [e.K for e in entries]) # K is even
            caps = [capped_at(alphas[k], K_full_circle[k]) for k in 1:(2n_entries)]
            lo = Float64[]
            hi = Float64[]
            lo_capped = Bool[]
            for k in 1:(2n_entries)
                j = mod1(k + 1, 2n_entries)
                caps[k] == caps[j] && continue
                push!(lo, alphas[k])
                push!(hi, alphas[j] + (j == 1 ? 2π : 0.0))
                push!(lo_capped, caps[k])
            end
            if !isempty(lo)
                for _ in 1:cfg.corner_bisect_iters
                    mids = (lo .+ hi) ./ 2
                    curvature_mid = eval_angles(mod.(mids, π), prog) # K is even
                    for b in eachindex(mids)
                        if capped_at(mids[b], curvature_mid[b][1]) == lo_capped[b]
                            lo[b] = mids[b]
                        else
                            hi[b] = mids[b]
                        end
                    end
                end
                # fold onto the half-circle (the mirror adds the antipode);
                # dedupe — a mirror-symmetric box yields the same φ twice
                phis_new = Float64[]
                for b in eachindex(lo)
                    phi_vertex = mod((lo[b] + hi[b]) / 2, Float64(π))
                    any(p -> abs(p - phi_vertex) < ANGLE_DEDUPE_TOL, phis_new) && continue
                    any(e -> abs(e.phi - phi_vertex) < ANGLE_DEDUPE_TOL, entries) &&
                        continue
                    push!(phis_new, phi_vertex)
                end
                if !isempty(phis_new)
                    curvature_new = eval_angles(phis_new, prog)
                    append!(
                        entries,
                        [
                            (
                                phi = phis_new[i],
                                K = curvature_new[i][1],
                                g = curvature_new[i][2],
                            )
                            for i in eachindex(phis_new)
                        ],
                    )
                    sort!(entries, by = e -> e.phi)
                end
                logline(
                    log_io,
                    @sprintf(
                        "        corner bisection: %d crossover(s) located, %d exact corner vertex(es) inserted (%d iterations)",
                        length(lo), length(phis_new), cfg.corner_bisect_iters)
                )
            end

            # Box-corner vertices: where two walls are simultaneously active,
            # consecutive samples sit on different walls and their chord cuts
            # the box corner. The corner direction is known analytically; if
            # the ray through a finite box corner is capped there, the true
            # boundary passes through that exact corner — insert it (one K
            # evaluation per finite corner, at most four).
            corner_alphas = [
                atan(cy, cx) for cx in (box[1], box[2]), cy in (box[3], box[4])
                if isfinite(cx) && isfinite(cy)
            ]
            if !isempty(corner_alphas)
                curvature_corner = eval_angles(mod.(corner_alphas, π), prog)
                corner_new = Tuple{Float64,Float64,Float64}[]
                for k in eachindex(corner_alphas)
                    capped_at(corner_alphas[k], curvature_corner[k][1]) || continue
                    phi_vertex = mod(corner_alphas[k], Float64(π))
                    any(e -> abs(e.phi - phi_vertex) < ANGLE_DEDUPE_TOL, entries) &&
                        continue
                    any(t -> abs(t[1] - phi_vertex) < ANGLE_DEDUPE_TOL, corner_new) &&
                        continue
                    push!(
                        corner_new,
                        (phi_vertex, curvature_corner[k][1], curvature_corner[k][2]),
                    )
                end
                if !isempty(corner_new)
                    append!(entries, [(phi = t[1], K = t[2], g = t[3]) for t in corner_new])
                    sort!(entries, by = e -> e.phi)
                    logline(
                        log_io,
                        @sprintf(
                            "        box-corner vertices: %d inserted",
                            length(corner_new)
                        )
                    )
                end
            end
        end
        finish!(prog)

        # Mirror to the full circle: K and g are exactly even in the direction,
        # so r_math is mirrored bitwise; r_box is re-evaluated with the exactly
        # negated direction components (the prior box need not be symmetric).
        half = length(entries)
        angle = Vector{Float64}(undef, 2half)
        K_raw = similar(angle)
        g_uu_values = similar(angle)
        r_math = similar(angle)
        r_box_values = similar(angle)
        r_cap = similar(angle)
        dir_cos = similar(angle)
        dir_sin = similar(angle)
        prior_lim = falses(2half)
        degen = falses(2half)
        for (i, e) in enumerate(entries), half_idx in (0, 1)
            k = i + half_idx * half
            c, s = cos(e.phi), sin(e.phi)
            half_idx == 1 && ((c, s) = (-c, -s))
            angle[k] = e.phi + half_idx * π
            dir_cos[k] = c
            dir_sin[k] = s
            K_raw[k] = e.K
            g_uu_values[k] = e.g
            r_math[k] = r_math_of(e.K)
            r_box_values[k] = ray_box_crossing(c, s, box...)
            r_cap[k] = min(r_math[k], r_box_values[k])
            prior_lim[k] = isfinite(r_box_values[k]) && r_math[k] >= r_box_values[k]
            degen[k] = e.g < cfg.g_uu_degenerate
        end

        if any(!isfinite, r_cap)
            r_cap_max = maximum(filter(isfinite, r_cap); init = 1.0)
            polygon_cap = cfg.unbounded_cap_factor * r_cap_max
            @warn "Map '$name': $(count(!isfinite, r_cap)) directions are unbounded " *
                  "(no curvature limit and no finite physical bound); capping them at " *
                  "$polygon_cap for the polygon. Consider adding [parameter_bounds]."
            r_cap[.!isfinite.(r_cap)] .= polygon_cap
        end

        X = r_cap .* dir_cos
        Y = r_cap .* dir_sin
        prior_frac = count(prior_lim) / length(prior_lim)
        degen_frac = count(degen) / length(degen)
        logline(
            log_io,
            @sprintf("        prior-limited directions : %.1f%%", 100prior_frac)
        )
        degen_frac > 0 &&
            logline(
                log_io,
                @sprintf("        degenerate directions    : %.1f%%", 100degen_frac)
            )
        if px in (5, 6) && py in (5, 6) && prior_frac > SPIN_PRIOR_NOTE_FRACTION
            logline(
                log_io,
                "  [note] the waveform depends on the spins only through " *
                "χ_eff = (χ₁+χ₂)/2; along the anti-symmetric combination the " *
                "manifold is flat, so the zone there is limited by the physical " *
                "spin prior [-1, 1], not by curvature.",
            )
        end

        if cfg.monitoring_enabled && progress_enabled()
            println(stdout, map_diagnostic_panel(angle, r_cap, prior_frac))
        end

        df_map = DataFrame(Angle = angle, X_Bound = X, Y_Bound = Y,
            Dir_Cos = dir_cos, Dir_Sin = dir_sin,
            R_Capped = r_cap, R_Math = r_math, R_Box = r_box_values,
            Prior_Limited = collect(prior_lim), Degenerate = collect(degen),
            K_Raw = K_raw, G_uu = g_uu_values)
        CSV.write(backup_existing!(joinpath(out_dir, "confusion_contour.csv")), df_map)

        try
            fig = zone_figure(X, Y, collect(prior_lim);
                px = px, py = py, box = box,
                prior_frac = prior_frac, degenerate_frac = degen_frac,
                x_math = r_math .* dir_cos, y_math = r_math .* dir_sin)
            save_figure(fig, joinpath(out_dir, "confusion_zone"))
        catch err
            @warn "Map '$name': figure generation failed; numerical results are saved." exception =
                (err, catch_backtrace())
        end

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

# -----------------------------------------------------------------------------
# Pipeline entry point
# -----------------------------------------------------------------------------

"""
$(TYPEDSIGNATURES)

Unified orchestrator: validates the TOML configuration (hard errors on
unusable input), allocates hardware within the `[safety]` memory budget,
snapshots the configuration and provenance metadata into a
configuration-hashed run directory, then executes the 1D sweep and 2D
mapping modules. Each sweep/map is guarded individually — a failing stage is
logged with its backtrace and the remaining stages continue.
"""
function run_pipeline(config_path::String, project_root::String, output_dir::String)
    cfg = load_and_validate_config(config_path)
    run_id = run_id_from_config(config_path)
    out_base = unique_run_dir(joinpath(project_root, output_dir), run_id)
    snapshot_config(config_path, out_base)

    log_stream = open(joinpath(out_base, "run.log"), "w")
    file_logger = FormatLogger(log_stream) do io, args
        println(io, "[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ",
            uppercase(string(args.level)), " ", args.message)
    end
    tee = TeeLogger(global_logger(), MinLevelLogger(file_logger, Logging.Info))
    try
        with_logger(tee) do
            _run_pipeline(cfg, project_root, out_base)
        end
    finally
        close(log_stream)
    end
    return out_base
end

function _run_pipeline(cfg::PipelineSettings, project_root::String, out_base::String)
    start_time = time()
    df = 1.0 / cfg.T_obs
    freqs = collect(cfg.f_min:df:cfg.f_max)
    Sn = analytic_noise_psd.(freqs; noise = cfg.noise)
    n_ch = n_channels(cfg.wp)

    backend = get_best_backend(prefer = cfg.gpu_backend)
    cfg.gpu_backend === :none &&
        @info "[hardware].gpu_backend = \"none\": GPU detection bypassed, running on the CPU backend."
    if !(backend isa KernelAbstractions.CPU) && n_channels(cfg.wp) == 3
        error(
            "GPU backends support the 2-channel (A, E) configuration only; " *
            "set [physics].include_t_channel = false (T is identically zero).",
        )
    end

    sweep_tasks, map_tasks, est_gb = plan_resources(cfg, length(freqs), n_ch, backend)

    freqs_dev = backend isa KernelAbstractions.CPU ? freqs : to_backend(freqs, backend)
    Sn_dev = backend isa KernelAbstractions.CPU ? Sn : to_backend(Sn, backend)
    ctx = RunContext(cfg, out_base, freqs, Sn, freqs_dev, Sn_dev, df, backend,
        sweep_tasks, map_tasks)

    write_run_metadata(out_base;
        run_id = basename(out_base), git = git_state(project_root),
        julia_version = string(VERSION), hostname = gethostname(),
        started = string(now()), backend = backend_name(backend),
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
    write_hardware_fingerprint(out_base;
        device_report = Backends.device_fingerprint(backend))

    println("=" ^ 78)
    println("  CurvatureDistinguishability Pipeline")
    println("=" ^ 78)
    @info "Run directory: $out_base"
    @info "Grid: $(length(freqs)) bins ($(cfg.f_min) – $(cfg.f_max) Hz, df = $df); channels: $n_ch"
    @info "Backend: $(backend_name(backend)); concurrency: $sweep_tasks sweep / " *
          "$map_tasks map tasks; estimated peak RAM $(round(est_gb, digits = 2)) GB " *
          "(budget $(cfg.max_ram_gb) GB)"
    @info "Noise: instrumental Robson Eq.12 + confusion $(cfg.noise.confusion_enabled ? "Eq.14" : "disabled")"
    @info "Optimizer: $(cfg.optimizer) with physical bounds"

    failures = String[]
    if cfg.run_1d_sweeps && !isempty(cfg.sweeps)
        @info ">>> 1D parameter sweeps ($(length(cfg.sweeps)) configurations)"
        for (i, s) in enumerate(cfg.sweeps)
            try
                run_sweep(s, i, length(cfg.sweeps), ctx)
            catch err
                push!(failures, s.name)
                @error "Sweep '$(s.name)' failed; continuing with remaining stages." exception =
                    (err, catch_backtrace())
            end
            maintain_memory!(ctx.backend, cfg.gc_between_stages)
        end
    end
    if cfg.run_2d_mapping && !isempty(cfg.maps)
        @info ">>> 2D confusion mapping ($(length(cfg.maps)) configurations)"
        ctx.map_tasks != ctx.sweep_tasks &&
            @info "2D mapping evaluates host-side automatic differentiation " *
                  "(no GPU kernels): concurrency restored to $(ctx.map_tasks) tasks."
        for (i, m) in enumerate(cfg.maps)
            try
                run_map(m, i, length(cfg.maps), ctx)
            catch err
                push!(failures, m.name)
                @error "Map '$(m.name)' failed; continuing with remaining stages." exception =
                    (err, catch_backtrace())
            end
            maintain_memory!(ctx.backend, cfg.gc_between_stages)
        end
    end

    elapsed = time() - start_time
    write_run_metadata(out_base; finished = string(now()),
        elapsed_seconds = round(elapsed, digits = 1),
        failed_stages = join(failures, ","))
    println("=" ^ 78)
    if isempty(failures)
        @info "Pipeline complete in $(format_time(elapsed)); results in $out_base"
    else
        @warn "Pipeline finished in $(format_time(elapsed)) with failed stages: " *
              "$(join(failures, ", ")) — see run.log for backtraces. Results in $out_base"
    end
    println("=" ^ 78)
    return nothing
end

end # module
