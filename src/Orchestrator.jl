"""
Pipeline driver: resource planning against the [safety] budget, the 1D
sweep and 2D confusion-mapping stages, structured logging, opt-in
monitoring and per-stage guardrails.
"""
module Orchestrator

using Printf: @sprintf
using Dates: Dates, now
using TOML: TOML
using CSV: CSV
using DataFrames: DataFrame
using Random: Xoshiro
using Statistics: mean
using ProgressMeter: Progress, ProgressUnknown, finish!, next!
using Logging: Logging, global_logger, with_logger
using LoggingExtras: FormatLogger, MinLevelLogger, TeeLogger
using KernelAbstractions: KernelAbstractions
using UnicodePlots: UnicodePlots

using ..Backends
using ..Physics
using ..Detector
using ..Geometry
using ..Inference
using ..Bounds
using ..Config
using ..Provenance
using ..Plotting

export run_pipeline

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

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

physics_kwargs(wp::WaveformParams) = (
    mass_scale = wp.mass_scale, time_scale = wp.time_scale, amp_scale = wp.amp_scale,
    eta = wp.eta, amp_33_factor = wp.amp_33_factor, sky_theta = wp.sky_theta,
    sky_phi = wp.sky_phi, inclination = wp.inclination, polarization = wp.polarization,
    include_t_channel = wp.include_t_channel)

"""
    loglog_slope(x, y) -> (slope, stderr)

Least-squares slope of `log10(y)` against `log10(x)` with its standard
error (NaN with fewer than 3 points). Fits the quartic-law exponent of a
sweep's clean window; shared with the display-time refit in `RunFigures`.
"""
function loglog_slope(x::AbstractVector, y::AbstractVector)
    lx, ly = log10.(x), log10.(y)
    mx, my = mean(lx), mean(ly)
    sxx = sum(abs2, lx .- mx)
    slope = sum((lx .- mx) .* (ly .- my)) / sxx
    n = length(lx)
    se = n > 2 ? sqrt(sum(abs2, ly .- my .- slope .* (lx .- mx)) / ((n - 2) * sxx)) : NaN
    return slope, se
end

"""
    ratio_correction_fit(deltas, ratio) -> (c1, c1_err, c2)

Least-squares fit of `ratio − 1 ≈ c₁δ + c₂δ²` (2×2 normal equations solved in
closed form), quantifying the leading `O(δ⁵)` correction to the quartic law
relative to `D²_th`. Returns NaNs with fewer than 3 points.
"""
function ratio_correction_fit(deltas::AbstractVector, ratio::AbstractVector)
    n = length(deltas)
    n >= 3 || return NaN, NaN, NaN
    y = ratio .- 1.0
    s2 = sum(d^2 for d in deltas)
    s3 = sum(d^3 for d in deltas)
    s4 = sum(d^4 for d in deltas)
    b1 = sum(deltas .* y)
    b2 = sum(deltas .^ 2 .* y)
    det = s2 * s4 - s3^2
    abs(det) < 1e-300 && return NaN, NaN, NaN
    c1 = (s4 * b1 - s3 * b2) / det
    c2 = (s2 * b2 - s3 * b1) / det
    resid = y .- c1 .* deltas .- c2 .* deltas .^ 2
    c1_err = n > 2 ? sqrt(max(0.0, sum(abs2, resid) / (n - 2)) * s4 / det) : NaN
    return c1, c1_err, c2
end

"""
    optimizer_floor(D2_num, ratio, cfg.floor_detection_ratio) -> floor_level

Bootstrap estimate of a sweep's optimizer floor: points whose
`D²_num/D²_theo` ratio is at or above `ratio_threshold`
(`[pipeline.sweep_settings].floor_detection_ratio`) are floor-dominated,
and the floor level is the largest floor-dominated `D²_num`. Returns `NaN`
when no point is floor-dominated.
"""
function optimizer_floor(
    D2_num::AbstractVector, ratio::AbstractVector, ratio_threshold::Real)
    floor_pts = findall(>=(ratio_threshold), ratio)
    return isempty(floor_pts) ? NaN : maximum(D2_num[floor_pts])
end

"""
    above_floor_mask(D2_num, floor_level) -> BitVector

Production clean-point rule, shared by the sweep stage and all display-time
figure regeneration (`RunFigures`): only points strictly above the optimizer
floor are clean (all positive points when no floor was detected). Borderline
points inside the floor band are excluded even when their ratio is close to
unity, and convergence flags never exclude a point — the iteration and
tolerance caps are strict enough that flagged points at ratio ≈ 1 are
genuine optima.
"""
above_floor_mask(D2_num::AbstractVector, floor_level::Real) =
    isnan(floor_level) ? (D2_num .> 0) : (D2_num .> floor_level)

"""
    sweep_diagnostic_panel(deltas, D2_num, D2_theo, clean, slope, slope_err) -> String

In-terminal diagnostic of a completed sweep: log-log `D²` against the
theoretical prediction (UnicodePlots), followed by the clean-point count and
fitted slope. Opt-in via `[monitoring].enabled`; printed to stdout on TTY
sessions only, never into the file logs.
"""
function sweep_diagnostic_panel(deltas::AbstractVector, D2_num::AbstractVector,
    D2_theo::AbstractVector, clean::AbstractVector{Bool},
    slope::Real, slope_err::Real)
    pos = D2_num .> 0
    any(pos) || return "sweep diagnostic: no positive D² values"
    plt = UnicodePlots.lineplot(log10.(collect(deltas)), log10.(collect(D2_theo));
        name = "theory", xlabel = "log₁₀ δ",
        ylabel = "log₁₀ D²", width = 64, height = 14)
    UnicodePlots.scatterplot!(plt, log10.(collect(deltas[pos])), log10.(D2_num[pos]);
        name = "numerical")
    footer = @sprintf("clean points %d/%d; fitted slope %.4f ± %.4f",
        count(clean), length(clean), slope, slope_err)
    return sprint(io -> show(io, plt)) * "\n" * footer
end

"""
    map_diagnostic_panel(angle, r_cap, prior_frac) -> String

In-terminal diagnostic of a completed confusion map: capped boundary radius
against direction angle (UnicodePlots), followed by the prior-limited
fraction. Opt-in via `[monitoring].enabled`; stdout on TTY sessions only.
"""
function map_diagnostic_panel(angle::AbstractVector, r_cap::AbstractVector,
    prior_frac::Real)
    plt = UnicodePlots.lineplot(collect(angle), collect(r_cap);
        xlabel = "φ [rad]", ylabel = "r_cap",
        width = 64, height = 14)
    footer = @sprintf("prior-limited directions: %.1f%%", 100 * prior_frac)
    return sprint(io -> show(io, plt)) * "\n" * footer
end

"""
    plan_resources(cfg, n_bins, nch, backend) -> (active_threads, est_gb)

Pre-flight memory estimate against the `[safety]` budget: refuses to start
when even a single task exceeds it, downscales concurrency otherwise, and
applies the VRAM budget on GPU backends.
"""
function plan_resources(cfg::PipelineSettings, n_bins::Int, nch::Int, backend)
    flatlen = 2 * nch * n_bins
    fixed = flatlen * 6 * 8 +      # tangent-basis Jacobian
            flatlen * 8 * 7 * 3 +  # nested-dual evaluation buffers
            6 * nch * n_bins * 16 + # orthonormal basis storage
            4 * n_bins * 8          # grid + PSD
    per_task = (2 + 3 * nch) * n_bins * 16 # per-δ data streams and temporaries
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

    if !(backend isa KernelAbstractions.CPU)
        # GPU execution is single-task by design: GPU drivers are not
        # reliably safe under concurrent multi-task access (observed Level
        # Zero segmentation fault), the library-level GPU lock serializes
        # kernel launches, and the device itself serializes kernels —
        # additional tasks provide no throughput.
        threads > 1 &&
            @info "GPU backend active: concurrency pinned to 1 task (kernels serialize " *
                  "on the device; concurrent driver access is unsafe)."
        threads = 1
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

    return threads, (fixed + threads * per_task) / 2^30
end

"""
Immutable per-run context threaded through the sweep and map stages:
validated configuration, output base directory, frequency grid and PSD
(host and device copies), and the planned concurrency. Parametric over the
backend and device-array types so every per-δ solve closure captures
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
    active_threads::Int
end

# -----------------------------------------------------------------------------
# Module 1: 1D parameter sweeps (δ⁴ validation)
# -----------------------------------------------------------------------------

"""
    run_sweep(sweep, idx, total, ctx)

Execute one 1D separation sweep: directional geometry (K(u), g(u,u)),
per-δ box-constrained optimization with optional multi-start, floor
detection and slope/correction fits, and persistence of the results table,
residual spectrum, metadata and figures into the run directory.
"""
function run_sweep(sweep::AbstractDict, idx::Int, total::Int, ctx::RunContext)
    cfg = ctx.cfg
    wp = cfg.wp
    phys = physics_kwargs(wp)
    name = String(sweep["name"])
    theta0 = Float64.(sweep["theta_0"])
    u_raw = Float64.(sweep["u_dir"])
    rho_sq = Float64(get(sweep, "rho_thresh", cfg.sweep_rho_thresh))^2
    q = Float64(get(sweep, "amp_ratio", 1.0))          # A₂/A₁
    amp_prefactor = (2q / (1 + q))^2                   # (A_harm/A)², = 1 at q = 1

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
        q != 1.0 && logline(
            log_io,
            @sprintf("  amplitude ratio : q = %.4g (A_harm prefactor %.5f)",
                q, amp_prefactor)
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
        s = 1.0 / sqrt(g_uu)
        u_norm = u_raw .* s
        K_norm = K_u * s^4
        delta_min = (16.0 * rho_sq / K_norm)^(1 / 4)
        logline(log_io, @sprintf("        normalized K(u)          : %.6e", K_norm))
        logline(log_io, @sprintf("        delta_min                : %.6e", delta_min))

        deltas = 10 .^ range(cfg.min_log_delta, cfg.max_log_delta, length = cfg.n_deltas)

        # verify the second source stays within the physical bounds at δ_max
        theta_far = theta0 .+ maximum(deltas) .* u_norm
        theta_far[1] = q * theta0[1]
        for i in 1:6
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
        best_fits = zeros(n, 6)
        conv = falses(n)
        iters = zeros(Int, n)
        gnorm = zeros(n)
        atbound = falses(n)
        ms_gain = ones(n)

        logline(
            log_io,
            "  [2/3] optimization sweep ($(n) separations, " *
            "$(ctx.active_threads) concurrent, optimizer = $(cfg.optimizer)" *
            (cfg.n_starts > 1 ? ", $(cfg.n_starts) starts" : "") * ")",
        )
        prog = Progress(n; desc = "  optimizing: ", enabled = progress_enabled())
        parallel_foreach(n, ctx.active_threads) do i
            d = deltas[i]
            p2 = theta0 .+ d .* u_norm
            p2[1] = q * theta0[1]
            h1 = scaled_waveform_model(theta0, ctx.freqs, wp)
            h2 = scaled_waveform_model(p2, ctx.freqs, wp)
            ch1 = project_to_tdi(h1, ctx.freqs, theta0, wp)
            ch2 = project_to_tdi(h2, ctx.freqs, p2, wp)
            data = map((a, b) -> a .+ b, ch1, ch2)
            if !(ctx.backend isa KernelAbstractions.CPU)
                data = map(a -> to_backend(a, ctx.backend), data)
            end
            # degenerate effective source: amplitude (1+q)A at the weighted midpoint
            guess = theta0 .+ (q / (1 + q) * d) .* u_norm
            guess[1] = (1 + q) * theta0[1]

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
                    phys...)
            dist, best, res = solve(guess)
            dist_canonical = dist
            for k in 2:cfg.n_starts
                # deterministic per-task stream: independent of thread scheduling
                rng = Xoshiro(hash((cfg.rng_seed, name, i, k)))
                pert =
                    guess .+ (cfg.multi_start_parallel_scale * d * randn(rng)) .* u_norm .+
                    cfg.multi_start_transverse_scale .* randn(rng, 6)
                dist_k, best_k, res_k = solve(clamp_interior(pert, cfg.bounds))
                if dist_k < dist
                    dist, best, res = dist_k, best_k, res_k
                end
            end
            diagnostics = optimization_diagnostics(res, best, cfg.bounds)
            D2_num[i] = dist
            ms_gain[i] = dist > 0 ? dist_canonical / dist : 1.0
            best_fits[i, :] .= best
            conv[i] = diagnostics.converged
            iters[i] = diagnostics.iterations
            gnorm[i] = diagnostics.g_norm
            atbound[i] = diagnostics.at_bound
            next!(prog)
        end
        finish!(prog)

        D2_theo = (amp_prefactor / 16.0) .* K_norm .* deltas .^ 4
        ratio = D2_num ./ D2_theo
        # the slope and ratio fits use only points strictly above the
        # bootstrapped optimizer floor (see optimizer_floor/above_floor_mask)
        floor_level = optimizer_floor(D2_num, ratio, cfg.floor_detection_ratio)
        clean = above_floor_mask(D2_num, floor_level)
        slope, slope_err =
            count(clean) >= 3 ? loglog_slope(deltas[clean], D2_num[clean]) : (NaN, NaN)
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
                    "        O(δ⁵) fit: ratio ≈ 1 + c₁δ + c₂δ² with c₁ = %.4g ± %.2g, c₂ = %.4g (10%%-validity δ ≈ %.3g)",
                    c1, c1_err, c2, delta_valid)
            )
        if cfg.n_starts > 1 && maximum(ms_gain) > cfg.secondary_minimum_gain
            @warn "Sweep '$name': multi-start found a lower minimum than the canonical " *
                  "start for $(count(>(cfg.secondary_minimum_gain), ms_gain))/$n separations (max gain " *
                  "$(round(maximum(ms_gain), digits = 2))) — evidence of secondary minima."
        end
        any(atbound) &&
            @warn "Sweep '$name': the best fit sits on a physical bound for " *
                  "$(count(atbound))/$n separations (see AtBound column)."
        all(conv) || @warn "Sweep '$name': optimizer did not converge for " *
              "$(count(!, conv))/$n separations (see Converged column)."

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
            BestFit_Time = best_fits[:, 3], BestFit_Phase = best_fits[:, 4],
            BestFit_Spin1 = best_fits[:, 5], BestFit_Spin2 = best_fits[:, 6],
            Converged = collect(conv), Iterations = iters,
            GradNorm = gnorm, AtBound = collect(atbound),
            Starts = fill(cfg.n_starts, n), MultiStartGain = ms_gain)
        CSV.write(backup_existing!(joinpath(out_dir, "results.csv")), df_res)

        # residual spectrum at the separation nearest the discernibility threshold
        pos = findall(>(0), D2_num)
        idx_star = isempty(pos) ? n : pos[argmin(abs.(log10.(D2_num[pos] ./ rho_sq)))]
        d_star = deltas[idx_star]
        spec, meta = residual_spectrum(theta0, u_norm, q, d_star, best_fits[idx_star, :],
            ctx, wp)
        CSV.write(backup_existing!(joinpath(out_dir, "residual_spectrum.csv")), spec)
        open(joinpath(out_dir, "sweep_meta.toml"), "w") do io
            TOML.print(
                io,
                Dict(
                    "name" => name, "delta_star" => d_star,
                    "d2_num_star" => D2_num[idx_star], "d2_theo_star" => D2_theo[idx_star],
                    "K_u_norm" => K_norm, "g_uu_raw" => g_uu, "rho_sq" => rho_sq,
                    "df" => ctx.df, "f_min" => cfg.f_min, "f_max" => cfg.f_max,
                    "delta_min" => delta_min, "amp_ratio" => q,
                    "slope" => slope, "slope_err" => slope_err,
                    "c1" => c1, "c1_err" => c1_err, "c2" => c2,
                    "delta_valid" => delta_valid,
                    "floor_level" => isnan(floor_level) ? -1.0 : floor_level,
                    "residual_d2_integral_A" => meta.int_A,
                    "residual_d2_integral_E" => meta.int_E),
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
                (delta_star = d_star, df = ctx.df,
                    f_min = cfg.f_min, f_max = cfg.f_max,
                    d2_num = D2_num[idx_star], d2_theo = D2_theo[idx_star]),
            )
            save_figure(rfig, joinpath(out_dir, "residual_plot"))
        catch err
            @warn "Sweep '$name': figure generation failed; numerical results are saved." exception =
                (err, catch_backtrace())
        end

        logline(log_io, "  [done] sweep '$name' completed in $(format_time(time() - t0))")
    finally
        close(log_io)
    end
    return nothing
end

"""
Decimated residual spectrum (density units, `d(SNR²)/df = 4|x|²/Sn`) of the
two-source data, the best-fit single source and the unabsorbed residual, for
channels A and E.
"""
function residual_spectrum(theta0, u_norm, q, d_star, best_fit,
    ctx::RunContext, wp::WaveformParams)
    p2 = theta0 .+ d_star .* u_norm
    p2[1] = q * theta0[1]
    h1 = scaled_waveform_model(theta0, ctx.freqs, wp)
    h2 = scaled_waveform_model(p2, ctx.freqs, wp)
    ch1 = project_to_tdi(h1, ctx.freqs, theta0, wp)
    ch2 = project_to_tdi(h2, ctx.freqs, p2, wp)
    data = map((a, b) -> a .+ b, ch1, ch2)
    hb = scaled_waveform_model(best_fit, ctx.freqs, wp)
    bf = project_to_tdi(hb, ctx.freqs, best_fit, wp)

    dens(x, i) = 4 * abs2(x) / ctx.Sn[i]
    n = length(ctx.freqs)
    # log-uniform decimation: ~equal plotted points per decade, and the first/
    # last plotted frequencies sit at the band ends. (Linear windows left a
    # half-window gap at the low end of the log axis and compressed the first
    # decade into a handful of points.) Log-sparse low-frequency windows hold
    # single bins and pass them through unaveraged; empty windows are skipped.
    nwin = min(ctx.cfg.residual_spectrum_windows, n)
    edges = 10.0 .^ range(log10(ctx.freqs[1]), log10(ctx.freqs[end]), nwin + 1)
    window_bounds = [searchsortedfirst(ctx.freqs, e) for e in edges]
    window_bounds[end] = n + 1
    windows = [
        window_bounds[i]:(window_bounds[i+1]-1) for
        i in 1:nwin if window_bounds[i+1] > window_bounds[i]
    ]
    agg(v, stat) = [stat(view(v, r)) for r in windows]
    rms(v) = sqrt(mean(abs2, v))

    cols = Dict{Symbol,Vector{Float64}}(:f => agg(ctx.freqs, mean))
    for (tag, cA, cE) in (("sig", data[1], data[2]), ("bf", bf[1], bf[2]),
        ("res", data[1] .- bf[1], data[2] .- bf[2]))
        for (ch, arr) in (("A", cA), ("E", cE))
            d = [dens(arr[i], i) for i in 1:n]
            cols[Symbol("$(tag)_rms_$ch")] = agg(d, rms)
            cols[Symbol("$(tag)_min_$ch")] = agg(d, minimum)
            cols[Symbol("$(tag)_max_$ch")] = agg(d, maximum)
        end
    end
    int_A = sum(dens(data[1][i] - bf[1][i], i) for i in 1:n) * ctx.df
    int_E = sum(dens(data[2][i] - bf[2][i], i) for i in 1:n) * ctx.df
    order = [:f; sort(collect(keys(delete!(copy(cols), :f))))]
    return DataFrame([c => cols[c] for c in order]), (int_A = int_A, int_E = int_E)
end

# -----------------------------------------------------------------------------
# Module 2: 2D confusion mapping (mirrored polar sampling, prior capping)
# -----------------------------------------------------------------------------

"""
    run_map(map_cfg, idx, total, ctx)

Execute one 2D confusion map: tangent basis at the base point, mirrored
angular sweep with adaptive refinement, exact prior-wall and box-corner
vertices, prior capping, and persistence of the contour table and zone
figure into the run directory.
"""
function run_map(map_cfg::AbstractDict, idx::Int, total::Int, ctx::RunContext)
    cfg = ctx.cfg
    wp = cfg.wp
    name = String(map_cfg["name"])
    px = Int(map_cfg["param_x"])
    py = Int(map_cfg["param_y"])
    # per-map override; defaults to the pipeline-wide threshold
    rho_sq = Float64(get(map_cfg, "rho_thresh", cfg.sweep_rho_thresh))^2
    theta0 = Float64.(map_cfg["theta_0"])
    n_angles = Int(get(map_cfg, "n_angles", cfg.map_n_angles))
    isodd(n_angles) && (n_angles += 1)

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

        function eval_angles(phis::Vector{Float64}, prog)
            out = Vector{NTuple{2,Float64}}(undef, length(phis))
            parallel_foreach(length(phis), ctx.active_threads) do i
                dir = zeros(6)
                dir[px] = cos(phis[i])
                dir[py] = sin(phis[i])
                K, g = compute_extrinsic_curvature_from_basis(theta0, dir, basis,
                    ctx.freqs, ctx.Sn, ctx.df, wp)
                out[i] = (K, g)
                next!(prog)
            end
            return out
        end

        M = n_angles ÷ 2
        logline(
            log_io,
            "  [2/2] mirrored angular sweep ($M base directions on [0, π), " *
            "adaptive refinement tol $(cfg.neighbor_ratio_tol), " *
            "$(cfg.max_refine_levels) levels)",
        )
        prog = ProgressUnknown(desc = "  mapping: ", enabled = progress_enabled())
        phis = [(k - 1) * π / M for k in 1:M]
        curvature_norm_pairs = eval_angles(phis, prog)
        entries = [
            (phi = phis[i], K = curvature_norm_pairs[i][1], g = curvature_norm_pairs[i][2]) for i in 1:M
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
            N = length(entries)
            alphas = vcat([e.phi for e in entries], [e.phi + π for e in entries])
            K_full_circle = vcat([e.K for e in entries], [e.K for e in entries]) # K is even
            caps = [capped_at(alphas[k], K_full_circle[k]) for k in 1:2N]
            lo = Float64[]
            hi = Float64[]
            lo_capped = Bool[]
            for k in 1:2N
                j = mod1(k + 1, 2N)
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
                    any(p -> abs(p - phi_vertex) < 1e-10, phis_new) && continue
                    any(e -> abs(e.phi - phi_vertex) < 1e-10, entries) && continue
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
                    any(e -> abs(e.phi - phi_vertex) < 1e-10, entries) && continue
                    any(t -> abs(t[1] - phi_vertex) < 1e-10, corner_new) && continue
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
        dircos = similar(angle)
        dirsin = similar(angle)
        prior_lim = falses(2half)
        degen = falses(2half)
        for (i, e) in enumerate(entries), half_idx in (0, 1)
            k = i + half_idx * half
            c, s = cos(e.phi), sin(e.phi)
            half_idx == 1 && ((c, s) = (-c, -s))
            angle[k] = e.phi + half_idx * π
            dircos[k] = c
            dirsin[k] = s
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
            @warn "Map '$name': $(count(!isfinite, r_cap)) directions are unbounded " *
                  "(no curvature limit and no finite physical bound); capping them at " *
                  "$(5r_cap_max) for the polygon. Consider adding [parameter_bounds]."
            r_cap[.!isfinite.(r_cap)] .= 5r_cap_max
        end

        X = r_cap .* dircos
        Y = r_cap .* dirsin
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
        if px in (5, 6) && py in (5, 6) && prior_frac > 0.25
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
            Dir_Cos = dircos, Dir_Sin = dirsin,
            R_Capped = r_cap, R_Math = r_math, R_Box = r_box_values,
            Prior_Limited = collect(prior_lim), Degenerate = collect(degen),
            K_Raw = K_raw, G_uu = g_uu_values)
        CSV.write(backup_existing!(joinpath(out_dir, "confusion_contour.csv")), df_map)

        try
            fig = zone_figure(angle, X, Y, collect(prior_lim);
                px = px, py = py, box = box,
                prior_frac = prior_frac, degenerate_frac = degen_frac,
                x_math = r_math .* dircos, y_math = r_math .* dirsin)
            save_figure(fig, joinpath(out_dir, "confusion_zone"))
        catch err
            @warn "Map '$name': figure generation failed; numerical results are saved." exception =
                (err, catch_backtrace())
        end

        logline(log_io, "  [done] map '$name' completed in $(format_time(time() - t0))")
    finally
        close(log_io)
    end
    return nothing
end

# -----------------------------------------------------------------------------
# Pipeline entry point
# -----------------------------------------------------------------------------

"""
    run_pipeline(config_path, project_root, output_dir)

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
            _run_pipeline(cfg, config_path, project_root, out_base)
        end
    finally
        close(log_stream)
    end
    return out_base
end

function _run_pipeline(cfg::PipelineSettings, config_path::String, project_root::String,
    out_base::String)
    start_time = time()
    df = 1.0 / cfg.T_obs
    freqs = collect(cfg.f_min:df:cfg.f_max)
    Sn = analytic_noise_psd.(freqs; noise = cfg.noise)
    nch = n_channels(cfg.wp)

    backend = get_best_backend(prefer = cfg.gpu_backend)
    cfg.gpu_backend === :none &&
        @info "[hardware].gpu_backend = \"none\": GPU detection bypassed, running on the CPU backend."
    if !(backend isa KernelAbstractions.CPU) && cfg.wp.include_t_channel
        error(
            "GPU backends support the 2-channel (A, E) configuration only; " *
            "set [physics].include_t_channel = false (T is identically zero).",
        )
    end

    active_threads, est_gb = plan_resources(cfg, length(freqs), nch, backend)

    freqs_dev = backend isa KernelAbstractions.CPU ? freqs : to_backend(freqs, backend)
    Sn_dev = backend isa KernelAbstractions.CPU ? Sn : to_backend(Sn, backend)
    ctx = RunContext(cfg, out_base, freqs, Sn, freqs_dev, Sn_dev, df, backend,
        active_threads)

    write_run_metadata(out_base;
        run_id = basename(out_base), git = git_state(project_root),
        julia_version = string(VERSION), hostname = gethostname(),
        started = string(now()), backend = backend_name(backend),
        julia_threads = Threads.nthreads(), active_tasks = active_threads,
        n_frequency_bins = length(freqs), channels = nch,
        estimated_ram_gb = round(est_gb, digits = 2),
        optimizer = string(cfg.optimizer),
        n_starts = cfg.n_starts, rng_seed = cfg.rng_seed,
        confusion_noise = cfg.noise.confusion_enabled)

    println("=" ^ 78)
    println("  CurvatureDistinguishability Pipeline")
    println("=" ^ 78)
    @info "Run directory: $out_base"
    @info "Grid: $(length(freqs)) bins ($(cfg.f_min) – $(cfg.f_max) Hz, df = $df); channels: $nch"
    @info "Backend: $(backend_name(backend)); concurrency: $active_threads tasks; " *
          "estimated peak RAM $(round(est_gb, digits = 2)) GB (budget $(cfg.max_ram_gb) GB)"
    @info "Noise: instrumental Robson Eq.12 + confusion $(cfg.noise.confusion_enabled ? "Eq.14" : "disabled")"
    @info "Optimizer: $(cfg.optimizer) with physical bounds"

    failures = String[]
    if cfg.run_1d_sweeps && !isempty(cfg.sweeps)
        @info ">>> 1D parameter sweeps ($(length(cfg.sweeps)) configurations)"
        for (i, s) in enumerate(cfg.sweeps)
            try
                run_sweep(s, i, length(cfg.sweeps), ctx)
            catch err
                push!(failures, String(s["name"]))
                @error "Sweep '$(s["name"])' failed; continuing with remaining stages." exception =
                    (err, catch_backtrace())
            end
        end
    end
    if cfg.run_2d_mapping && !isempty(cfg.maps)
        @info ">>> 2D confusion mapping ($(length(cfg.maps)) configurations)"
        for (i, m) in enumerate(cfg.maps)
            try
                run_map(m, i, length(cfg.maps), ctx)
            catch err
                push!(failures, String(m["name"]))
                @error "Map '$(m["name"])' failed; continuing with remaining stages." exception =
                    (err, catch_backtrace())
            end
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
