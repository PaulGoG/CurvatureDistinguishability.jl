module Config

using TOML
using ..Physics
using ..Bounds

export PipelineSettings, load_and_validate_config

"""
Fully parsed and validated pipeline configuration. Construction goes through
[`load_and_validate_config`](@ref), which fails fast with a descriptive
error for unusable input and emits warnings for suspicious-but-runnable
values — silent misconfiguration
is designed out by warning on every unknown key.
"""
struct PipelineSettings
    # pipeline
    run_1d_sweeps::Bool
    run_2d_mapping::Bool
    optimizer::Symbol
    rng_seed::Int
    # sweep settings
    n_deltas::Int
    min_log_delta::Float64
    max_log_delta::Float64
    sweep_rho_thresh::Float64
    g_uu_degenerate::Float64
    n_starts::Int
    g_tol::Float64
    max_iterations::Int
    # grid
    T_obs::Float64
    f_min::Float64
    f_max::Float64
    # physics / noise
    wp::WaveformParams{Float64}
    noise::NoiseParams
    # mapping
    map_n_angles::Int
    neighbor_ratio_tol::Float64
    max_refine_levels::Int
    corner_bisect_iters::Int
    # hardware
    gpu_backend::Symbol
    max_threads::Int
    hessian_chunk::Int
    # safety
    max_ram_gb::Float64
    bytes_per_bin_per_task_gpu::Int
    max_vram_gb::Float64
    os_vram_overhead_gb::Float64
    # bounds & work items
    bounds::ParameterBounds
    sweeps::Vector{Dict{String,Any}}
    maps::Vector{Dict{String,Any}}
end

const KNOWN_KEYS = Dict(
    "" => ["pipeline", "grid", "physics", "noise", "mapping", "hardware",
           "safety", "parameter_bounds", "sweeps", "maps"],
    "pipeline" => ["run_1d_sweeps", "run_2d_mapping", "optimizer", "rng_seed",
                   "sweep_settings"],
    "pipeline.sweep_settings" => ["n_deltas", "min_log_delta", "max_log_delta",
                                  "rho_thresh", "g_uu_degenerate", "n_starts",
                                  "g_tol", "max_iterations"],
    "grid" => ["T_obs", "f_min", "f_max"],
    "physics" => ["mass_scale", "time_scale", "amp_scale", "eta", "amp_33_factor",
                  "sky_theta", "sky_phi", "inclination", "polarization",
                  "include_t_channel"],
    "noise" => ["confusion_enabled", "confusion_amp", "confusion_alpha",
                "confusion_beta", "confusion_kappa", "confusion_gamma", "confusion_fk"],
    "mapping" => ["n_angles", "neighbor_ratio_tol", "max_refine_levels", "corner_bisect_iters"],
    "hardware" => ["gpu_backend", "max_threads", "hessian_chunk"],
    "safety" => ["max_ram_gb", "bytes_per_bin_per_task_gpu", "max_vram_gb",
                 "os_vram_overhead_gb"],
    "parameter_bounds" => collect(PARAM_KEYS),
    "sweeps[]" => ["name", "theta_0", "u_dir", "rho_thresh", "amp_ratio"],
    "maps[]" => ["name", "param_x", "param_y", "rho_thresh", "theta_0", "n_angles"],
)

function warn_unknown_keys(table::AbstractDict, context::String)
    known = get(KNOWN_KEYS, context, String[])
    for key in keys(table)
        key in known || @warn "Unknown configuration key '$key' in " *
                             "[$(isempty(context) ? "top level" : context)] — ignored. " *
                             "Check for typos; known keys: $(join(known, ", "))."
    end
    return nothing
end

getnum(t, key, default, context) = begin
    v = get(t, key, default)
    v isa Real || error("[$context].$key must be a number, got $(typeof(v))")
    Float64(v)
end

getint(t, key, default, context) = begin
    v = get(t, key, default)
    (v isa Integer || (v isa Real && isinteger(v))) ||
        error("[$context].$key must be an integer, got $(repr(v))")
    Int(v)
end

getbool(t, key, default, context) = begin
    v = get(t, key, default)
    v isa Bool || error("[$context].$key must be true or false, got $(repr(v))")
    v
end

function validate_theta6(v, what)
    (v isa AbstractVector && length(v) == 6) ||
        error("$what must be a 6-element numeric vector, got $(repr(v))")
    all(x -> x isa Real && isfinite(x), v) ||
        error("$what must contain only finite numbers, got $(repr(v))")
    return Float64.(v)
end

fs_safe(name) = !isempty(name) && all(c -> isletter(c) || isdigit(c) || c in "._-", name)

"""
    load_and_validate_config(config_path) -> PipelineSettings

Parse `config.toml`, validating every field (types, ranges, name uniqueness,
interior base points) with descriptive errors; warn on unknown keys and
physically suspicious values.
"""
function load_and_validate_config(config_path::AbstractString)
    isfile(config_path) || error("Configuration file not found: $config_path")
    config = try
        TOML.parsefile(config_path)
    catch err
        error("Failed to parse $config_path as TOML: $(sprint(showerror, err))")
    end
    warn_unknown_keys(config, "")

    pipeline = get(config, "pipeline", Dict{String,Any}())
    warn_unknown_keys(pipeline, "pipeline")
    run_sweeps = getbool(pipeline, "run_1d_sweeps", true, "pipeline")
    run_maps = getbool(pipeline, "run_2d_mapping", true, "pipeline")
    opt_str = get(pipeline, "optimizer", "ipnewton")
    optimizer = Symbol(lowercase(String(opt_str)))
    optimizer in (:ipnewton, :lbfgs_box) ||
        error("[pipeline].optimizer must be one of ipnewton | lbfgs_box, got '$opt_str'")
    rng_seed = getint(pipeline, "rng_seed", 42, "pipeline")

    ss = get(pipeline, "sweep_settings", Dict{String,Any}())
    warn_unknown_keys(ss, "pipeline.sweep_settings")
    n_deltas = getint(ss, "n_deltas", 20, "pipeline.sweep_settings")
    n_deltas >= 2 || error("[pipeline.sweep_settings].n_deltas must be >= 2, got $n_deltas")
    min_log = getnum(ss, "min_log_delta", -4.5, "pipeline.sweep_settings")
    max_log = getnum(ss, "max_log_delta", -0.5, "pipeline.sweep_settings")
    min_log < max_log ||
        error("[pipeline.sweep_settings]: min_log_delta ($min_log) must be < max_log_delta ($max_log)")
    sweep_rho = getnum(ss, "rho_thresh", 1.0, "pipeline.sweep_settings")
    sweep_rho > 0 || error("[pipeline.sweep_settings].rho_thresh must be > 0")
    g_deg = getnum(ss, "g_uu_degenerate", 1e-6, "pipeline.sweep_settings")
    g_deg > 0 || error("[pipeline.sweep_settings].g_uu_degenerate must be > 0")
    n_starts = getint(ss, "n_starts", 1, "pipeline.sweep_settings")
    n_starts >= 1 || error("[pipeline.sweep_settings].n_starts must be >= 1, got $n_starts")
    # Safe-by-default optimizer tolerances: fits at the
    # numerical precision floor cannot reach very tight gradient norms and
    # would otherwise grind against the iteration cap; clean-region fits
    # converge in 16–40 Newton iterations, so 100 is generous.
    g_tol = getnum(ss, "g_tol", 1e-10, "pipeline.sweep_settings")
    g_tol > 0 || error("[pipeline.sweep_settings].g_tol must be > 0, got $g_tol")
    g_tol < 1e-11 &&
        @warn "[pipeline.sweep_settings].g_tol = $g_tol is tighter than the numerical " *
              "precision floor of small-separation fits; expect iteration-cap stalls there."
    max_iterations = getint(ss, "max_iterations", 100, "pipeline.sweep_settings")
    max_iterations >= 1 || error("[pipeline.sweep_settings].max_iterations must be >= 1")
    max_iterations > 300 &&
        @warn "[pipeline.sweep_settings].max_iterations = $max_iterations: floor fits burn " *
              "the full cap by construction — large caps cost wall time, not accuracy."

    grid = get(config, "grid", Dict{String,Any}())
    warn_unknown_keys(grid, "grid")
    T_obs = getnum(grid, "T_obs", SECONDS_PER_YEAR, "grid")
    T_obs > 0 || error("[grid].T_obs must be > 0, got $T_obs")
    f_min = getnum(grid, "f_min", 1.0e-3, "grid")
    f_max = getnum(grid, "f_max", 0.01, "grid")
    (f_min > 0 && f_max > f_min) ||
        error("[grid]: need 0 < f_min < f_max, got f_min = $f_min, f_max = $f_max")
    n_bins = floor(Int, (f_max - f_min) * T_obs) + 1
    n_bins >= 8 || error("[grid]: only $n_bins frequency bins at df = 1/T_obs — " *
                         "increase T_obs or the [f_min, f_max] band")

    phys = get(config, "physics", Dict{String,Any}())
    warn_unknown_keys(phys, "physics")
    wp = WaveformParams(
        mass_scale = getnum(phys, "mass_scale", 10.0, "physics"),
        time_scale = getnum(phys, "time_scale", 1000.0, "physics"),
        amp_scale = getnum(phys, "amp_scale", 1e-21, "physics"),
        eta = getnum(phys, "eta", 0.25, "physics"),
        amp_33_factor = getnum(phys, "amp_33_factor", 0.1, "physics"),
        sky_theta = getnum(phys, "sky_theta", 1.047, "physics"),
        sky_phi = getnum(phys, "sky_phi", 0.0, "physics"),
        inclination = getnum(phys, "inclination", 0.523, "physics"),
        polarization = getnum(phys, "polarization", 0.0, "physics"),
        include_t_channel = getbool(phys, "include_t_channel", false, "physics"),
    )
    for (fname, val, lo, hi) in (("mass_scale", wp.mass_scale, 0.0, Inf),
                                 ("time_scale", wp.time_scale, 0.0, Inf),
                                 ("amp_scale", wp.amp_scale, 0.0, Inf),
                                 ("amp_33_factor", wp.amp_33_factor, 0.0, Inf))
        val > lo || error("[physics].$fname must be > $lo, got $val")
    end
    0.0 < wp.eta <= 0.25 ||
        error("[physics].eta must be in (0, 0.25] (symmetric mass ratio), got $(wp.eta)")
    0.0 <= wp.sky_theta <= π ||
        @warn "[physics].sky_theta = $(wp.sky_theta) is outside [0, π]; interpreting as-is."

    noise_cfg = get(config, "noise", Dict{String,Any}())
    warn_unknown_keys(noise_cfg, "noise")
    base_noise = robson_confusion_params(T_obs)
    noise = NoiseParams(
        confusion_enabled = getbool(noise_cfg, "confusion_enabled", true, "noise"),
        confusion_amp = getnum(noise_cfg, "confusion_amp", base_noise.confusion_amp, "noise"),
        confusion_alpha = getnum(noise_cfg, "confusion_alpha", base_noise.confusion_alpha, "noise"),
        confusion_beta = getnum(noise_cfg, "confusion_beta", base_noise.confusion_beta, "noise"),
        confusion_kappa = getnum(noise_cfg, "confusion_kappa", base_noise.confusion_kappa, "noise"),
        confusion_gamma = getnum(noise_cfg, "confusion_gamma", base_noise.confusion_gamma, "noise"),
        confusion_fk = getnum(noise_cfg, "confusion_fk", base_noise.confusion_fk, "noise"),
    )
    noise.confusion_amp >= 0 || error("[noise].confusion_amp must be >= 0")

    mapping = get(config, "mapping", Dict{String,Any}())
    warn_unknown_keys(mapping, "mapping")
    map_n_angles = getint(mapping, "n_angles", 2000, "mapping")
    map_n_angles >= 8 || error("[mapping].n_angles must be >= 8, got $map_n_angles")
    if isodd(map_n_angles)
        @warn "[mapping].n_angles = $map_n_angles is odd; rounding up to " *
              "$(map_n_angles + 1) (mirrored sampling needs an even count)."
        map_n_angles += 1
    end
    ratio_tol = getnum(mapping, "neighbor_ratio_tol", 1.25, "mapping")
    ratio_tol > 1 || error("[mapping].neighbor_ratio_tol must be > 1, got $ratio_tol")
    refine_levels = getint(mapping, "max_refine_levels", 6, "mapping")
    0 <= refine_levels <= 16 ||
        error("[mapping].max_refine_levels must be in 0:16, got $refine_levels")
    corner_iters = getint(mapping, "corner_bisect_iters", 25, "mapping")
    0 <= corner_iters <= 60 ||
        error("[mapping].corner_bisect_iters must be in 0:60 (0 disables corner " *
              "bisection), got $corner_iters")

    hardware = get(config, "hardware", Dict{String,Any}())
    warn_unknown_keys(hardware, "hardware")
    gpu_str = get(hardware, "gpu_backend", "auto")
    gpu_backend = Symbol(lowercase(String(gpu_str)))
    gpu_backend in (:auto, :none, :cuda, :amdgpu, :metal, :oneapi) ||
        error("[hardware].gpu_backend must be auto | none | cuda | amdgpu | metal | oneapi, got '$gpu_str'")
    max_threads = getint(hardware, "max_threads", Threads.nthreads(), "hardware")
    max_threads >= 1 || error("[hardware].max_threads must be >= 1, got $max_threads")
    hessian_chunk = getint(hardware, "hessian_chunk", 0, "hardware")
    0 <= hessian_chunk <= 6 ||
        error("[hardware].hessian_chunk must be in 0:6 (0 = full 6-parameter chunk), " *
              "got $hessian_chunk")

    safety = get(config, "safety", Dict{String,Any}())
    warn_unknown_keys(safety, "safety")
    default_ram = 0.8 * Sys.total_memory() / 2^30
    max_ram_gb = getnum(safety, "max_ram_gb", default_ram, "safety")
    max_ram_gb > 0 || error("[safety].max_ram_gb must be > 0, got $max_ram_gb")
    gpu_bytes = getint(safety, "bytes_per_bin_per_task_gpu", 1000, "safety")
    max_vram_gb = getnum(safety, "max_vram_gb", 8.0, "safety")
    os_vram_gb = getnum(safety, "os_vram_overhead_gb", 1.0, "safety")

    bounds = try
        bounds_from_config(get(config, "parameter_bounds", Dict{String,Any}()))
    catch err
        error("Invalid [parameter_bounds]: $(sprint(showerror, err))")
    end
    warn_unknown_keys(get(config, "parameter_bounds", Dict{String,Any}()), "parameter_bounds")

    sweeps = Vector{Dict{String,Any}}(get(config, "sweeps", []))
    maps = Vector{Dict{String,Any}}(get(config, "maps", []))
    seen = Set{String}()
    for s in sweeps
        warn_unknown_keys(s, "sweeps[]")
        name = String(get(s, "name", ""))
        fs_safe(name) || error("[[sweeps]] entry has missing or non-filesystem-safe name: $(repr(name))")
        name in seen && error("Duplicate sweep/map name '$name'")
        push!(seen, name)
        theta0 = validate_theta6(get(s, "theta_0", nothing), "[[sweeps]] '$name'.theta_0")
        check_interior(theta0, bounds, "sweep '$name'")
        u = validate_theta6(get(s, "u_dir", nothing), "[[sweeps]] '$name'.u_dir")
        norm_u = sqrt(sum(abs2, u))
        norm_u > 0 || error("[[sweeps]] '$name'.u_dir must be nonzero")
        amp_ratio = getnum(s, "amp_ratio", 1.0, "sweeps[]")
        amp_ratio > 0 || error("[[sweeps]] '$name'.amp_ratio must be > 0, got $amp_ratio")
        if amp_ratio != 1.0 && u[1] != 0
            error("[[sweeps]] '$name': amp_ratio ≠ 1 requires u_dir[1] = 0 — amplitude " *
                  "separation is expressed via amp_ratio (the A_harm law), not via the " *
                  "sweep direction.")
        end
        u[1] == 0 ||
            @warn "Sweep '$name': u_dir has an amplitude component — the quartic law's " *
                  "equal-amplitude absorption argument assumes u_dir[1] = 0."
        haskey(s, "rho_thresh") &&
            (getnum(s, "rho_thresh", sweep_rho, "sweeps[]") > 0 ||
             error("[[sweeps]] '$name'.rho_thresh must be > 0"))
    end
    for m in maps
        warn_unknown_keys(m, "maps[]")
        name = String(get(m, "name", ""))
        fs_safe(name) || error("[[maps]] entry has missing or non-filesystem-safe name: $(repr(name))")
        name in seen && error("Duplicate sweep/map name '$name'")
        push!(seen, name)
        px = getint(m, "param_x", 0, "maps[]")
        py = getint(m, "param_y", 0, "maps[]")
        (1 <= px <= 6 && 1 <= py <= 6) ||
            error("[[maps]] '$name': param_x/param_y must be in 1:6, got ($px, $py)")
        px != py || error("[[maps]] '$name': param_x and param_y must differ")
        getnum(m, "rho_thresh", sweep_rho, "maps[]") > 0 ||
            error("[[maps]] '$name'.rho_thresh must be > 0")
        theta0 = validate_theta6(get(m, "theta_0", nothing), "[[maps]] '$name'.theta_0")
        check_interior(theta0, bounds, "map '$name'")
        if haskey(m, "n_angles")
            na = getint(m, "n_angles", map_n_angles, "maps[]")
            na >= 8 || error("[[maps]] '$name'.n_angles must be >= 8, got $na")
        end
    end
    (run_sweeps && isempty(sweeps)) &&
        @warn "[pipeline].run_1d_sweeps = true but no [[sweeps]] entries are defined."
    (run_maps && isempty(maps)) &&
        @warn "[pipeline].run_2d_mapping = true but no [[maps]] entries are defined."

    return PipelineSettings(run_sweeps, run_maps, optimizer, rng_seed,
                            n_deltas, min_log, max_log, sweep_rho, g_deg, n_starts,
                            g_tol, max_iterations,
                            T_obs, f_min, f_max, wp, noise,
                            map_n_angles, ratio_tol, refine_levels, corner_iters,
                            gpu_backend, max_threads, hessian_chunk,
                            max_ram_gb, gpu_bytes, max_vram_gb, os_vram_gb,
                            bounds, sweeps, maps)
end

function check_interior(theta0::AbstractVector, b::ParameterBounds, what::String)
    for i in eachindex(theta0)
        b.periodic[i] && continue
        (b.lower[i] < theta0[i] < b.upper[i]) ||
            error("$what: theta_0[$i] = $(theta0[i]) is not strictly inside its physical " *
                  "bounds [$(b.lower[i]), $(b.upper[i])] — the local geometry expansion " *
                  "and zone capping require an interior base point.")
    end
    return nothing
end

end # module
