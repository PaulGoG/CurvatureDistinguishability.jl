"""
TOML configuration parsing and validation into an immutable
[`PipelineSettings`](@ref).
"""
module Config

using DocStringExtensions: TYPEDSIGNATURES
using ..Physics
using ..Physics: N_PARAMS
using ..Detector: n_channels
using ..Bounds
using ..Provenance: effective_config

export PipelineSettings, SweepSpec, MapSpec, load_and_validate_config

"""
    SweepSpec

Validated work item of the 1D sweep stage, parsed once from a `[[sweeps]]`
entry: filesystem-safe `name`, interior base point `theta_0`, nonzero sweep
direction `u_dir`, resolved discernibility threshold `rho_thresh` (the
per-sweep override or the pipeline-wide default) and amplitude ratio
`amp_ratio` (`A₂/A₁`; 1 for identical amplitudes).
"""
struct SweepSpec
    name::String
    theta_0::Vector{Float64}
    u_dir::Vector{Float64}
    rho_thresh::Float64
    amp_ratio::Float64
end

"""
    MapSpec

Validated work item of the 2D confusion-mapping stage, parsed once from a
`[[maps]]` entry: filesystem-safe `name`, distinct parameter plane
`(param_x, param_y)`, interior base point `theta_0`, resolved
discernibility threshold `rho_thresh` and even angular sample count
`n_angles` (the per-map override or `[mapping].n_angles`).
"""
struct MapSpec
    name::String
    param_x::Int
    param_y::Int
    theta_0::Vector{Float64}
    rho_thresh::Float64
    n_angles::Int
end

"""
Fully parsed and validated pipeline configuration. Construction goes through
[`load_and_validate_config`](@ref), which fails fast with a descriptive
error for unusable input, emits warnings for suspicious-but-runnable
values, and warns on every unknown key (typo protection). Constructed by
keyword only (every field is required): the per-section parse helpers
return NamedTuples keyed by field name, so no positional field order exists
to get wrong.
"""
Base.@kwdef struct PipelineSettings
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
    floor_detection_ratio::Float64
    secondary_minimum_gain::Float64
    multi_start_parallel_scale::Float64
    multi_start_transverse_scale::Float64
    correction_validity_fraction::Float64
    residual_spectrum_windows::Int
    # grid
    T_obs::Float64
    f_min::Float64
    f_max::Float64
    # physics / noise
    # abstract over the channel-count parameter; extracted once per stage
    # behind function barriers, so the hot paths see the concrete type
    wp::WaveformParams{Float64}
    noise::NoiseParams
    # mapping
    map_n_angles::Int
    neighbor_ratio_tol::Float64
    max_refine_levels::Int
    corner_bisect_iters::Int
    unbounded_cap_factor::Float64
    # hardware
    gpu_backend::Symbol
    max_threads::Int
    hessian_chunk::Int
    gc_between_stages::Bool
    # safety
    max_ram_gb::Float64
    bytes_per_bin_per_task_gpu::Int
    max_vram_gb::Float64
    os_vram_overhead_gb::Float64
    # monitoring
    monitoring_enabled::Bool
    progress_log_fraction::Float64
    # bounds & work items
    bounds::ParameterBounds
    sweeps::Vector{SweepSpec}
    maps::Vector{MapSpec}
end

"""
Recognized configuration keys per `[section]` context — the whitelist behind
[`warn_unknown_keys`](@ref)'s typo protection.
"""
const KNOWN_KEYS = Dict(
    "" => ["base_config", "pipeline", "grid", "physics", "noise", "mapping",
        "hardware", "safety", "monitoring", "parameter_bounds", "sweeps", "maps"],
    "monitoring" => ["enabled", "progress_log_fraction"],
    "pipeline" => ["run_1d_sweeps", "run_2d_mapping", "optimizer", "rng_seed",
        "sweep_settings"],
    "pipeline.sweep_settings" => ["n_deltas", "min_log_delta", "max_log_delta",
        "rho_thresh", "g_uu_degenerate", "n_starts",
        "floor_detection_ratio", "secondary_minimum_gain",
        "multi_start_parallel_scale", "multi_start_transverse_scale",
        "correction_validity_fraction", "residual_spectrum_windows",
        "g_tol", "max_iterations"],
    "grid" => ["T_obs", "f_min", "f_max"],
    "physics" => ["mass_scale", "time_scale", "amp_scale", "eta", "amp_33_factor",
        "sky_theta", "sky_phi", "inclination", "polarization",
        "include_t_channel"],
    "noise" => ["confusion_enabled", "confusion_amp", "confusion_alpha",
        "confusion_beta", "confusion_kappa", "confusion_gamma",
        "confusion_knee_freq", "arm_length", "oms_amplitude",
        "oms_reddening_freq", "acc_amplitude", "acc_knee_low", "acc_knee_high"],
    "mapping" =>
        ["n_angles", "neighbor_ratio_tol", "max_refine_levels", "corner_bisect_iters",
            "unbounded_cap_factor"],
    "hardware" => ["gpu_backend", "max_threads", "hessian_chunk",
        "gc_between_stages", "heap_size_hint_gb"],
    "safety" => ["max_ram_gb", "bytes_per_bin_per_task_gpu", "max_vram_gb",
        "os_vram_overhead_gb"],
    "parameter_bounds" => collect(PARAM_KEYS),
    "sweeps[]" => ["name", "theta_0", "u_dir", "rho_thresh", "amp_ratio"],
    "maps[]" => ["name", "param_x", "param_y", "rho_thresh", "theta_0", "n_angles"],
)

"""
Warn (typo protection) on every key of `table` that is not in the
[`KNOWN_KEYS`](@ref) list for `context`.
"""
function warn_unknown_keys(table::AbstractDict, context::String)
    known = get(KNOWN_KEYS, context, String[])
    for key in keys(table)
        key in known || @warn "Unknown configuration key '$key' in " *
              "[$(isempty(context) ? "top level" : context)] — ignored. " *
              "Check for typos; known keys: $(join(known, ", "))."
    end
    return nothing
end

"""
Fetch `key` from table `t` (or `default`) as a `Float64`, erroring with the
offending `[context]` path on a non-numeric value.
"""
get_number(t, key, default, context) = begin
    v = get(t, key, default)
    v isa Real || error("[$context].$key must be a number, got $(typeof(v))")
    Float64(v)
end

"""
Fetch `key` from table `t` (or `default`) as an `Int`, erroring with the
offending `[context]` path on a non-integer value.
"""
get_integer(t, key, default, context) = begin
    v = get(t, key, default)
    (v isa Integer || (v isa Real && isinteger(v))) ||
        error("[$context].$key must be an integer, got $(repr(v))")
    Int(v)
end

"""
Fetch `key` from table `t` (or `default`) as a `Bool`, erroring with the
offending `[context]` path on a non-boolean value.
"""
get_boolean(t, key, default, context) = begin
    v = get(t, key, default)
    v isa Bool || error("[$context].$key must be true or false, got $(repr(v))")
    v
end

"""
Validate a 6-element finite numeric parameter vector, returning it as
`Vector{Float64}`; `what` names the offending config entry in errors.
"""
function validate_theta6(v, what)
    (v isa AbstractVector && length(v) == N_PARAMS) ||
        error("$what must be a $(N_PARAMS)-element numeric vector, got $(repr(v))")
    all(x -> x isa Real && isfinite(x), v) ||
        error("$what must contain only finite numbers, got $(repr(v))")
    return Float64.(v)
end

"""
Whether `name` is safe to use as a directory name (letters, digits,
`._-` only).
"""
fs_safe(name) = !isempty(name) && all(c -> isletter(c) || isdigit(c) || c in "._-", name)

# fraction of total system memory used as the RAM budget when [safety] does
# not set max_ram_gb explicitly
const DEFAULT_RAM_FRACTION = 0.8

"""
$(TYPEDSIGNATURES)

Parse a configuration TOML file — an overlay's `base_config` resolved and
merged first ([`effective_config`](@ref)) — validating every field (types,
ranges, name uniqueness, interior base points) with descriptive errors;
warn on unknown keys and physically suspicious values.
"""
function load_and_validate_config(config_path::AbstractString)
    isfile(config_path) || error("Configuration file not found: $config_path")
    config = try
        effective_config(config_path)
    catch err
        error("Failed to load $config_path: $(sprint(showerror, err))")
    end
    return settings_from_config(config)
end

"""
Validate a parsed configuration table section by section and assemble the
[`PipelineSettings`](@ref). Sections are parsed in file order by dedicated
helpers, each returning a NamedTuple whose keys are `PipelineSettings`
field names; the keyword constructor assembles them.
"""
function settings_from_config(config::AbstractDict)
    warn_unknown_keys(config, "")
    pipeline = parse_pipeline(config)
    sweep_settings = parse_sweep_settings(config)
    grid = parse_grid(config)
    physics = parse_physics(config)
    noise = parse_noise(config, grid.T_obs)
    mapping = parse_mapping(config)
    hardware = parse_hardware(config)
    safety = parse_safety(config)
    bounds = parse_bounds(config)
    work_items = parse_work_items(config, bounds.bounds, sweep_settings.sweep_rho_thresh,
        mapping.map_n_angles, pipeline.run_1d_sweeps, pipeline.run_2d_mapping)
    monitoring = parse_monitoring(config)
    return PipelineSettings(; pipeline..., sweep_settings..., grid..., physics...,
        noise..., mapping..., hardware..., safety..., bounds..., work_items...,
        monitoring...)
end

"""
`[pipeline]` stage switches, optimizer choice and RNG seed.
"""
function parse_pipeline(config::AbstractDict)
    pipeline = get(config, "pipeline", Dict{String,Any}())
    warn_unknown_keys(pipeline, "pipeline")
    run_1d_sweeps = get_boolean(pipeline, "run_1d_sweeps", true, "pipeline")
    run_2d_mapping = get_boolean(pipeline, "run_2d_mapping", true, "pipeline")
    opt_str = get(pipeline, "optimizer", "ipnewton")
    optimizer = Symbol(lowercase(String(opt_str)))
    optimizer in (:ipnewton, :lbfgs_box) ||
        error("[pipeline].optimizer must be one of ipnewton | lbfgs_box, got '$opt_str'")
    rng_seed = get_integer(pipeline, "rng_seed", 42, "pipeline")
    return (; run_1d_sweeps, run_2d_mapping, optimizer, rng_seed)
end

"""
`[pipeline.sweep_settings]`: separation grid, discernibility threshold,
optimizer tolerances, multi-start and analysis tunables.
"""
function parse_sweep_settings(config::AbstractDict)
    pipeline = get(config, "pipeline", Dict{String,Any}())
    sweep_settings = get(pipeline, "sweep_settings", Dict{String,Any}())
    warn_unknown_keys(sweep_settings, "pipeline.sweep_settings")
    n_deltas = get_integer(sweep_settings, "n_deltas", 20, "pipeline.sweep_settings")
    n_deltas >= 2 || error("[pipeline.sweep_settings].n_deltas must be >= 2, got $n_deltas")
    min_log_delta =
        get_number(sweep_settings, "min_log_delta", -4.5, "pipeline.sweep_settings")
    max_log_delta =
        get_number(sweep_settings, "max_log_delta", -0.5, "pipeline.sweep_settings")
    min_log_delta < max_log_delta ||
        error(
            "[pipeline.sweep_settings]: min_log_delta ($min_log_delta) must be < max_log_delta ($max_log_delta)",
        )
    sweep_rho_thresh =
        get_number(sweep_settings, "rho_thresh", 1.0, "pipeline.sweep_settings")
    sweep_rho_thresh > 0 || error("[pipeline.sweep_settings].rho_thresh must be > 0")
    g_uu_degenerate =
        get_number(sweep_settings, "g_uu_degenerate", 1e-6, "pipeline.sweep_settings")
    g_uu_degenerate > 0 || error("[pipeline.sweep_settings].g_uu_degenerate must be > 0")
    n_starts = get_integer(sweep_settings, "n_starts", 1, "pipeline.sweep_settings")
    n_starts >= 1 || error("[pipeline.sweep_settings].n_starts must be >= 1, got $n_starts")
    # Safe-by-default optimizer tolerances: fits at the numerical precision
    # floor cannot reach very tight gradient norms and would otherwise
    # exhaust the iteration cap; clean-region fits converge in 16–40 Newton
    # iterations, so 100 leaves ample margin.
    g_tol = get_number(sweep_settings, "g_tol", 1e-10, "pipeline.sweep_settings")
    g_tol > 0 || error("[pipeline.sweep_settings].g_tol must be > 0, got $g_tol")
    g_tol < 1e-11 &&
        @warn "[pipeline.sweep_settings].g_tol = $g_tol is tighter than the numerical " *
              "precision floor of small-separation fits; expect iteration-cap stalls there."
    max_iterations =
        get_integer(sweep_settings, "max_iterations", 100, "pipeline.sweep_settings")
    max_iterations >= 1 || error("[pipeline.sweep_settings].max_iterations must be >= 1")
    max_iterations > 300 &&
        @warn "[pipeline.sweep_settings].max_iterations = $max_iterations: floor fits " *
              "exhaust the full cap by construction — a large cap costs wall time " *
              "without improving accuracy."
    floor_detection_ratio = get_number(
        sweep_settings, "floor_detection_ratio", 2.0, "pipeline.sweep_settings")
    floor_detection_ratio > 1 ||
        error(
            "[pipeline.sweep_settings].floor_detection_ratio must be > 1, got $floor_detection_ratio",
        )
    secondary_minimum_gain = get_number(
        sweep_settings, "secondary_minimum_gain", 1.5, "pipeline.sweep_settings")
    secondary_minimum_gain > 1 ||
        error(
            "[pipeline.sweep_settings].secondary_minimum_gain must be > 1, got $secondary_minimum_gain",
        )
    multi_start_parallel_scale = get_number(
        sweep_settings, "multi_start_parallel_scale", 0.35, "pipeline.sweep_settings")
    multi_start_parallel_scale > 0 ||
        error("[pipeline.sweep_settings].multi_start_parallel_scale must be > 0")
    multi_start_transverse_scale = get_number(
        sweep_settings, "multi_start_transverse_scale", 1e-3, "pipeline.sweep_settings")
    multi_start_transverse_scale >= 0 ||
        error("[pipeline.sweep_settings].multi_start_transverse_scale must be >= 0")
    correction_validity_fraction = get_number(
        sweep_settings, "correction_validity_fraction", 0.1, "pipeline.sweep_settings")
    correction_validity_fraction > 0 ||
        error("[pipeline.sweep_settings].correction_validity_fraction must be > 0")
    residual_spectrum_windows = get_integer(
        sweep_settings, "residual_spectrum_windows", 600, "pipeline.sweep_settings")
    residual_spectrum_windows >= 8 ||
        error("[pipeline.sweep_settings].residual_spectrum_windows must be >= 8")
    return (; n_deltas, min_log_delta, max_log_delta, sweep_rho_thresh, g_uu_degenerate,
        n_starts, g_tol, max_iterations, floor_detection_ratio, secondary_minimum_gain,
        multi_start_parallel_scale, multi_start_transverse_scale,
        correction_validity_fraction, residual_spectrum_windows)
end

"""
`[grid]`: observation time and frequency band; the implied bin count is
checked for a usable minimum.
"""
function parse_grid(config::AbstractDict)
    grid = get(config, "grid", Dict{String,Any}())
    warn_unknown_keys(grid, "grid")
    T_obs = get_number(grid, "T_obs", SECONDS_PER_YEAR, "grid")
    T_obs > 0 || error("[grid].T_obs must be > 0, got $T_obs")
    f_min = get_number(grid, "f_min", 1.0e-3, "grid")
    f_max = get_number(grid, "f_max", 0.01, "grid")
    (f_min > 0 && f_max > f_min) ||
        error("[grid]: need 0 < f_min < f_max, got f_min = $f_min, f_max = $f_max")
    n_bins = floor(Int, (f_max - f_min) * T_obs) + 1
    n_bins >= 8 || error(
        "[grid]: only $n_bins frequency bins at df = 1/T_obs — " *
        "increase T_obs or the [f_min, f_max] band",
    )
    return (; T_obs, f_min, f_max)
end

"""
`[physics]` into a [`WaveformParams`](@ref); defaults are owned by the
struct and never restated here.
"""
function parse_physics(config::AbstractDict)
    phys = get(config, "physics", Dict{String,Any}())
    warn_unknown_keys(phys, "physics")
    wp_default = WaveformParams()
    wp = WaveformParams(
        mass_scale = get_number(phys, "mass_scale", wp_default.mass_scale, "physics"),
        time_scale = get_number(phys, "time_scale", wp_default.time_scale, "physics"),
        amp_scale = get_number(phys, "amp_scale", wp_default.amp_scale, "physics"),
        eta = get_number(phys, "eta", wp_default.eta, "physics"),
        amp_33_factor = get_number(
            phys, "amp_33_factor", wp_default.amp_33_factor, "physics"),
        sky_theta = get_number(phys, "sky_theta", wp_default.sky_theta, "physics"),
        sky_phi = get_number(phys, "sky_phi", wp_default.sky_phi, "physics"),
        inclination = get_number(phys, "inclination", wp_default.inclination, "physics"),
        polarization = get_number(
            phys, "polarization", wp_default.polarization, "physics"),
        include_t_channel = get_boolean(
            phys, "include_t_channel", n_channels(wp_default) == 3, "physics"),
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
    return (; wp)
end

"""
`[noise]` into a [`NoiseParams`](@ref): every numeric field defaults to the
Robson Table-1 selection for `T_obs` and is overridable individually.
"""
function parse_noise(config::AbstractDict, T_obs::Real)
    noise_cfg = get(config, "noise", Dict{String,Any}())
    warn_unknown_keys(noise_cfg, "noise")
    base_noise = robson_confusion_params(T_obs)
    noise_numeric = (:confusion_amp, :confusion_alpha, :confusion_beta,
        :confusion_kappa, :confusion_gamma, :confusion_knee_freq, :arm_length,
        :oms_amplitude, :oms_reddening_freq, :acc_amplitude, :acc_knee_low,
        :acc_knee_high)
    noise_values = Dict(
        field => get_number(noise_cfg, String(field), getfield(base_noise, field),
            "noise") for field in noise_numeric)
    noise = NoiseParams(;
        confusion_enabled = get_boolean(noise_cfg, "confusion_enabled", true, "noise"),
        noise_values...)
    for field in (:arm_length, :oms_amplitude, :oms_reddening_freq,
        :acc_amplitude, :acc_knee_low, :acc_knee_high)
        getfield(noise, field) > 0 ||
            error("[noise].$field must be > 0, got $(getfield(noise, field))")
    end
    noise.confusion_amp >= 0 || error("[noise].confusion_amp must be >= 0")
    return (; noise)
end

"""
`[mapping]`: angular resolution (rounded up to an even count), refinement
and corner-bisection controls, and the unbounded-direction polygon cap.
"""
function parse_mapping(config::AbstractDict)
    mapping = get(config, "mapping", Dict{String,Any}())
    warn_unknown_keys(mapping, "mapping")
    map_n_angles = get_integer(mapping, "n_angles", 2000, "mapping")
    map_n_angles >= 8 || error("[mapping].n_angles must be >= 8, got $map_n_angles")
    if isodd(map_n_angles)
        @warn "[mapping].n_angles = $map_n_angles is odd; rounding up to " *
              "$(map_n_angles + 1) (mirrored sampling needs an even count)."
        map_n_angles += 1
    end
    neighbor_ratio_tol = get_number(mapping, "neighbor_ratio_tol", 1.25, "mapping")
    neighbor_ratio_tol > 1 ||
        error("[mapping].neighbor_ratio_tol must be > 1, got $neighbor_ratio_tol")
    max_refine_levels = get_integer(mapping, "max_refine_levels", 6, "mapping")
    0 <= max_refine_levels <= 16 ||
        error("[mapping].max_refine_levels must be in 0:16, got $max_refine_levels")
    corner_bisect_iters = get_integer(mapping, "corner_bisect_iters", 25, "mapping")
    0 <= corner_bisect_iters <= 60 ||
        error(
            "[mapping].corner_bisect_iters must be in 0:60 (0 disables corner " *
            "bisection), got $corner_bisect_iters",
        )
    unbounded_cap_factor = get_number(mapping, "unbounded_cap_factor", 5.0, "mapping")
    unbounded_cap_factor > 1 ||
        error("[mapping].unbounded_cap_factor must be > 1, got $unbounded_cap_factor")
    return (; map_n_angles, neighbor_ratio_tol, max_refine_levels, corner_bisect_iters,
        unbounded_cap_factor)
end

"""
`[hardware]`: backend selection, concurrency cap, Hessian chunking and
inter-stage memory maintenance. `heap_size_hint_gb` is validated here but
consumed only by the detached launcher.
"""
function parse_hardware(config::AbstractDict)
    hardware = get(config, "hardware", Dict{String,Any}())
    warn_unknown_keys(hardware, "hardware")
    gpu_str = get(hardware, "gpu_backend", "auto")
    gpu_backend = Symbol(lowercase(String(gpu_str)))
    gpu_backend in (:auto, :none, :cuda, :amdgpu, :metal, :oneapi) ||
        error(
            "[hardware].gpu_backend must be auto | none | cuda | amdgpu | metal | oneapi, got '$gpu_str'",
        )
    max_threads = get_integer(hardware, "max_threads", Threads.nthreads(), "hardware")
    max_threads >= 1 || error("[hardware].max_threads must be >= 1, got $max_threads")
    hessian_chunk = get_integer(hardware, "hessian_chunk", 0, "hardware")
    0 <= hessian_chunk <= 6 ||
        error(
            "[hardware].hessian_chunk must be in 0:6 (0 = full 6-parameter chunk), " *
            "got $hessian_chunk",
        )
    gc_between_stages = get_boolean(hardware, "gc_between_stages", true, "hardware")
    heap_size_hint_gb = get_number(hardware, "heap_size_hint_gb", 0.0, "hardware")
    heap_size_hint_gb >= 0 ||
        error(
            "[hardware].heap_size_hint_gb must be >= 0 (0 = no hint), got $heap_size_hint_gb",
        )
    return (; gpu_backend, max_threads, hessian_chunk, gc_between_stages)
end

"""
`[safety]`: host and device memory budgets.
"""
function parse_safety(config::AbstractDict)
    safety = get(config, "safety", Dict{String,Any}())
    warn_unknown_keys(safety, "safety")
    default_ram = DEFAULT_RAM_FRACTION * Sys.total_memory() / 2^30
    max_ram_gb = get_number(safety, "max_ram_gb", default_ram, "safety")
    max_ram_gb > 0 || error("[safety].max_ram_gb must be > 0, got $max_ram_gb")
    bytes_per_bin_per_task_gpu =
        get_integer(safety, "bytes_per_bin_per_task_gpu", 1000, "safety")
    bytes_per_bin_per_task_gpu > 0 ||
        error(
            "[safety].bytes_per_bin_per_task_gpu must be > 0, got $bytes_per_bin_per_task_gpu",
        )
    max_vram_gb = get_number(safety, "max_vram_gb", 8.0, "safety")
    max_vram_gb > 0 || error("[safety].max_vram_gb must be > 0, got $max_vram_gb")
    os_vram_overhead_gb = get_number(safety, "os_vram_overhead_gb", 1.0, "safety")
    os_vram_overhead_gb >= 0 ||
        error("[safety].os_vram_overhead_gb must be >= 0, got $os_vram_overhead_gb")
    return (; max_ram_gb, bytes_per_bin_per_task_gpu, max_vram_gb, os_vram_overhead_gb)
end

"""
`[parameter_bounds]` into a [`ParameterBounds`](@ref).
"""
function parse_bounds(config::AbstractDict)
    table = get(config, "parameter_bounds", Dict{String,Any}())
    bounds = try
        bounds_from_config(table)
    catch err
        error("Invalid [parameter_bounds]: $(sprint(showerror, err))")
    end
    warn_unknown_keys(table, "parameter_bounds")
    return (; bounds)
end

"""
`[[sweeps]]` and `[[maps]]` into [`SweepSpec`](@ref) / [`MapSpec`](@ref)
vectors: unique filesystem-safe names, interior base points, resolved
per-item thresholds and even angular counts.
"""
function parse_work_items(config::AbstractDict, bounds::ParameterBounds,
    sweep_rho_thresh::Real, map_n_angles::Integer,
    run_1d_sweeps::Bool, run_2d_mapping::Bool)
    sweep_tables = Vector{Dict{String,Any}}(get(config, "sweeps", []))
    map_tables = Vector{Dict{String,Any}}(get(config, "maps", []))
    seen = Set{String}()
    sweeps = map(sweep_tables) do s
        warn_unknown_keys(s, "sweeps[]")
        name = String(get(s, "name", ""))
        fs_safe(name) ||
            error("[[sweeps]] entry has missing or non-filesystem-safe name: $(repr(name))")
        name in seen && error("Duplicate sweep/map name '$name'")
        push!(seen, name)
        theta0 =
            validate_theta6(get(s, "theta_0", nothing), "[[sweeps]] '$name'.theta_0")
        check_interior(theta0, bounds, "sweep '$name'")
        u = validate_theta6(get(s, "u_dir", nothing), "[[sweeps]] '$name'.u_dir")
        norm_u = sqrt(sum(abs2, u))
        norm_u > 0 || error("[[sweeps]] '$name'.u_dir must be nonzero")
        amp_ratio = get_number(s, "amp_ratio", 1.0, "sweeps[]")
        amp_ratio > 0 ||
            error("[[sweeps]] '$name'.amp_ratio must be > 0, got $amp_ratio")
        if amp_ratio != 1.0 && u[1] != 0
            error(
                "[[sweeps]] '$name': amp_ratio ≠ 1 requires u_dir[1] = 0 — amplitude " *
                "separation is expressed via amp_ratio (the A_harm law), not via the " *
                "sweep direction.",
            )
        end
        u[1] == 0 ||
            @warn "Sweep '$name': u_dir has an amplitude component — the quartic law's " *
                  "equal-amplitude absorption argument assumes u_dir[1] = 0."
        rho = get_number(s, "rho_thresh", sweep_rho_thresh, "sweeps[]")
        rho > 0 || error("[[sweeps]] '$name'.rho_thresh must be > 0")
        SweepSpec(name, theta0, u, rho, amp_ratio)
    end
    maps = map(map_tables) do m
        warn_unknown_keys(m, "maps[]")
        name = String(get(m, "name", ""))
        fs_safe(name) ||
            error("[[maps]] entry has missing or non-filesystem-safe name: $(repr(name))")
        name in seen && error("Duplicate sweep/map name '$name'")
        push!(seen, name)
        px = get_integer(m, "param_x", 0, "maps[]")
        py = get_integer(m, "param_y", 0, "maps[]")
        (1 <= px <= N_PARAMS && 1 <= py <= N_PARAMS) ||
            error(
                "[[maps]] '$name': param_x/param_y must be in 1:$(N_PARAMS), " *
                "got ($px, $py)",
            )
        px != py || error("[[maps]] '$name': param_x and param_y must differ")
        rho = get_number(m, "rho_thresh", sweep_rho_thresh, "maps[]")
        rho > 0 || error("[[maps]] '$name'.rho_thresh must be > 0")
        theta0 = validate_theta6(get(m, "theta_0", nothing), "[[maps]] '$name'.theta_0")
        check_interior(theta0, bounds, "map '$name'")
        na = get_integer(m, "n_angles", map_n_angles, "maps[]")
        na >= 8 || error("[[maps]] '$name'.n_angles must be >= 8, got $na")
        if isodd(na)
            @warn "[[maps]] '$name'.n_angles = $na is odd; rounding up to $(na + 1) " *
                  "(mirrored sampling needs an even count)."
            na += 1
        end
        MapSpec(name, px, py, theta0, rho, na)
    end
    (run_1d_sweeps && isempty(sweeps)) &&
        @warn "[pipeline].run_1d_sweeps = true but no [[sweeps]] entries are defined."
    (run_2d_mapping && isempty(maps)) &&
        @warn "[pipeline].run_2d_mapping = true but no [[maps]] entries are defined."
    return (; sweeps, maps)
end

"""
`[monitoring]`: in-terminal diagnostics switch and run-log progress cadence.
"""
function parse_monitoring(config::AbstractDict)
    monitoring_cfg = get(config, "monitoring", Dict{String,Any}())
    warn_unknown_keys(monitoring_cfg, "monitoring")
    monitoring_enabled = get_boolean(monitoring_cfg, "enabled", false, "monitoring")
    progress_log_fraction =
        get_number(monitoring_cfg, "progress_log_fraction", 0.25, "monitoring")
    0.0 <= progress_log_fraction <= 1.0 ||
        error(
            "[monitoring].progress_log_fraction must be in [0, 1] (0 disables " *
            "stage-progress log lines), got $progress_log_fraction",
        )
    return (; monitoring_enabled, progress_log_fraction)
end

"""
Error unless every non-periodic component of `theta0` lies strictly inside
its physical bounds — the local geometry expansion and zone capping
require an interior base point.
"""
function check_interior(theta0::AbstractVector, b::ParameterBounds, what::String)
    for i in eachindex(theta0)
        b.periodic[i] && continue
        (b.lower[i] < theta0[i] < b.upper[i]) ||
            error(
                "$what: theta_0[$i] = $(theta0[i]) is not strictly inside its physical " *
                "bounds [$(b.lower[i]), $(b.upper[i])] — the local geometry expansion " *
                "and zone capping require an interior base point.",
            )
    end
    return nothing
end

end # module
