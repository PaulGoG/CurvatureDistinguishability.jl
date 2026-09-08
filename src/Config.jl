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
using ..Bounds: PARAM_KEYS
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
    min_log_delta_ratio::Float64
    max_log_delta_ratio::Float64
    sweep_rho_thresh::Float64
    g_uu_degenerate::Float64
    n_starts::Int
    g_tol::Float64
    f_reltol::Float64
    max_iterations::Int
    floor_detection_ratio::Float64
    secondary_minimum_gain::Float64
    multi_start_parallel_scale::Float64
    multi_start_transverse_scale::Float64
    correction_validity_fraction::Float64
    correction_fit_max_departure::Float64
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
    "pipeline.sweep_settings" =>
        ["n_deltas", "min_log_delta_ratio", "max_log_delta_ratio",
            "rho_thresh", "g_uu_degenerate", "n_starts",
            "floor_detection_ratio", "secondary_minimum_gain",
            "multi_start_parallel_scale", "multi_start_transverse_scale",
            "correction_validity_fraction", "correction_fit_max_departure",
            "residual_spectrum_windows",
            "g_tol", "f_reltol", "max_iterations"],
    "grid" => ["T_obs", "f_min", "f_max"],
    "physics" => ["mass_scale", "time_scale", "distance_scale", "eta",
        "ecliptic_longitude", "ecliptic_latitude", "inclination", "polarization",
        "orbit_phase", "constellation_phase", "cutoff_width", "include_t_channel"],
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
    v isa Real || config_error("[$context].$key must be a number, got $(typeof(v))")
    Float64(v)
end

"""
Fetch `key` from table `t` (or `default`) as an `Int`, erroring with the
offending `[context]` path on a non-integer value.
"""
get_integer(t, key, default, context) = begin
    v = get(t, key, default)
    (v isa Integer || (v isa Real && isinteger(v))) ||
        config_error("[$context].$key must be an integer, got $(repr(v))")
    Int(v)
end

"""
Fetch `key` from table `t` (or `default`) as a `Bool`, erroring with the
offending `[context]` path on a non-boolean value.
"""
get_boolean(t, key, default, context) = begin
    v = get(t, key, default)
    v isa Bool || config_error("[$context].$key must be true or false, got $(repr(v))")
    v
end

"""
Validate a 6-element finite numeric parameter vector, returning it as
`Vector{Float64}`; `what` names the offending config entry in errors.
"""
function validate_theta6(v, what)
    (v isa AbstractVector && length(v) == N_PARAMS) ||
        config_error("$what must be a $(N_PARAMS)-element numeric vector, got $(repr(v))")
    all(x -> x isa Real && isfinite(x), v) ||
        config_error("$what must contain only finite numbers, got $(repr(v))")
    return Float64.(v)
end

"""
Whether `name` is safe to use as a directory name: letters, digits and
`._-` only, with at least one letter or digit (so `.` and `..`, which would
escape the case directory, are rejected).
"""
fs_safe(name) =
    !isempty(name) && any(c -> isletter(c) || isdigit(c), name) &&
    all(c -> isletter(c) || isdigit(c) || c in "._-", name)

"""
Raise an `ArgumentError` for an unusable configuration value; every
validation failure of this module funnels through it.
"""
config_error(msg::AbstractString) = throw(ArgumentError(msg))

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
    isfile(config_path) || config_error("Configuration file not found: $config_path")
    config = try
        effective_config(config_path)
    catch err
        config_error("Failed to load $config_path: $(sprint(showerror, err))")
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
    noise = parse_noise(config, grid.T_obs)
    physics = parse_physics(config, noise.noise.arm_length)
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
        config_error(
            "[pipeline].optimizer must be one of ipnewton | lbfgs_box, got '$opt_str'",
        )
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
    n_deltas >= 2 ||
        config_error("[pipeline.sweep_settings].n_deltas must be >= 2, got $n_deltas")
    min_log_delta_ratio =
        get_number(sweep_settings, "min_log_delta_ratio", -2.0, "pipeline.sweep_settings")
    max_log_delta_ratio =
        get_number(sweep_settings, "max_log_delta_ratio", 0.3, "pipeline.sweep_settings")
    min_log_delta_ratio < max_log_delta_ratio ||
        config_error(
            "[pipeline.sweep_settings]: min_log_delta_ratio ($min_log_delta_ratio) must be < max_log_delta_ratio ($max_log_delta_ratio)",
        )
    sweep_rho_thresh =
        get_number(sweep_settings, "rho_thresh", 1.0, "pipeline.sweep_settings")
    sweep_rho_thresh > 0 || config_error("[pipeline.sweep_settings].rho_thresh must be > 0")
    g_uu_degenerate =
        get_number(sweep_settings, "g_uu_degenerate", 1e-6, "pipeline.sweep_settings")
    g_uu_degenerate > 0 ||
        config_error("[pipeline.sweep_settings].g_uu_degenerate must be > 0")
    n_starts = get_integer(sweep_settings, "n_starts", 1, "pipeline.sweep_settings")
    n_starts >= 1 ||
        config_error("[pipeline.sweep_settings].n_starts must be >= 1, got $n_starts")
    # Safe-by-default optimizer tolerances: fits at the numerical precision
    # floor cannot reach very tight gradient norms and would otherwise
    # exhaust the iteration cap; clean-region fits converge in 16–40 Newton
    # iterations, so 100 leaves ample margin.
    g_tol = get_number(sweep_settings, "g_tol", 1e-10, "pipeline.sweep_settings")
    g_tol > 0 || config_error("[pipeline.sweep_settings].g_tol must be > 0, got $g_tol")
    g_tol < 1e-11 &&
        @warn "[pipeline.sweep_settings].g_tol = $g_tol is tighter than the numerical " *
              "precision floor of small-separation fits; expect iteration-cap stalls there."
    f_reltol = get_number(sweep_settings, "f_reltol", 1e-10, "pipeline.sweep_settings")
    f_reltol > 0 ||
        config_error("[pipeline.sweep_settings].f_reltol must be > 0, got $f_reltol")
    max_iterations =
        get_integer(sweep_settings, "max_iterations", 100, "pipeline.sweep_settings")
    max_iterations >= 1 ||
        config_error("[pipeline.sweep_settings].max_iterations must be >= 1")
    max_iterations > 300 &&
        @warn "[pipeline.sweep_settings].max_iterations = $max_iterations: floor fits " *
              "exhaust the full cap by construction — a large cap costs wall time " *
              "without improving accuracy."
    floor_detection_ratio = get_number(
        sweep_settings, "floor_detection_ratio", 2.0, "pipeline.sweep_settings")
    floor_detection_ratio > 1 ||
        config_error(
            "[pipeline.sweep_settings].floor_detection_ratio must be > 1, got $floor_detection_ratio",
        )
    secondary_minimum_gain = get_number(
        sweep_settings, "secondary_minimum_gain", 1.5, "pipeline.sweep_settings")
    secondary_minimum_gain > 1 ||
        config_error(
            "[pipeline.sweep_settings].secondary_minimum_gain must be > 1, got $secondary_minimum_gain",
        )
    multi_start_parallel_scale = get_number(
        sweep_settings, "multi_start_parallel_scale", 0.35, "pipeline.sweep_settings")
    multi_start_parallel_scale > 0 ||
        config_error("[pipeline.sweep_settings].multi_start_parallel_scale must be > 0")
    multi_start_transverse_scale = get_number(
        sweep_settings, "multi_start_transverse_scale", 1e-3, "pipeline.sweep_settings")
    multi_start_transverse_scale >= 0 ||
        config_error("[pipeline.sweep_settings].multi_start_transverse_scale must be >= 0")
    correction_validity_fraction = get_number(
        sweep_settings, "correction_validity_fraction", 0.1, "pipeline.sweep_settings")
    correction_validity_fraction > 0 ||
        config_error("[pipeline.sweep_settings].correction_validity_fraction must be > 0")
    correction_fit_max_departure = get_number(
        sweep_settings, "correction_fit_max_departure", 0.3, "pipeline.sweep_settings")
    correction_fit_max_departure > 0 ||
        config_error("[pipeline.sweep_settings].correction_fit_max_departure must be > 0")
    residual_spectrum_windows = get_integer(
        sweep_settings, "residual_spectrum_windows", 600, "pipeline.sweep_settings")
    residual_spectrum_windows >= 8 ||
        config_error("[pipeline.sweep_settings].residual_spectrum_windows must be >= 8")
    return (; n_deltas, min_log_delta_ratio, max_log_delta_ratio, sweep_rho_thresh,
        g_uu_degenerate,
        n_starts, g_tol, f_reltol, max_iterations, floor_detection_ratio,
        secondary_minimum_gain,
        multi_start_parallel_scale, multi_start_transverse_scale,
        correction_validity_fraction, correction_fit_max_departure,
        residual_spectrum_windows)
end

"""
`[grid]`: observation time and frequency band; the implied bin count is
checked for a usable minimum.
"""
function parse_grid(config::AbstractDict)
    grid = get(config, "grid", Dict{String,Any}())
    warn_unknown_keys(grid, "grid")
    T_obs = get_number(grid, "T_obs", SECONDS_PER_YEAR, "grid")
    T_obs > 0 || config_error("[grid].T_obs must be > 0, got $T_obs")
    f_min = get_number(grid, "f_min", 1.0e-3, "grid")
    f_max = get_number(grid, "f_max", 0.01, "grid")
    (f_min > 0 && f_max > f_min) ||
        config_error("[grid]: need 0 < f_min < f_max, got f_min = $f_min, f_max = $f_max")
    n_bins = floor(Int, (f_max - f_min) * T_obs) + 1
    n_bins >= 8 || config_error(
        "[grid]: only $n_bins frequency bins at df = 1/T_obs — " *
        "increase T_obs or the [f_min, f_max] band",
    )
    return (; T_obs, f_min, f_max)
end

"""
`[physics]` into a [`WaveformParams`](@ref); defaults are owned by the
struct and never restated here. The instrument's `arm_length` (parsed with
`[noise]`) sets the constellation eccentricity and the transfer frequency of
the response, so the noise model and the signal response share one arm
length.
"""
function parse_physics(config::AbstractDict, arm_length::Real)
    phys = get(config, "physics", Dict{String,Any}())
    warn_unknown_keys(phys, "physics")
    wp_default = WaveformParams()
    number(key, default) = get_number(phys, key, default, "physics")
    wp = WaveformParams(
        mass_scale = number("mass_scale", wp_default.mass_scale),
        time_scale = number("time_scale", wp_default.time_scale),
        distance_scale = number("distance_scale", wp_default.distance_scale),
        eta = number("eta", wp_default.eta),
        ecliptic_longitude = number("ecliptic_longitude", wp_default.ecliptic_longitude),
        ecliptic_latitude = number("ecliptic_latitude", wp_default.ecliptic_latitude),
        inclination = number("inclination", wp_default.inclination),
        polarization = number("polarization", wp_default.polarization),
        orbit_phase = number("orbit_phase", wp_default.orbit_phase),
        constellation_phase = number("constellation_phase", wp_default.constellation_phase),
        cutoff_width = number("cutoff_width", wp_default.cutoff_width),
        arm_length = arm_length,
        include_t_channel = get_boolean(
            phys, "include_t_channel", n_channels(wp_default) == 3, "physics"),
    )
    for (fname, val) in (("mass_scale", wp.mass_scale), ("time_scale", wp.time_scale),
        ("distance_scale", wp.distance_scale), ("cutoff_width", wp.cutoff_width))
        val > 0 || config_error("[physics].$fname must be > 0, got $val")
    end
    0.0 < wp.eta <= 0.25 || config_error(
        "[physics].eta must be in (0, 0.25] (symmetric mass ratio), got $(wp.eta)")
    abs(wp.ecliptic_latitude) <= π / 2 || config_error(
        "[physics].ecliptic_latitude must be in [-π/2, π/2], got $(wp.ecliptic_latitude)",
    )
    0.0 <= wp.inclination <= π ||
        config_error("[physics].inclination must be in [0, π], got $(wp.inclination)")
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
            config_error("[noise].$field must be > 0, got $(getfield(noise, field))")
    end
    noise.confusion_amp >= 0 || config_error("[noise].confusion_amp must be >= 0")
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
    map_n_angles >= 8 || config_error("[mapping].n_angles must be >= 8, got $map_n_angles")
    if isodd(map_n_angles)
        @warn "[mapping].n_angles = $map_n_angles is odd; rounding up to " *
              "$(map_n_angles + 1) (mirrored sampling needs an even count)."
        map_n_angles += 1
    end
    neighbor_ratio_tol = get_number(mapping, "neighbor_ratio_tol", 1.25, "mapping")
    neighbor_ratio_tol > 1 ||
        config_error("[mapping].neighbor_ratio_tol must be > 1, got $neighbor_ratio_tol")
    max_refine_levels = get_integer(mapping, "max_refine_levels", 6, "mapping")
    0 <= max_refine_levels <= 16 ||
        config_error("[mapping].max_refine_levels must be in 0:16, got $max_refine_levels")
    corner_bisect_iters = get_integer(mapping, "corner_bisect_iters", 25, "mapping")
    0 <= corner_bisect_iters <= 60 ||
        config_error(
            "[mapping].corner_bisect_iters must be in 0:60 (0 disables corner " *
            "bisection), got $corner_bisect_iters",
        )
    unbounded_cap_factor = get_number(mapping, "unbounded_cap_factor", 5.0, "mapping")
    unbounded_cap_factor > 1 ||
        config_error(
            "[mapping].unbounded_cap_factor must be > 1, got $unbounded_cap_factor",
        )
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
        config_error(
            "[hardware].gpu_backend must be auto | none | cuda | amdgpu | metal | oneapi, got '$gpu_str'",
        )
    max_threads = get_integer(hardware, "max_threads", Threads.nthreads(), "hardware")
    max_threads >= 1 ||
        config_error("[hardware].max_threads must be >= 1, got $max_threads")
    hessian_chunk = get_integer(hardware, "hessian_chunk", 0, "hardware")
    0 <= hessian_chunk <= 6 ||
        config_error(
            "[hardware].hessian_chunk must be in 0:6 (0 = full 6-parameter chunk), " *
            "got $hessian_chunk",
        )
    gc_between_stages = get_boolean(hardware, "gc_between_stages", true, "hardware")
    heap_size_hint_gb = get_number(hardware, "heap_size_hint_gb", 0.0, "hardware")
    heap_size_hint_gb >= 0 ||
        config_error(
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
    max_ram_gb > 0 || config_error("[safety].max_ram_gb must be > 0, got $max_ram_gb")
    bytes_per_bin_per_task_gpu =
        get_integer(safety, "bytes_per_bin_per_task_gpu", 1000, "safety")
    bytes_per_bin_per_task_gpu > 0 ||
        config_error(
            "[safety].bytes_per_bin_per_task_gpu must be > 0, got $bytes_per_bin_per_task_gpu",
        )
    max_vram_gb = get_number(safety, "max_vram_gb", 8.0, "safety")
    max_vram_gb > 0 || config_error("[safety].max_vram_gb must be > 0, got $max_vram_gb")
    os_vram_overhead_gb = get_number(safety, "os_vram_overhead_gb", 1.0, "safety")
    os_vram_overhead_gb >= 0 ||
        config_error("[safety].os_vram_overhead_gb must be >= 0, got $os_vram_overhead_gb")
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
        config_error("Invalid [parameter_bounds]: $(sprint(showerror, err))")
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
            config_error(
                "[[sweeps]] entry has missing or non-filesystem-safe name: $(repr(name))",
            )
        name in seen && config_error("Duplicate sweep/map name '$name'")
        push!(seen, name)
        theta0 =
            validate_theta6(get(s, "theta_0", nothing), "[[sweeps]] '$name'.theta_0")
        check_interior(theta0, bounds, "sweep '$name'")
        u = validate_theta6(get(s, "u_dir", nothing), "[[sweeps]] '$name'.u_dir")
        norm_u = sqrt(sum(abs2, u))
        norm_u > 0 || config_error("[[sweeps]] '$name'.u_dir must be nonzero")
        amp_ratio = get_number(s, "amp_ratio", 1.0, "sweeps[]")
        amp_ratio > 0 ||
            config_error("[[sweeps]] '$name'.amp_ratio must be > 0, got $amp_ratio")
        if amp_ratio != 1.0 && u[1] != 0
            config_error(
                "[[sweeps]] '$name': amp_ratio ≠ 1 requires u_dir[1] = 0 — amplitude " *
                "separation is expressed via amp_ratio (the A_harm law), not via the " *
                "sweep direction.",
            )
        end
        u[1] == 0 || config_error(
            "[[sweeps]] '$name'.u_dir must have a zero amplitude component: the " *
            "second source's amplitude is set by amp_ratio, so an amplitude " *
            "displacement in u_dir would enter the theory curve but not the data",
        )
        rho = get_number(s, "rho_thresh", sweep_rho_thresh, "sweeps[]")
        rho > 0 || config_error("[[sweeps]] '$name'.rho_thresh must be > 0")
        SweepSpec(name, theta0, u, rho, amp_ratio)
    end
    maps = map(map_tables) do m
        warn_unknown_keys(m, "maps[]")
        name = String(get(m, "name", ""))
        fs_safe(name) ||
            config_error(
                "[[maps]] entry has missing or non-filesystem-safe name: $(repr(name))",
            )
        name in seen && config_error("Duplicate sweep/map name '$name'")
        push!(seen, name)
        px = get_integer(m, "param_x", 0, "maps[]")
        py = get_integer(m, "param_y", 0, "maps[]")
        (1 <= px <= N_PARAMS && 1 <= py <= N_PARAMS) ||
            config_error(
                "[[maps]] '$name': param_x/param_y must be in 1:$(N_PARAMS), " *
                "got ($px, $py)",
            )
        px != py || config_error("[[maps]] '$name': param_x and param_y must differ")
        rho = get_number(m, "rho_thresh", sweep_rho_thresh, "maps[]")
        rho > 0 || config_error("[[maps]] '$name'.rho_thresh must be > 0")
        theta0 = validate_theta6(get(m, "theta_0", nothing), "[[maps]] '$name'.theta_0")
        check_interior(theta0, bounds, "map '$name'")
        na = get_integer(m, "n_angles", map_n_angles, "maps[]")
        na >= 8 || config_error("[[maps]] '$name'.n_angles must be >= 8, got $na")
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
        config_error(
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
            config_error(
                "$what: theta_0[$i] = $(theta0[i]) is not strictly inside its physical " *
                "bounds [$(b.lower[i]), $(b.upper[i])] — the local geometry expansion " *
                "and zone capping require an interior base point.",
            )
    end
    return nothing
end

end # module
