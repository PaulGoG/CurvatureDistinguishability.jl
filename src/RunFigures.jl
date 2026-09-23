"""
Single implementation of display-time figure regeneration from persisted
run artifacts, consumed by `scripts/replot.jl` and
`scripts/collect_plots.jl`. Point classification and refits reuse the
pipeline's own rules (`Fitting`), so run-time and display-time figures
cannot diverge.
"""
module RunFigures

using DocStringExtensions: TYPEDSIGNATURES
using CSV: CSV
using DataFrames: DataFrame
using TOML: TOML
using ..Bounds: deviation_box
using ..Geometry: boundary_radius, cap_unbounded_radii!, cap_at_prior
using ..Config: load_and_validate_config
using ..Plotting
using ..Fitting:
    MIN_FIT_POINTS, loglog_slope, ratio_correction_fit,
    above_floor_mask, perturbative_mask

export run_cases, sweep_figures, zone_map_figure

rho_tag(rho) = "_rho" * replace(string(rho), "." => "p")

"""
$(TYPEDSIGNATURES)

Read a persisted run table. The tables are kilobytes to megabytes, so
parsing is single-task by construction: the chunked multithreaded parser
mis-detects row boundaries on the wide residual-spectrum tables.
"""
read_run_table(path::AbstractString) = CSV.read(path, DataFrame; ntasks = 1)

"""
$(TYPEDSIGNATURES)

Enumerate the sweep and map case names present in a pipeline run directory
(sorted); a case counts when its primary CSV exists.
"""
function run_cases(run_dir::AbstractString)
    sweeps = String[]
    sdir = joinpath(run_dir, "sweeps")
    isdir(sdir) && for name in sort(readdir(sdir))
        isfile(joinpath(sdir, name, "results.csv")) && push!(sweeps, name)
    end
    maps = String[]
    mdir = joinpath(run_dir, "maps")
    isdir(mdir) && for name in sort(readdir(mdir))
        isfile(joinpath(mdir, name, "confusion_contour.csv")) && push!(maps, name)
    end
    return (sweeps = sweeps, maps = maps)
end

"""
Validated configuration from a run directory's config snapshot; errors when
the snapshot is absent (pre-snapshot runs cannot be regenerated faithfully).
"""
function run_config(run_dir::AbstractString)
    snapshot = joinpath(run_dir, "config.toml")
    isfile(snapshot) ||
        throw(
            ArgumentError("No config snapshot in $run_dir — cannot recover run metadata."),
        )
    return load_and_validate_config(snapshot)
end

# required key of a run's sweep_meta.toml
function meta_value(meta::AbstractDict, key::AbstractString, path::AbstractString)
    haskey(meta, key) ||
        throw(ArgumentError("$path lacks the key '$key'; the run predates this version"))
    return Float64(meta[key])
end

"""
$(TYPEDSIGNATURES)

Rebuild the figures of one 1D sweep from its persisted CSVs — no geometry or
optimization is recomputed. Point classification always follows the
production above-floor rule on the persisted optimizer floor
(`Fitting.above_floor_mask`). With `refit = false` the
annotated slope and correction coefficients are the persisted run-time
values from `sweep_meta.toml`; with `refit = true` they are refitted from
the CSV with the current fitting code (persisted metadata is never
modified). The scaling figure's legend quotes the perturbative-window
exponent `slope_window`, fitted over the points admitted to the `O(δ⁵)`
correction fit, and falls back to the all-clean slope when that window holds
fewer than `MIN_FIT_POINTS` points. With `refit = false` a missing or
sentinel (negative) `slope_window` in `sweep_meta.toml` is recomputed from
the CSV, so runs that predate the key regenerate with the window exponent.
With `rho` the discernibility threshold is rescaled exactly from
the persisted normalized curvature and the sweep's amplitude ratio: `suffix`
carries the `_rho…` tag and `residual` is `nothing` (the residual spectrum is
threshold-independent). `delta_min` is the threshold separation drawn in the
scaling figure.

Returns the NamedTuple `(scaling, residual, residual_threshold, suffix,
delta_min, slope, slope_err, slope_window, slope_window_err)`: the figures,
the threshold data, and the all-clean and window log-log slopes with their
standard errors (NaN when not fittable).
"""
function sweep_figures(run_dir::AbstractString, case::AbstractString;
    refit::Bool = false, rho::Union{Nothing,Real} = nothing)
    dir = joinpath(run_dir, "sweeps", case)
    res = read_run_table(joinpath(dir, "results.csv"))
    meta_path = joinpath(dir, "sweep_meta.toml")
    isfile(meta_path) || throw(
        ArgumentError(
            "No sweep_meta.toml in $dir — the sweep did not complete; " *
            "its figures cannot be rebuilt.",
        ),
    )
    meta = TOML.parsefile(meta_path)
    floor_level = Float64(get(meta, "floor_level", -1.0))
    floor_level < 0 && (floor_level = NaN)
    clean = above_floor_mask(res.D2_Numerical, floor_level)
    ratio = res.D2_Numerical ./ res.D2_Theoretical
    window = perturbative_mask(ratio, collect(clean),
        run_config(run_dir).correction_fit_max_departure)
    # exponent over the O(δ⁵) fit window (NaN below MIN_FIT_POINTS points)
    fit_window_slope() =
        count(window) >= MIN_FIT_POINTS ?
        loglog_slope(res.Delta[window], res.D2_Numerical[window]) : (NaN, NaN)
    if refit
        slope, slope_err =
            count(clean) >= MIN_FIT_POINTS ?
            loglog_slope(res.Delta[clean], res.D2_Numerical[clean]) : (NaN, NaN)
        slope_window, slope_window_err = fit_window_slope()
        c1, _, c2 = ratio_correction_fit(res.Delta[window], ratio[window])
    else
        slope = Float64(get(meta, "slope", NaN))
        slope_err = Float64(get(meta, "slope_err", NaN))
        c1 = Float64(get(meta, "c1", NaN))
        c2 = Float64(get(meta, "c2", NaN))
        # negative: sentinel for NaN, or a run that predates the key
        slope_window = Float64(get(meta, "slope_window", -1.0))
        slope_window_err = Float64(get(meta, "slope_window_err", -1.0))
        if slope_window < 0
            slope_window, slope_window_err = fit_window_slope()
        end
    end
    rho_sq = rho === nothing ? meta_value(meta, "rho_sq", meta_path) : Float64(rho)^2
    # D² = (p/16) K δ⁴ with p = (2q/(1+q))², q = A₂/A₁ (as in run_sweep)
    amp_ratio = meta_value(meta, "amp_ratio", meta_path)
    amp_prefactor = (2amp_ratio / (1 + amp_ratio))^2
    delta_min =
        rho === nothing ? meta_value(meta, "delta_min", meta_path) :
        (16.0 * rho_sq / (amp_prefactor * meta_value(meta, "K_u_norm", meta_path)))^(1 / 4)
    scaling = scaling_figure(res.Delta, res.D2_Numerical, res.D2_Theoretical;
        rho_sq = rho_sq, delta_min = delta_min,
        slope = isfinite(slope_window) ? slope_window : slope,
        slope_err = isfinite(slope_window) ? slope_window_err : slope_err,
        clean = collect(clean), floor_level = floor_level,
        c1 = c1, c2 = c2)
    residual = nothing
    spec_path = joinpath(dir, "residual_spectrum.csv")
    if rho === nothing && isfile(spec_path)
        spec = read_run_table(spec_path)
        residual = residual_figure(spec,
            ResidualFigureMeta(meta_value(meta, "delta_star", meta_path),
                meta_value(meta, "df", meta_path),
                meta_value(meta, "d2_num_star", meta_path),
                meta_value(meta, "d2_theo_star", meta_path)))
    end
    # threshold companion (persisted only when the threshold point lies
    # outside the validity window of the leading-order fit)
    residual_threshold = nothing
    thr_path = joinpath(dir, "residual_spectrum_threshold.csv")
    if rho === nothing && isfile(thr_path) && haskey(meta, "delta_thr")
        spec_thr = read_run_table(thr_path)
        residual_threshold = residual_figure(spec_thr,
            ResidualFigureMeta(meta_value(meta, "delta_thr", meta_path),
                meta_value(meta, "df", meta_path),
                meta_value(meta, "d2_num_thr", meta_path),
                meta_value(meta, "d2_theo_thr", meta_path));
            delta_symbol = "\\delta_{\\mathrm{thr}}")
    end
    return (scaling = scaling, residual = residual,
        residual_threshold = residual_threshold,
        suffix = rho === nothing ? "" : rho_tag(rho), delta_min = delta_min,
        slope = slope, slope_err = slope_err,
        slope_window = slope_window, slope_window_err = slope_window_err)
end

"""
$(TYPEDSIGNATURES)

Rebuild one 2D confusion-zone figure from the persisted contour CSV and the
run's config snapshot (which supplies the plane, base point and prior box) —
no curvature is recomputed. With `rho` the boundary is rescaled exactly from
the persisted per-direction curvature (`r_math = (16ρ²/K)^{1/4}`), re-capped
at the stored prior box; `contour` then holds the rescaled contour table for
persisting alongside the original (`nothing` otherwise) and `suffix` carries
the `_rho…` tag. Caveat: the angular refinement was driven by the original
threshold's prior-box crossover, so segments near a new crossover may be
under-refined — rerun the map stage for publication-grade maps at a very
different ρ.
"""
function zone_map_figure(run_dir::AbstractString, case::AbstractString;
    rho::Union{Nothing,Real} = nothing)
    dir = joinpath(run_dir, "maps", case)
    contour_stored = read_run_table(joinpath(dir, "confusion_contour.csv"))
    cfg = run_config(run_dir)
    matches = filter(m -> m.name == case, cfg.maps)
    isempty(matches) &&
        throw(
            ArgumentError("Map '$case' is not present in the config snapshot of $run_dir."),
        )
    map_cfg = only(matches)
    px, py = map_cfg.param_x, map_cfg.param_y
    theta0 = map_cfg.theta_0
    box = deviation_box(cfg.bounds, theta0, px, py)
    required = (:Angle, :X_Bound, :Y_Bound, :Dir_Cos, :Dir_Sin, :R_Math, :R_Box,
        :Prior_Limited, :Degenerate, :K_Raw, :G_uu)
    missing_cols = filter(c -> !hasproperty(contour_stored, c), required)
    isempty(missing_cols) || throw(
        ArgumentError(
            "contour CSV of map '$case' lacks the columns $(join(missing_cols, ", ")); " *
            "regenerate the run with the current pipeline",
        ),
    )
    degen = collect(Bool, contour_stored.Degenerate)
    contour = nothing
    suffix = ""
    if rho === nothing
        X, Y = contour_stored.X_Bound, contour_stored.Y_Bound
        prior = collect(Bool, contour_stored.Prior_Limited)
        x_math = contour_stored.R_Math .* contour_stored.Dir_Cos
        y_math = contour_stored.R_Math .* contour_stored.Dir_Sin
    else
        suffix = rho_tag(rho)
        r_math = boundary_radius.(contour_stored.K_Raw, Float64(rho)^2)
        capped = cap_at_prior.(r_math, contour_stored.R_Box)
        r_cap = first.(capped)
        prior = collect(last.(capped))
        cap_unbounded_radii!(r_cap, cfg.unbounded_cap_factor)
        dir_cos = contour_stored.Dir_Cos
        dir_sin = contour_stored.Dir_Sin
        X = r_cap .* dir_cos
        Y = r_cap .* dir_sin
        x_math = r_math .* dir_cos
        y_math = r_math .* dir_sin
        contour = DataFrame(Angle = contour_stored.Angle, X_Bound = X, Y_Bound = Y,
            Dir_Cos = dir_cos, Dir_Sin = dir_sin,
            R_Capped = r_cap, R_Math = r_math, R_Box = contour_stored.R_Box,
            Prior_Limited = prior, Degenerate = degen,
            K_Raw = contour_stored.K_Raw, G_uu = contour_stored.G_uu)
    end
    figure = zone_figure(X, Y, prior;
        px = px, py = py, box = box,
        prior_frac = count(prior) / max(1, length(prior)),
        degenerate_frac = count(degen) / max(1, length(degen)),
        x_math = x_math, y_math = y_math)
    return (figure = figure, contour = contour, suffix = suffix)
end

end # module
