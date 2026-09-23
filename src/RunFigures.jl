"""
Single implementation of display-time figure regeneration from persisted
run artifacts, consumed by `scripts/replot.jl` and
`scripts/collect_plots.jl`, and of the layout-driven multi-panel composites
behind `scripts/compose_figures.jl`. Point classification and refits reuse the
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

export run_cases, sweep_figures, zone_map_figure, composite_figures,
    sweep_panel_data, zone_panel_data, residual_panel_data

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

Scaling-panel data of one 1D sweep from its persisted CSVs — no geometry or
optimization is recomputed. Point classification always follows the
production above-floor rule on the persisted optimizer floor
(`Fitting.above_floor_mask`). With `refit = false` the slopes and correction
coefficients are the persisted run-time values from `sweep_meta.toml`; with
`refit = true` they are refitted from the CSV with the current fitting code
(persisted metadata is never modified). With `refit = false` a missing or
sentinel (negative) `slope_window` in `sweep_meta.toml` is recomputed from
the CSV, so runs that predate the key regenerate with the window exponent.

Returns the NamedTuple `(deltas, d2_num, d2_theo, rho_sq, delta_min, slope,
slope_err, clean, floor_level, c1, c2, slope_all, slope_all_err, slope_window,
slope_window_err)`, accepted panel by panel by
`Plotting.composite_scaling_figure`. `slope`/`slope_err` are the exponent the
scaling figure quotes: the perturbative-window exponent `slope_window`, fitted
over the points admitted to the `O(δ⁵)` correction fit, falling back to the
all-clean slope `slope_all` when that window holds fewer than
`MIN_FIT_POINTS` points. Standard errors are NaN when not fittable;
`floor_level` is NaN when no optimizer floor was detected.
"""
function sweep_panel_data(run_dir::AbstractString, case::AbstractString;
    refit::Bool = false)
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
        slope_all, slope_all_err =
            count(clean) >= MIN_FIT_POINTS ?
            loglog_slope(res.Delta[clean], res.D2_Numerical[clean]) : (NaN, NaN)
        slope_window, slope_window_err = fit_window_slope()
        c1, _, c2 = ratio_correction_fit(res.Delta[window], ratio[window])
    else
        slope_all = Float64(get(meta, "slope", NaN))
        slope_all_err = Float64(get(meta, "slope_err", NaN))
        c1 = Float64(get(meta, "c1", NaN))
        c2 = Float64(get(meta, "c2", NaN))
        # negative: sentinel for NaN, or a run that predates the key
        slope_window = Float64(get(meta, "slope_window", -1.0))
        slope_window_err = Float64(get(meta, "slope_window_err", -1.0))
        if slope_window < 0
            slope_window, slope_window_err = fit_window_slope()
        end
    end
    windowed = isfinite(slope_window)
    return (deltas = res.Delta, d2_num = res.D2_Numerical, d2_theo = res.D2_Theoretical,
        rho_sq = meta_value(meta, "rho_sq", meta_path),
        delta_min = meta_value(meta, "delta_min", meta_path),
        slope = windowed ? slope_window : slope_all,
        slope_err = windowed ? slope_window_err : slope_all_err,
        clean = collect(clean), floor_level = floor_level, c1 = c1, c2 = c2,
        slope_all = slope_all, slope_all_err = slope_all_err,
        slope_window = slope_window, slope_window_err = slope_window_err)
end

"""
$(TYPEDSIGNATURES)

Residual-panel data of one 1D sweep from its persisted spectrum table:
`residual_spectrum.csv` with the `δ*` annotation values for
`spectrum = "star"`, or the threshold companion
`residual_spectrum_threshold.csv` with the `δ_thr` values for
`spectrum = "threshold"`. Throws an `ArgumentError` when the requested table
or the sweep metadata does not exist.

Returns the NamedTuple `(spec, meta, delta_symbol)`, accepted panel by panel
by `Plotting.composite_residual_figure`.
"""
function residual_panel_data(run_dir::AbstractString, case::AbstractString;
    spectrum::AbstractString = "star")
    spectrum in ("star", "threshold") || throw(
        ArgumentError("spectrum = \"$spectrum\" must be one of: \"star\" | \"threshold\""),
    )
    dir = joinpath(run_dir, "sweeps", case)
    star = spectrum == "star"
    table = star ? "residual_spectrum.csv" : "residual_spectrum_threshold.csv"
    path = joinpath(dir, table)
    isfile(path) || throw(ArgumentError("No $table in $dir."))
    meta_path = joinpath(dir, "sweep_meta.toml")
    isfile(meta_path) || throw(ArgumentError("No sweep_meta.toml in $dir."))
    meta = TOML.parsefile(meta_path)
    tag = star ? "star" : "thr"
    figure_meta = ResidualFigureMeta(
        meta_value(meta, star ? "delta_star" : "delta_thr", meta_path),
        meta_value(meta, "df", meta_path),
        meta_value(meta, "d2_num_" * tag, meta_path),
        meta_value(meta, "d2_theo_" * tag, meta_path))
    return (spec = read_run_table(path), meta = figure_meta,
        delta_symbol = star ? "\\delta^*" : "\\delta_{\\mathrm{thr}}")
end

"""
$(TYPEDSIGNATURES)

Rebuild the figures of one 1D sweep from its persisted CSVs — no geometry or
optimization is recomputed. The scaling data are loaded by
[`sweep_panel_data`](@ref) (point classification, `refit` semantics and the
quoted window exponent with its fallback are described there), the residual
spectra by [`residual_panel_data`](@ref).
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
    data = sweep_panel_data(run_dir, case; refit = refit)
    dir = joinpath(run_dir, "sweeps", case)
    meta_path = joinpath(dir, "sweep_meta.toml")
    meta = TOML.parsefile(meta_path)
    if rho === nothing
        rho_sq = data.rho_sq
        delta_min = data.delta_min
    else
        rho_sq = Float64(rho)^2
        # D² = (p/16) K δ⁴ with p = (2q/(1+q))², q = A₂/A₁ (as in run_sweep)
        amp_ratio = meta_value(meta, "amp_ratio", meta_path)
        amp_prefactor = (2amp_ratio / (1 + amp_ratio))^2
        K_u_norm = meta_value(meta, "K_u_norm", meta_path)
        delta_min = (16.0 * rho_sq / (amp_prefactor * K_u_norm))^(1 / 4)
    end
    scaling = scaling_figure(data.deltas, data.d2_num, data.d2_theo;
        rho_sq = rho_sq, delta_min = delta_min,
        slope = data.slope, slope_err = data.slope_err,
        clean = data.clean, floor_level = data.floor_level,
        c1 = data.c1, c2 = data.c2)
    residual = nothing
    if rho === nothing && isfile(joinpath(dir, "residual_spectrum.csv"))
        star = residual_panel_data(run_dir, case; spectrum = "star")
        residual = residual_figure(star.spec, star.meta; delta_symbol = star.delta_symbol)
    end
    # threshold companion (persisted only when the threshold point lies
    # outside the validity window of the leading-order fit)
    residual_threshold = nothing
    thr_path = joinpath(dir, "residual_spectrum_threshold.csv")
    if rho === nothing && isfile(thr_path) && haskey(meta, "delta_thr")
        thr = residual_panel_data(run_dir, case; spectrum = "threshold")
        residual_threshold =
            residual_figure(thr.spec, thr.meta; delta_symbol = thr.delta_symbol)
    end
    return (scaling = scaling, residual = residual,
        residual_threshold = residual_threshold,
        suffix = rho === nothing ? "" : rho_tag(rho), delta_min = delta_min,
        slope = data.slope_all, slope_err = data.slope_all_err,
        slope_window = data.slope_window, slope_window_err = data.slope_window_err)
end

"""
$(TYPEDSIGNATURES)

Persisted contour table of map `case` with the plane, prior box and
degeneracy flags recovered from the run's config snapshot:
`(contour, cfg, px, py, box, degen)`. Throws an `ArgumentError` when the map
is absent from the snapshot or the table lacks a required column.
"""
function load_map(run_dir::AbstractString, case::AbstractString)
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
    return (contour = contour_stored, cfg = cfg, px = px, py = py, box = box,
        degen = degen)
end

"""
$(TYPEDSIGNATURES)

Zone-panel data of one 2D confusion map from the persisted contour CSV and the
run's config snapshot (plane, base point and prior box) at the run's own
threshold — no curvature is recomputed.

Returns the NamedTuple `(x, y, prior_limited, px, py, box, prior_frac,
degenerate_frac, x_math, y_math)`, accepted panel by panel by
`Plotting.composite_zone_figure`: the capped boundary polygon with its
prior-limited flags, the parameter indices of the plane, the deviation-space
prior box, the prior-limited and degenerate direction fractions, and the
uncapped mathematical contour.
"""
function zone_panel_data(run_dir::AbstractString, case::AbstractString)
    stored = load_map(run_dir, case)
    contour = stored.contour
    prior = collect(Bool, contour.Prior_Limited)
    return (x = contour.X_Bound, y = contour.Y_Bound, prior_limited = prior,
        px = stored.px, py = stored.py, box = stored.box,
        prior_frac = count(prior) / max(1, length(prior)),
        degenerate_frac = count(stored.degen) / max(1, length(stored.degen)),
        x_math = contour.R_Math .* contour.Dir_Cos,
        y_math = contour.R_Math .* contour.Dir_Sin)
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
different ρ. The unscaled map is loaded by [`zone_panel_data`](@ref).
"""
function zone_map_figure(run_dir::AbstractString, case::AbstractString;
    rho::Union{Nothing,Real} = nothing)
    if rho === nothing
        d = zone_panel_data(run_dir, case)
        figure = zone_figure(d.x, d.y, d.prior_limited;
            px = d.px, py = d.py, box = d.box,
            prior_frac = d.prior_frac, degenerate_frac = d.degenerate_frac,
            x_math = d.x_math, y_math = d.y_math)
        return (figure = figure, contour = nothing, suffix = "")
    end
    stored = load_map(run_dir, case)
    contour_stored = stored.contour
    degen = stored.degen
    suffix = rho_tag(rho)
    r_math = boundary_radius.(contour_stored.K_Raw, Float64(rho)^2)
    capped = cap_at_prior.(r_math, contour_stored.R_Box)
    r_cap = first.(capped)
    prior = collect(last.(capped))
    cap_unbounded_radii!(r_cap, stored.cfg.unbounded_cap_factor)
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
    figure = zone_figure(X, Y, prior;
        px = stored.px, py = stored.py, box = stored.box,
        prior_frac = count(prior) / max(1, length(prior)),
        degenerate_frac = count(degen) / max(1, length(degen)),
        x_math = x_math, y_math = y_math)
    return (figure = figure, contour = contour, suffix = suffix)
end

# --- layout-driven composites ------------------------------------------------

const LAYOUT_FIGURE_KEYS =
    ("name", "kind", "ncols", "panel_width", "panel_height", "labels", "panels")
const LAYOUT_PANEL_KEYS = ("run", "case", "spectrum")
const COMPOSITE_KINDS = ("scaling", "zone", "residual")

# default panel height of each composite kind
default_panel_height(kind::AbstractString) =
    kind == "scaling" ? 620 : kind == "zone" ? 520 : 640

# required string entry of a layout table
function layout_string(table::AbstractDict, key::String, scope::String,
    path::AbstractString)
    haskey(table, key) || throw(ArgumentError("$path: $scope lacks the key '$key'"))
    value = table[key]
    value isa AbstractString ||
        throw(ArgumentError("$path: $scope.$key = $(repr(value)) must be a string"))
    return String(value)
end

# optional positive integer entry of a layout table
function layout_positive_int(table::AbstractDict, key::String, default::Int,
    path::AbstractString)
    value = get(table, key, default)
    (value isa Integer && value >= 1) || throw(
        ArgumentError("$path: figures.$key = $(repr(value)) must be an integer >= 1"),
    )
    return Int(value)
end

"""
$(TYPEDSIGNATURES)

Panel data of one `[[figures.panels]]` entry of a composite layout at `path`,
for a figure of `kind`; relative run directories resolve against `base_dir`,
the layout file's directory.
"""
function layout_panel(entry::AbstractDict, kind::String, base_dir::AbstractString,
    path::AbstractString)
    for key in keys(entry)
        key in LAYOUT_PANEL_KEYS ||
            throw(ArgumentError("$path: unknown key '$key' in [[figures.panels]]"))
    end
    run = layout_string(entry, "run", "figures.panels", path)
    case = layout_string(entry, "case", "figures.panels", path)
    run_dir = isabspath(run) ? run : normpath(joinpath(base_dir, run))
    isfile(joinpath(run_dir, "config.toml")) || throw(
        ArgumentError(
            "$path: figures.panels.run = \"$run\" has no config.toml ($run_dir)",
        ),
    )
    if kind == "residual"
        spectrum = get(entry, "spectrum", "star")
        spectrum in ("star", "threshold") || throw(
            ArgumentError(
                "$path: figures.panels.spectrum = $(repr(spectrum)) must be one of: " *
                "\"star\" | \"threshold\"",
            ),
        )
        return residual_panel_data(run_dir, case; spectrum = spectrum)
    end
    haskey(entry, "spectrum") && throw(
        ArgumentError("$path: figures.panels.spectrum applies to kind = \"residual\" only"),
    )
    return kind == "scaling" ? sweep_panel_data(run_dir, case) :
           zone_panel_data(run_dir, case)
end

"""
$(TYPEDSIGNATURES)

Composite figure of one `[[figures]]` entry of the layout at `path`:
validates the entry, loads its panels and draws them with the composite
builder of its kind. Returns `(name, figure)`.
"""
function layout_figure(entry::AbstractDict, base_dir::AbstractString,
    path::AbstractString)
    for key in keys(entry)
        key in LAYOUT_FIGURE_KEYS ||
            throw(ArgumentError("$path: unknown key '$key' in [[figures]]"))
    end
    name = layout_string(entry, "name", "figures", path)
    kind = layout_string(entry, "kind", "figures", path)
    kind in COMPOSITE_KINDS || throw(
        ArgumentError(
            "$path: figures.kind = \"$kind\" must be one of: " *
            "\"scaling\" | \"zone\" | \"residual\"",
        ),
    )
    ncols = layout_positive_int(entry, "ncols", 2, path)
    width = layout_positive_int(entry, "panel_width", 600, path)
    height = layout_positive_int(entry, "panel_height", default_panel_height(kind), path)
    labels = get(entry, "labels", String[])
    (labels isa AbstractVector && all(l -> l isa AbstractString, labels)) || throw(
        ArgumentError("$path: figures.labels of '$name' must be an array of strings"),
    )
    panels = get(entry, "panels", nothing)
    (
        panels isa AbstractVector && !isempty(panels) &&
        all(p -> p isa AbstractDict, panels)
    ) ||
        throw(ArgumentError("$path: figure '$name' needs at least one [[figures.panels]]"))
    data = [layout_panel(p, kind, base_dir, path) for p in panels]
    builder =
        kind == "scaling" ? composite_scaling_figure :
        kind == "zone" ? composite_zone_figure : composite_residual_figure
    figure = builder(data; ncols = ncols, panel_size = (width, height),
        labels = String[String(l) for l in labels])
    return (name = name, figure = figure)
end

"""
$(TYPEDSIGNATURES)

Build the multi-panel publication figures described by the TOML layout at
`layout_path` from the persisted tables of completed runs — no geometry or
optimization is recomputed. Each `[[figures]]` table holds `name` (output
base name), `kind` (`"scaling"`, `"zone"` or `"residual"`), the optional
`ncols` (default 2), `panel_width` (default 600), `panel_height` (default 620
scaling, 520 zone, 640 residual) and `labels` (optional panel labels, one per
panel; none by default), and
its `[[figures.panels]]` tables hold `run` (run directory, absolute or relative
to the layout file's directory), `case` (sweep or map name) and, for residual
figures only, `spectrum` (`"star"` or `"threshold"`, default `"star"`).
Panel data come from [`sweep_panel_data`](@ref), [`zone_panel_data`](@ref) and
[`residual_panel_data`](@ref); the figures from the composite builders of
`Plotting`. Unknown keys, invalid values and run directories without a
`config.toml` throw an `ArgumentError` naming the offending key or value.

Returns a `Vector` of `(name, figure)` NamedTuples in file order.

# Example
```julia
for (name, figure) in composite_figures("configs/figures/quickstart_composites.toml")
    save_figure(figure, joinpath("plots", name))
end
```
"""
function composite_figures(layout_path::AbstractString)
    isfile(layout_path) ||
        throw(ArgumentError("Composite layout not found: $layout_path"))
    layout = TOML.parsefile(layout_path)
    for key in keys(layout)
        key == "figures" ||
            throw(ArgumentError("$layout_path: unknown top-level key '$key'"))
    end
    figures = get(layout, "figures", nothing)
    (
        figures isa AbstractVector && !isempty(figures) &&
        all(f -> f isa AbstractDict, figures)
    ) ||
        throw(ArgumentError("$layout_path: at least one [[figures]] table is required"))
    base_dir = dirname(abspath(layout_path))
    return [layout_figure(entry, base_dir, layout_path) for entry in figures]
end

end # module
