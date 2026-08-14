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
using DataFrames: DataFrame, nrow
using TOML: TOML
using ..Bounds: deviation_box
using ..Geometry: K_UNDERFLOW
using ..Config: load_and_validate_config
using ..Plotting
using ..Fitting: MIN_FIT_POINTS, loglog_slope, ratio_correction_fit,
    above_floor_mask

export run_cases, sweep_figures, zone_map_figure

rho_tag(rho) = "_rho" * replace(string(rho), "." => "p")

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
        error("No config snapshot in $run_dir — cannot recover run metadata.")
    return load_and_validate_config(snapshot)
end

"""
$(TYPEDSIGNATURES)

Rebuild the figures of one 1D sweep from its persisted CSVs — no geometry or
optimization is recomputed. Point classification always follows the
production above-floor rule on the persisted optimizer floor
(`Orchestrator.above_floor_mask`). With `refit = false` the
annotated slope and correction coefficients are the persisted run-time
values from `sweep_meta.toml`; with `refit = true` they are refitted from
the CSV with the current fitting code (persisted metadata is never
modified). With `rho` the discernibility threshold is rescaled exactly from
the persisted normalized curvature: `suffix` carries the `_rho…` tag and
`residual` is `nothing` (the residual spectrum is threshold-independent).
"""
function sweep_figures(run_dir::AbstractString, case::AbstractString;
    refit::Bool = false, rho::Union{Nothing,Real} = nothing)
    dir = joinpath(run_dir, "sweeps", case)
    res = CSV.read(joinpath(dir, "results.csv"), DataFrame)
    meta_path = joinpath(dir, "sweep_meta.toml")
    meta = isfile(meta_path) ? TOML.parsefile(meta_path) : Dict{String,Any}()
    floor_level = Float64(get(meta, "floor_level", -1.0))
    floor_level < 0 && (floor_level = NaN)
    clean = above_floor_mask(res.D2_Numerical, floor_level)
    if refit
        slope, slope_err =
            count(clean) >= MIN_FIT_POINTS ?
            loglog_slope(res.Delta[clean], res.D2_Numerical[clean]) : (NaN, NaN)
        ratio = res.D2_Numerical ./ res.D2_Theoretical
        c1, _, c2 = ratio_correction_fit(res.Delta[clean], ratio[clean])
    else
        slope = Float64(get(meta, "slope", NaN))
        slope_err = Float64(get(meta, "slope_err", NaN))
        c1 = Float64(get(meta, "c1", NaN))
        c2 = Float64(get(meta, "c2", NaN))
    end
    rho_sq = rho === nothing ? Float64(get(meta, "rho_sq", 1.0)) : Float64(rho)^2
    delta_min =
        rho === nothing ? Float64(get(meta, "delta_min", NaN)) :
        (16.0 * rho_sq / Float64(get(meta, "K_u_norm", NaN)))^(1 / 4)
    scaling = scaling_figure(res.Delta, res.D2_Numerical, res.D2_Theoretical;
        rho_sq = rho_sq, delta_min = delta_min,
        slope = slope, slope_err = slope_err,
        clean = collect(clean), floor_level = floor_level,
        c1 = c1, c2 = c2)
    residual = nothing
    spec_path = joinpath(dir, "residual_spectrum.csv")
    if rho === nothing && isfile(spec_path)
        spec = CSV.read(spec_path, DataFrame)
        residual = residual_figure(spec,
            (delta_star = Float64(get(meta, "delta_star", NaN)),
                df = Float64(get(meta, "df", 1.0)),
                d2_num = Float64(get(meta, "d2_num_star", NaN)),
                d2_theo = Float64(get(meta, "d2_theo_star", NaN))))
    end
    return (scaling = scaling, residual = residual,
        suffix = rho === nothing ? "" : rho_tag(rho))
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
    contour_stored = CSV.read(joinpath(dir, "confusion_contour.csv"), DataFrame)
    cfg = run_config(run_dir)
    matches = filter(m -> String(m["name"]) == case, cfg.maps)
    isempty(matches) &&
        error("Map '$case' is not present in the config snapshot of $run_dir.")
    map_cfg = only(matches)
    px, py = Int(map_cfg["param_x"]), Int(map_cfg["param_y"])
    theta0 = Float64.(map_cfg["theta_0"])
    box = deviation_box(cfg.bounds, theta0, px, py)
    n = nrow(contour_stored)
    degen =
        hasproperty(contour_stored, :Degenerate) ?
        collect(Bool, contour_stored.Degenerate) : falses(n)
    contour = nothing
    suffix = ""
    if rho === nothing
        X, Y = contour_stored.X_Bound, contour_stored.Y_Bound
        prior =
            hasproperty(contour_stored, :Prior_Limited) ?
            collect(Bool, contour_stored.Prior_Limited) : falses(n)
        has_math =
            hasproperty(contour_stored, :R_Math) && hasproperty(contour_stored, :Dir_Cos)
        x_math = has_math ? contour_stored.R_Math .* contour_stored.Dir_Cos : nothing
        y_math = has_math ? contour_stored.R_Math .* contour_stored.Dir_Sin : nothing
    else
        suffix = rho_tag(rho)
        hasproperty(contour_stored, :K_Raw) && hasproperty(contour_stored, :R_Box) ||
            error(
                "contour CSV lacks K_Raw/R_Box columns (pre-upgrade run) — " *
                "rerun the pipeline to enable threshold rescaling",
            )
        r_math = [
            K > K_UNDERFLOW ? (16.0 * Float64(rho)^2 / K)^(1 / 4) : Inf
            for K in contour_stored.K_Raw
        ]
        r_cap = min.(r_math, contour_stored.R_Box)
        if any(!isfinite, r_cap)
            r_cap_max = maximum(filter(isfinite, r_cap); init = 1.0)
            r_cap[.!isfinite.(r_cap)] .= cfg.unbounded_cap_factor * r_cap_max
        end
        dc =
            hasproperty(contour_stored, :Dir_Cos) ? contour_stored.Dir_Cos :
            cos.(contour_stored.Angle)
        ds =
            hasproperty(contour_stored, :Dir_Sin) ? contour_stored.Dir_Sin :
            sin.(contour_stored.Angle)
        X = r_cap .* dc
        Y = r_cap .* ds
        x_math = r_math .* dc
        y_math = r_math .* ds
        prior =
            collect(isfinite.(contour_stored.R_Box) .& (r_math .>= contour_stored.R_Box))
        contour = DataFrame(Angle = contour_stored.Angle, X_Bound = X, Y_Bound = Y,
            Dir_Cos = dc, Dir_Sin = ds,
            R_Capped = r_cap, R_Math = r_math, R_Box = contour_stored.R_Box,
            Prior_Limited = prior, Degenerate = degen,
            K_Raw = contour_stored.K_Raw,
            G_uu = hasproperty(contour_stored, :G_uu) ?
                   contour_stored.G_uu : fill(NaN, n))
    end
    figure = zone_figure(X, Y, prior;
        px = px, py = py, box = box,
        prior_frac = count(prior) / max(1, length(prior)),
        degenerate_frac = count(degen) / max(1, length(degen)),
        x_math = x_math, y_math = y_math)
    return (figure = figure, contour = contour, suffix = suffix)
end

end # module
