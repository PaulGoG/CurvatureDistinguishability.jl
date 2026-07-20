# Regenerate every figure of a completed run from its persisted CSVs —
# no geometry or optimization is recomputed.
#
#   julia --project scripts/replot.jl <run_dir>
#
using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)

using CSV
using DataFrames
using TOML
using TwoWaveformDistinguishability
using TwoWaveformDistinguishability.Bounds: deviation_box
using TwoWaveformDistinguishability.Config: load_and_validate_config

length(ARGS) == 1 || error("Usage: julia --project scripts/replot.jl <run_dir>")
const RUN_DIR = abspath(ARGS[1])
isdir(RUN_DIR) || error("Run directory not found: $RUN_DIR")
config_snapshot = joinpath(RUN_DIR, "config.toml")
isfile(config_snapshot) || error("No config snapshot in $RUN_DIR — cannot recover map metadata.")
cfg = load_and_validate_config(config_snapshot)

replotted = 0

sweeps_dir = joinpath(RUN_DIR, "sweeps")
if isdir(sweeps_dir)
    for name in sort(readdir(sweeps_dir))
        dir = joinpath(sweeps_dir, name)
        res_path = joinpath(dir, "results.csv")
        meta_path = joinpath(dir, "sweep_meta.toml")
        isfile(res_path) || continue
        try
            res = CSV.read(res_path, DataFrame)
            meta = isfile(meta_path) ? TOML.parsefile(meta_path) : Dict{String,Any}()
            ratio = res.D2_Numerical ./ res.D2_Theoretical
            conv = hasproperty(res, :Converged) ? res.Converged : trues(nrow(res))
            clean = (ratio .> 0.5) .& (ratio .< 2.0) .& conv
            floor_level = Float64(get(meta, "floor_level", -1.0))
            floor_level < 0 && (floor_level = NaN)
            fig = scaling_figure(res.Delta, res.D2_Numerical, res.D2_Theoretical;
                                 rho_sq = Float64(get(meta, "rho_sq", 1.0)),
                                 delta_min = Float64(get(meta, "delta_min", NaN)),
                                 slope = Float64(get(meta, "slope", NaN)),
                                 slope_err = Float64(get(meta, "slope_err", NaN)),
                                 clean = collect(clean), floor_level = floor_level)
            save_figure(fig, joinpath(dir, "scaling_plot"))
            spec_path = joinpath(dir, "residual_spectrum.csv")
            if isfile(spec_path)
                spec = CSV.read(spec_path, DataFrame)
                rfig = residual_figure(spec, (delta_star = Float64(get(meta, "delta_star", NaN)),
                                              df = Float64(get(meta, "df", 1.0)),
                                              d2_num = Float64(get(meta, "d2_num_star", NaN)),
                                              d2_theo = Float64(get(meta, "d2_theo_star", NaN))))
                save_figure(rfig, joinpath(dir, "residual_plot"))
            end
            global replotted += 1
            println("replotted sweep: $name")
        catch err
            @warn "Failed to replot sweep '$name'" exception = (err, catch_backtrace())
        end
    end
end

maps_dir = joinpath(RUN_DIR, "maps")
if isdir(maps_dir)
    map_meta = Dict(String(m["name"]) => m for m in cfg.maps)
    for name in sort(readdir(maps_dir))
        dir = joinpath(maps_dir, name)
        csv_path = joinpath(dir, "confusion_contour.csv")
        isfile(csv_path) || continue
        haskey(map_meta, name) ||
            (@warn "Map '$name' not present in the config snapshot; skipping."; continue)
        try
            m = map_meta[name]
            px, py = Int(m["param_x"]), Int(m["param_y"])
            theta0 = Float64.(m["theta_0"])
            box = deviation_box(cfg.bounds, theta0, px, py)
            df_map = CSV.read(csv_path, DataFrame)
            prior = hasproperty(df_map, :Prior_Limited) ? collect(Bool, df_map.Prior_Limited) :
                    falses(nrow(df_map))
            degen = hasproperty(df_map, :Degenerate) ? collect(Bool, df_map.Degenerate) :
                    falses(nrow(df_map))
            fig = zone_figure(df_map.Angle, df_map.X_Bound, df_map.Y_Bound, prior;
                              px = px, py = py, box = box,
                              prior_frac = count(prior) / max(1, length(prior)),
                              degenerate_frac = count(degen) / max(1, length(degen)))
            save_figure(fig, joinpath(dir, "confusion_zone"))
            global replotted += 1
            println("replotted map: $name")
        catch err
            @warn "Failed to replot map '$name'" exception = (err, catch_backtrace())
        end
    end
end

println("Replot complete: $replotted item(s) regenerated in $RUN_DIR")
