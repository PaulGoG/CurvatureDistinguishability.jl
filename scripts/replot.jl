# Regenerate every figure of a completed run from its persisted CSVs —
# no geometry or optimization is recomputed.
#
#   julia --project scripts/replot.jl <run_dir> [--rho R]
#
# With --rho R the discernibility threshold is changed WITHOUT recomputation:
# the boundary radius is r = (16ρ²/K)^{1/4} and K is persisted per direction,
# so maps (and the scaling-plot threshold markers) are rescaled exactly and
# written to *_rho<R> files, leaving the originals untouched. Caveat: the
# angular refinement was driven by the original threshold's prior-box
# crossover; segments near a *new* crossover may be under-refined — rerun the
# pipeline for publication-grade maps at a very different ρ.
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

args = copy(ARGS)
rho_new = nothing
if (idx = findfirst(==("--rho"), args)) !== nothing
    idx < length(args) || error("--rho requires a value")
    rho_new = parse(Float64, args[idx+1])
    rho_new > 0 || error("--rho must be > 0")
    deleteat!(args, idx:idx+1)
end
length(args) == 1 || error("Usage: julia --project scripts/replot.jl <run_dir> [--rho R]")
rho_tag(r) = "_rho" * replace(string(r), "." => "p")
const RUN_DIR = abspath(args[1])
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
            rho_sq_eff = rho_new === nothing ? Float64(get(meta, "rho_sq", 1.0)) : rho_new^2
            Kn = Float64(get(meta, "K_u_norm", NaN))
            delta_min_eff = rho_new === nothing ? Float64(get(meta, "delta_min", NaN)) :
                            (16.0 * rho_sq_eff / Kn)^(1 / 4)
            suffix = rho_new === nothing ? "" : rho_tag(rho_new)
            fig = scaling_figure(res.Delta, res.D2_Numerical, res.D2_Theoretical;
                                 rho_sq = rho_sq_eff, delta_min = delta_min_eff,
                                 slope = Float64(get(meta, "slope", NaN)),
                                 slope_err = Float64(get(meta, "slope_err", NaN)),
                                 clean = collect(clean), floor_level = floor_level,
                                 c1 = Float64(get(meta, "c1", NaN)),
                                 c2 = Float64(get(meta, "c2", NaN)))
            save_figure(fig, joinpath(dir, "scaling_plot" * suffix))
            spec_path = joinpath(dir, "residual_spectrum.csv")
            if rho_new === nothing && isfile(spec_path)
                spec = CSV.read(spec_path, DataFrame)
                # f_min/f_max come from the config snapshot (cfg), so the
                # band-edge ticks/limits work even for runs whose sweep_meta
                # predates the f_min/f_max fields.
                rfig = residual_figure(spec, (delta_star = Float64(get(meta, "delta_star", NaN)),
                                              df = Float64(get(meta, "df", 1.0)),
                                              f_min = cfg.f_min, f_max = cfg.f_max,
                                              d2_num = Float64(get(meta, "d2_num_star", NaN)),
                                              d2_theo = Float64(get(meta, "d2_theo_star", NaN))))
                save_figure(rfig, joinpath(dir, "residual_plot"))
            end
            global replotted += 1
            println("replotted sweep: $name$suffix")
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
            degen = hasproperty(df_map, :Degenerate) ? collect(Bool, df_map.Degenerate) :
                    falses(nrow(df_map))
            suffix = ""
            if rho_new === nothing
                X, Y = df_map.X_Bound, df_map.Y_Bound
                prior = hasproperty(df_map, :Prior_Limited) ? collect(Bool, df_map.Prior_Limited) :
                        falses(nrow(df_map))
            else
                # exact threshold rescale from the persisted per-direction K:
                # r_math = (16 ρ² / K)^{1/4}, re-capped at the stored prior box
                suffix = rho_tag(rho_new)
                hasproperty(df_map, :K_Raw) && hasproperty(df_map, :R_Box) ||
                    error("contour CSV lacks K_Raw/R_Box columns (pre-upgrade run) — " *
                          "rerun the pipeline to enable --rho replots")
                r_math = [K > 1e-300 ? (16.0 * rho_new^2 / K)^(1 / 4) : Inf for K in df_map.K_Raw]
                r_cap = min.(r_math, df_map.R_Box)
                if any(!isfinite, r_cap)
                    biggest = maximum(filter(isfinite, r_cap); init = 1.0)
                    r_cap[.!isfinite.(r_cap)] .= 5biggest
                end
                dc = hasproperty(df_map, :Dir_Cos) ? df_map.Dir_Cos : cos.(df_map.Angle)
                ds = hasproperty(df_map, :Dir_Sin) ? df_map.Dir_Sin : sin.(df_map.Angle)
                X = r_cap .* dc
                Y = r_cap .* ds
                prior = collect(isfinite.(df_map.R_Box) .& (r_math .>= df_map.R_Box))
                out = DataFrame(Angle = df_map.Angle, X_Bound = X, Y_Bound = Y,
                                Dir_Cos = dc, Dir_Sin = ds,
                                R_Capped = r_cap, R_Math = r_math, R_Box = df_map.R_Box,
                                Prior_Limited = prior, Degenerate = degen,
                                K_Raw = df_map.K_Raw,
                                G_uu = hasproperty(df_map, :G_uu) ? df_map.G_uu : fill(NaN, nrow(df_map)))
                CSV.write(backup_existing!(joinpath(dir, "confusion_contour$suffix.csv")), out)
            end
            fig = zone_figure(df_map.Angle, X, Y, prior;
                              px = px, py = py, box = box,
                              prior_frac = count(prior) / max(1, length(prior)),
                              degenerate_frac = count(degen) / max(1, length(degen)))
            save_figure(fig, joinpath(dir, "confusion_zone" * suffix))
            global replotted += 1
            println("replotted map: $name$suffix")
        catch err
            @warn "Failed to replot map '$name'" exception = (err, catch_backtrace())
        end
    end
end

println("Replot complete: $replotted item(s) regenerated in $RUN_DIR")
