# Audit a completed run's confusion maps against the physical parameter
# bounds: per-axis extents, bound compliance of the capped polygon, and
# prior-limited / degenerate direction statistics.
#
#   julia --project scripts/audit_bounds.jl <run_dir>
#
using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)

using CSV
using DataFrames
using Printf
using TwoWaveformDistinguishability
using TwoWaveformDistinguishability.Bounds: deviation_box, PARAM_KEYS
using TwoWaveformDistinguishability.Config: load_and_validate_config

length(ARGS) == 1 || error("Usage: julia --project scripts/audit_bounds.jl <run_dir>")
const RUN_DIR = abspath(ARGS[1])
config_snapshot = joinpath(RUN_DIR, "config.toml")
isfile(config_snapshot) || error("No config snapshot in $RUN_DIR")
cfg = load_and_validate_config(config_snapshot)
map_meta = Dict(String(m["name"]) => m for m in cfg.maps)

maps_dir = joinpath(RUN_DIR, "maps")
isdir(maps_dir) || error("No maps/ directory under $RUN_DIR")

violations = 0
for name in sort(readdir(maps_dir))
    global violations
    csv_path = joinpath(maps_dir, name, "confusion_contour.csv")
    (isfile(csv_path) && haskey(map_meta, name)) || continue
    m = map_meta[name]
    px, py = Int(m["param_x"]), Int(m["param_y"])
    theta0 = Float64.(m["theta_0"])
    box = deviation_box(cfg.bounds, theta0, px, py)
    df = CSV.read(csv_path, DataFrame)

    println("=== $name  ($(PARAM_KEYS[px]) vs $(PARAM_KEYS[py])) ===")
    tol = 1e-9
    for (axis, vals, lo, hi) in (("x", df.X_Bound, box[1], box[2]),
                                 ("y", df.Y_Bound, box[3], box[4]))
        lo_v, hi_v = extrema(vals)
        ok = lo_v >= lo - tol && hi_v <= hi + tol
        ok || (violations += 1)
        @printf("  %s: data [%.4g, %.4g]  box [%.4g, %.4g]  %s\n",
                axis, lo_v, hi_v, lo, hi, ok ? "OK" : "VIOLATION")
    end
    if hasproperty(df, :Prior_Limited)
        @printf("  prior-limited: %.1f%%   degenerate: %.1f%%   directions: %d\n",
                100 * count(df.Prior_Limited) / nrow(df),
                hasproperty(df, :Degenerate) ? 100 * count(df.Degenerate) / nrow(df) : 0.0,
                nrow(df))
    end
end

if violations == 0
    println("\nAll capped contours respect the physical bounds.")
else
    println("\n$violations axis violation(s) found — the capping stage is broken; inspect the run.")
    exit(1)
end
