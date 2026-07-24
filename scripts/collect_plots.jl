# Collect every figure of one or more pipeline runs as PNGs into a single
# flat, human-browsable folder (plots/ by default), rendered
# fresh from the persisted CSVs with the current plotting code. The source
# run directories are not modified.
#
#   julia --project scripts/collect_plots.jl [dest_dir] [run_id ...]
#
# With no run_id arguments, every data/run_* directory is rendered;
# figure files are prefixed with the run id (run_<hash>_...).
using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)

using CSV, DataFrames, TOML, CairoMakie
using TwoWaveformDistinguishability
using TwoWaveformDistinguishability.Bounds: deviation_box
using TwoWaveformDistinguishability.Config: load_and_validate_config

const OUT = joinpath(PROJECT_ROOT, "data")
const DESTARG = !isempty(ARGS) && !startswith(ARGS[1], "run_")
const DEST = DESTARG ? abspath(ARGS[1]) : joinpath(PROJECT_ROOT, "plots")
const RUNSEL = filter(a -> startswith(a, "run_"), ARGS)
mkpath(DEST)

runs = isempty(RUNSEL) ?
    sort(filter(d -> startswith(d, "run_") && isdir(joinpath(OUT, d)), readdir(OUT))) :
    RUNSEL
const CAMPAIGN = [(run, run,
                   isdir(joinpath(OUT, run, "sweeps")) ? sort(readdir(joinpath(OUT, run, "sweeps"))) : String[],
                   isdir(joinpath(OUT, run, "maps")) ? sort(readdir(joinpath(OUT, run, "maps"))) : String[])
                  for run in runs]

png_at(name) = joinpath(DEST, name * ".png")
n = 0

function render_sweep(label, run, case)
    global n
    d = joinpath(OUT, run, "sweeps", case)
    isfile(joinpath(d, "results.csv")) || return
    cfg = load_and_validate_config(joinpath(OUT, run, "config.toml"))
    res = CSV.read(joinpath(d, "results.csv"), DataFrame)
    m = TOML.parsefile(joinpath(d, "sweep_meta.toml"))
    ratio = res.D2_Numerical ./ res.D2_Theoretical
    fl = Float64(get(m, "floor_level", -1.0)); fl < 0 && (fl = NaN)
    # display-time refit under the above-floor fit rule (the persisted
    # sweep_meta.toml keeps the campaign-era values untouched): fits use only
    # points strictly above the optimizer floor; convergence flags do not
    # exclude a point
    TWDO = TwoWaveformDistinguishability.Orchestrator
    clean = isnan(fl) ? (res.D2_Numerical .> 0) : (res.D2_Numerical .> fl)
    slope, slope_err = count(clean) >= 3 ?
        TWDO.loglog_slope(res.Delta[clean], res.D2_Numerical[clean]) : (NaN, NaN)
    c1, _, c2 = TWDO.ratio_correction_fit(res.Delta[clean], ratio[clean])
    fig = scaling_figure(res.Delta, res.D2_Numerical, res.D2_Theoretical;
                         rho_sq = Float64(get(m, "rho_sq", 1.0)),
                         delta_min = Float64(get(m, "delta_min", NaN)),
                         slope = slope, slope_err = slope_err,
                         clean = collect(clean), floor_level = fl,
                         c1 = c1, c2 = c2)
    save(png_at("$(label)_sweep_$(case)_scaling"), fig; px_per_unit = 4); n += 1
    sp = joinpath(d, "residual_spectrum.csv")
    if isfile(sp)
        spec = CSV.read(sp, DataFrame)
        rfig = residual_figure(spec, (delta_star = Float64(get(m, "delta_star", NaN)),
                                      df = Float64(get(m, "df", 1.0)),
                                      f_min = cfg.f_min, f_max = cfg.f_max,
                                      d2_num = Float64(get(m, "d2_num_star", NaN)),
                                      d2_theo = Float64(get(m, "d2_theo_star", NaN))))
        save(png_at("$(label)_sweep_$(case)_residual"), rfig; px_per_unit = 4); n += 1
    end
    println("  sweep: $label/$case")
end

function render_map(label, run, case)
    global n
    csv = joinpath(OUT, run, "maps", case, "confusion_contour.csv")
    isfile(csv) || return
    cfg = load_and_validate_config(joinpath(OUT, run, "config.toml"))
    mm = only(filter(m -> String(m["name"]) == case, cfg.maps))
    px, py = Int(mm["param_x"]), Int(mm["param_y"])
    t0 = Float64.(mm["theta_0"])
    box = deviation_box(cfg.bounds, t0, px, py)
    df = CSV.read(csv, DataFrame)
    prior = collect(Bool, df.Prior_Limited)
    degen = collect(Bool, df.Degenerate)
    fig = zone_figure(df.Angle, df.X_Bound, df.Y_Bound, prior;
                      px = px, py = py, box = box,
                      prior_frac = count(prior) / nrow(df),
                      degenerate_frac = count(degen) / nrow(df),
                      x_math = df.R_Math .* df.Dir_Cos, y_math = df.R_Math .* df.Dir_Sin)
    save(png_at("$(label)_map_$(case)"), fig; px_per_unit = 4); n += 1
    println("  map:   $label/$case")
end

for (label, run, sweeps, maps) in CAMPAIGN
    for s in sweeps; render_sweep(label, run, s); end
    for mp in maps; render_map(label, run, mp); end
end

println("Collected $n PNG figures into $DEST")
