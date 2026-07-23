# Collect the campaign figures as PNGs into a single flat, human-browsable
# folder (data/regenerated-plots), rendered fresh from the persisted CSVs with
# the current plotting code. Descriptive filenames; the source run directories
# are not modified.
#
# By default only the CPU set is rendered — the GPU sweeps reproduce the CPU
# figures to visual identity (slopes agree to 4 decimals; cross-validated at
# the 1e-8 level), so duplicating them just doubles the inspection load.
# Pass --all to also render the gpu_oneapi_* counterparts.
#
#   julia --project scripts/collect_plots.jl [dest_dir] [--all]
#
using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)

using CSV, DataFrames, TOML, CairoMakie
using TwoWaveformDistinguishability
using TwoWaveformDistinguishability.Bounds: deviation_box
using TwoWaveformDistinguishability.Config: load_and_validate_config

const OUT = joinpath(PROJECT_ROOT, "data", "outputs")
const INCLUDE_GPU = "--all" in ARGS
const POSARGS = filter(a -> a != "--all", ARGS)
const DEST = length(POSARGS) >= 1 ? abspath(POSARGS[1]) : joinpath(PROJECT_ROOT, "data", "regenerated-plots")
mkpath(DEST)

# (label, run_dir, sweep_case_names, map_case_names)
const CAMPAIGN = [
    ("cpu", "run_a2bec346", ["six_dimensional_diagonal_stress"], String[]),
    ("cpu", "run_0d3a0b6b",
        ["massive_binary_mass_time", "unequal_amplitude_mass_time",
         "extreme_spin_orbit_coupling", "low_mass_time_shift"],
        ["mass_vs_time_degeneracy", "spin1_vs_spin2_coupling", "mass_vs_spin1_twist",
         "time_vs_phase_doppler", "mass_vs_phase_low_mass"]),
    ("gpu_oneapi", "run_0dbc0494",
        ["six_dimensional_diagonal_stress", "massive_binary_mass_time",
         "unequal_amplitude_mass_time", "extreme_spin_orbit_coupling"], String[]),
    ("gpu_oneapi", "run_110359a2", ["low_mass_time_shift"], String[]),
]

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
    conv = hasproperty(res, :Converged) ? res.Converged : trues(nrow(res))
    clean = (ratio .> 0.5) .& (ratio .< 2.0) .& conv
    fl = Float64(get(m, "floor_level", -1.0)); fl < 0 && (fl = NaN)
    fig = scaling_figure(res.Delta, res.D2_Numerical, res.D2_Theoretical;
                         rho_sq = Float64(get(m, "rho_sq", 1.0)),
                         delta_min = Float64(get(m, "delta_min", NaN)),
                         slope = Float64(get(m, "slope", NaN)),
                         slope_err = Float64(get(m, "slope_err", NaN)),
                         clean = collect(clean), floor_level = fl,
                         c1 = Float64(get(m, "c1", NaN)), c2 = Float64(get(m, "c2", NaN)))
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
    label != "cpu" && !INCLUDE_GPU && continue
    for s in sweeps; render_sweep(label, run, s); end
    for mp in maps; render_map(label, run, mp); end
end

println("Collected $n PNG figures into $DEST")
