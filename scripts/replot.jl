# Regenerate every figure of a completed run from its persisted CSVs —
# no geometry or optimization is recomputed. All rebuild logic lives in
# src/RunFigures.jl (shared with collect_plots.jl); this script only selects
# the run and writes {pdf,png} pairs back into it with backup semantics.
#
#   julia --project scripts/replot.jl <run_dir> [--rho R]
#
# With --rho R the discernibility threshold is changed WITHOUT recomputation:
# the boundary radius is r = (16ρ²/K)^{1/4} and K is persisted per direction,
# so maps (and the scaling-plot threshold markers) are rescaled exactly and
# written to *_rho<R> files, leaving the originals untouched.
using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)

using CSV
using TwoWaveformDistinguishability

function parse_arguments(argv)
    args = copy(argv)
    rho = nothing
    if (idx = findfirst(==("--rho"), args)) !== nothing
        idx < length(args) || error("--rho requires a value")
        rho = parse(Float64, args[idx+1])
        rho > 0 || error("--rho must be > 0")
        deleteat!(args, idx:idx+1)
    end
    length(args) == 1 ||
        error("Usage: julia --project scripts/replot.jl <run_dir> [--rho R]")
    run_dir = abspath(args[1])
    isdir(run_dir) || error("Run directory not found: $run_dir")
    isfile(joinpath(run_dir, "config.toml")) ||
        error("No config snapshot in $run_dir — cannot recover run metadata.")
    return run_dir, rho
end

function replot(run_dir, rho)
    replotted = 0
    cases = run_cases(run_dir)
    for name in cases.sweeps
        dir = joinpath(run_dir, "sweeps", name)
        try
            figs = sweep_figures(run_dir, name; rho = rho)
            save_figure(figs.scaling, joinpath(dir, "scaling_plot" * figs.suffix))
            figs.residual === nothing ||
                save_figure(figs.residual, joinpath(dir, "residual_plot"))
            replotted += 1
            println("replotted sweep: $name$(figs.suffix)")
        catch err
            @warn "Failed to replot sweep '$name'" exception = (err, catch_backtrace())
        end
    end
    for name in cases.maps
        dir = joinpath(run_dir, "maps", name)
        try
            rendered = zone_map_figure(run_dir, name; rho = rho)
            rendered.contour === nothing ||
                CSV.write(backup_existing!(joinpath(dir, "confusion_contour" * rendered.suffix * ".csv")),
                          rendered.contour)
            save_figure(rendered.figure, joinpath(dir, "confusion_zone" * rendered.suffix))
            replotted += 1
            println("replotted map: $name$(rendered.suffix)")
        catch err
            @warn "Failed to replot map '$name'" exception = (err, catch_backtrace())
        end
    end
    return replotted
end

run_dir, rho = parse_arguments(ARGS)
replotted = replot(run_dir, rho)
println("Replot complete: $replotted item(s) regenerated in $run_dir")
