# Collect every figure of one or more pipeline runs as PNGs into a single
# flat, human-browsable folder (plots/ by default), rendered fresh from the
# persisted CSVs with the current plotting code: slope and correction
# coefficients are refitted for display, persisted run metadata is never
# modified, and the source run directories are not touched. All rebuild
# logic lives in src/RunFigures.jl (shared with replot.jl).
#
#   julia --project scripts/collect_plots.jl [dest_dir] [run ...]
#
# A run selector is either a run id under data/ (run_<hash>) or a path to a
# run directory (e.g. one produced with --output-dir elsewhere); selectors
# that resolve to run directories take precedence over the destination
# argument. With no selectors, every data/run_* directory is rendered.
# Figure files are prefixed with the run id (run_<hash>_...).
using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)

using CairoMakie
using CurvatureDistinguishability

const DATA_ROOT = joinpath(PROJECT_ROOT, "data")

is_run_dir(path) = isdir(path) && isfile(joinpath(path, "config.toml"))

function resolve_run(arg)
    is_run_dir(abspath(arg)) && return abspath(arg)
    startswith(basename(arg), "run_") && return joinpath(DATA_ROOT, arg)
    return nothing
end

function parse_arguments(argv)
    runs = String[]
    dest = nothing
    for arg in argv
        resolved = resolve_run(arg)
        if resolved !== nothing
            push!(runs, resolved)
        elseif dest === nothing
            dest = abspath(arg)
        else
            error("Unrecognized argument '$arg' (not a run id, a run directory, " *
                  "or the single destination directory).")
        end
    end
    if isempty(runs)
        runs = isdir(DATA_ROOT) ?
            [joinpath(DATA_ROOT, d) for d in sort(readdir(DATA_ROOT))
             if startswith(d, "run_") && is_run_dir(joinpath(DATA_ROOT, d))] : String[]
    end
    return runs, something(dest, joinpath(PROJECT_ROOT, "plots"))
end

function collect_figures(runs, dest)
    n = 0
    for run_dir in runs
        label = basename(run_dir)
        cases = run_cases(run_dir)
        for case in cases.sweeps
            figs = sweep_figures(run_dir, case; refit = true)
            save(joinpath(dest, "$(label)_sweep_$(case)_scaling.png"), figs.scaling;
                 px_per_unit = 4); n += 1
            if figs.residual !== nothing
                save(joinpath(dest, "$(label)_sweep_$(case)_residual.png"), figs.residual;
                     px_per_unit = 4); n += 1
            end
            println("  sweep: $label/$case")
        end
        for case in cases.maps
            rendered = zone_map_figure(run_dir, case)
            save(joinpath(dest, "$(label)_map_$(case).png"), rendered.figure;
                 px_per_unit = 4); n += 1
            println("  map:   $label/$case")
        end
    end
    return n
end

runs, dest = parse_arguments(ARGS)
mkpath(dest)
n = collect_figures(runs, dest)
println("Collected $n PNG figures into $dest")
