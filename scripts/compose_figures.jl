# Assemble multi-panel publication figures from the persisted tables of one or
# more completed runs — no geometry or optimization is recomputed. A TOML
# layout lists the figures (name, kind, grid) and their panels (run directory
# and sweep or map name); all loading and drawing lives in src/RunFigures.jl
# and src/Plotting.jl. Each figure is written as a {pdf,png} pair with backup
# semantics.
#
#   julia scripts/compose_figures.jl <layout.toml> [--output-dir DIR]
#
# Run directories in the layout resolve relative to the layout file; the
# output directory defaults to plots/ under the project root. Example layout:
# configs/figures/quickstart_composites.toml.
const PROJECT_ROOT = dirname(@__DIR__)
include(joinpath(PROJECT_ROOT, "activate.jl"))

using CurvatureDistinguishability

const USAGE = "Usage: julia scripts/compose_figures.jl <layout.toml> [--output-dir DIR]"

function parse_arguments(argv)
    args = copy(argv)
    out_dir = joinpath(PROJECT_ROOT, "plots")
    if (idx = findfirst(==("--output-dir"), args)) !== nothing
        idx < length(args) || error("--output-dir requires a value\n" * USAGE)
        out_dir = abspath(args[idx+1])
        deleteat!(args, idx:(idx+1))
    end
    length(args) == 1 || error(USAGE)
    layout_path = abspath(args[1])
    isfile(layout_path) || error("Layout file not found: $layout_path\n" * USAGE)
    return layout_path, out_dir
end

function compose(layout_path, out_dir)
    mkpath(out_dir)
    figures = composite_figures(layout_path)
    for entry in figures
        save_figure(entry.figure, joinpath(out_dir, entry.name))
        println("composed: $(entry.name)")
    end
    return length(figures)
end

layout_path, out_dir = parse_arguments(ARGS)
n = compose(layout_path, out_dir)
println("Composition complete: $n figure(s) in $out_dir")
