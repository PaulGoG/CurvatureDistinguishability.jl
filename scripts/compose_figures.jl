# Assemble multi-panel publication figures from the persisted tables of one or
# more completed runs — no geometry or optimization is recomputed. A TOML
# layout lists the figures (name, kind, grid) and their panels (run directory
# and sweep or map name); all loading and drawing lives in src/RunFigures.jl
# and src/Plotting.jl. Each figure is written as a {pdf,png} pair with backup
# semantics.
#
#   julia scripts/compose_figures.jl <layout.toml> [--output-dir DIR] [--print-width PT]
#
# Run directories in the layout resolve relative to the layout file; the
# output directory defaults to plots/ under the project root. With
# --print-width every PDF is scaled to PT points of width (the manuscript's
# text width), so it enters the document at native size. Example layout:
# configs/figures/quickstart_composites.toml.
const PROJECT_ROOT = dirname(@__DIR__)
include(joinpath(PROJECT_ROOT, "activate.jl"))

using CurvatureDistinguishability

const USAGE = "Usage: julia scripts/compose_figures.jl <layout.toml> [--output-dir DIR] [--print-width PT]"

function parse_arguments(argv)
    args = copy(argv)
    out_dir = joinpath(PROJECT_ROOT, "plots")
    print_width = nothing
    if (idx = findfirst(==("--output-dir"), args)) !== nothing
        idx < length(args) || error("--output-dir requires a value\n" * USAGE)
        out_dir = abspath(args[idx+1])
        deleteat!(args, idx:(idx+1))
    end
    if (idx = findfirst(==("--print-width"), args)) !== nothing
        idx < length(args) || error("--print-width requires a value\n" * USAGE)
        print_width = tryparse(Float64, args[idx+1])
        (print_width === nothing || print_width <= 0) &&
            error("--print-width must be a positive number of points\n" * USAGE)
        deleteat!(args, idx:(idx+1))
    end
    length(args) == 1 || error(USAGE)
    layout_path = abspath(args[1])
    isfile(layout_path) || error("Layout file not found: $layout_path\n" * USAGE)
    return layout_path, out_dir, print_width
end

# PDF scale of `figure`: `pt_per_unit` such that its canvas prints `print_width`
# points wide; the default scale when no print width is requested
figure_scale(figure, ::Nothing) = 0.75
figure_scale(figure, print_width::Real) = print_width / canvas_width(figure)

function compose(layout_path, out_dir, print_width)
    mkpath(out_dir)
    figures = composite_figures(layout_path)
    for entry in figures
        save_figure(entry.figure, joinpath(out_dir, entry.name);
            pt_per_unit = figure_scale(entry.figure, print_width))
        println("composed: $(entry.name)")
    end
    return length(figures)
end

layout_path, out_dir, print_width = parse_arguments(ARGS)
n = compose(layout_path, out_dir, print_width)
println("Composition complete: $n figure(s) in $out_dir")
