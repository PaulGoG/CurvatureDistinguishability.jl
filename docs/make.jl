include(joinpath(@__DIR__, "activate.jl"))

using Documenter
using DocumenterCitations
using Literate
using CurvatureDistinguishability

Literate.markdown(joinpath(@__DIR__, "src", "literate", "quartic_law.jl"),
    joinpath(@__DIR__, "src", "generated"); documenter = true)

DocMeta.setdocmeta!(CurvatureDistinguishability, :DocTestSetup,
    :(using CurvatureDistinguishability); recursive = true)

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"); style = :numeric)

makedocs(
    plugins = [bib],
    sitename = "CurvatureDistinguishability.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
    ),
    modules = [CurvatureDistinguishability],
    checkdocs = :public,
    warnonly = false,
    pages = [
        "Home" => "index.md",
        "System Architecture" => "architecture.md",
        "Scientific Context" => "science.md",
        "Waveform Physics" => "physics.md",
        "Complex Run Parameters" => "parameters.md",
        "Executable Example" => "generated/quartic_law.md",
        "Roadmap" => "roadmap.md",
        "API Reference" => "api.md",
        "References" => "references.md",
    ],
)
