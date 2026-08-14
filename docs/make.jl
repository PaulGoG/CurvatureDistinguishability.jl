using Pkg
Pkg.activate(@__DIR__; io = devnull)
# the package resolves by path via [sources] (unregistered dependency)
Pkg.instantiate(; io = devnull)

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
        # directory-style URLs break file:// browsing of the CI artifact and
        # of local builds; restore the CI-conditional form when the site is
        # deployed to GitHub Pages (at repository publication)
        prettyurls = false,
        assets = String[],
    ),
    modules = [CurvatureDistinguishability],
    checkdocs = :exports,
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
