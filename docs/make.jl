using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")); io = devnull)
Pkg.instantiate(; io = devnull)

using Documenter
using CurvatureDistinguishability

makedocs(
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
        "Roadmap" => "roadmap.md",
        "API Reference" => "api.md",
    ],
)
