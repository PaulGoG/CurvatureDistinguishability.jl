using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.develop(PackageSpec(path = joinpath(@__DIR__, "..")); io = devnull)
Pkg.instantiate(; io = devnull)

using Documenter
using TwoWaveformDistinguishability

makedocs(
    sitename = "TwoWaveformDistinguishability.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
    ),
    modules = [TwoWaveformDistinguishability],
    checkdocs = :exports,
    warnonly = false,
    pages = [
        "Home" => "index.md",
        "System Architecture" => "architecture.md",
        "Scientific Context" => "science.md",
        "Waveform Physics" => "physics.md",
        "Complex Run Parameters" => "parameters.md",
        "Roadmap" => "roadmap.md",
        "API Reference" => "api.md"
    ]
)
