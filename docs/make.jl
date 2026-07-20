using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(PackageSpec(path=joinpath(@__DIR__, "..")))

using Documenter
using TwoWaveformDistinguishability

makedocs(
    sitename = "TwoWaveformDistinguishability.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        assets = String[],
    ),
    modules = [TwoWaveformDistinguishability],
    warnonly = true,
    pages = [
        "Home" => "index.md",
        "System Architecture" => "architecture.md",
        "Scientific Context" => "science.md",
        "Waveform Physics" => "physics.md",
        "Complex Run Parameters" => "parameters.md",
        "API Reference" => "api.md"
    ]
)

# Documenter can also automatically deploy docs to GitHub pages.
# deploydocs(
#     repo = "github.com/USER/TwoWaveformDistinguishability.jl.git",
# )