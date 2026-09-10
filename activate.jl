# Activates and instantiates the package environment silently. Every entry
# point under scripts/ includes this file; for an interactive session:
#
#   julia -i activate.jl
#
using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)
