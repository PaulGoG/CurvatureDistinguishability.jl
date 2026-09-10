# Activates and instantiates the documentation environment silently; the
# package resolves by path via [sources]. Included by make.jl; for an
# interactive session:
#
#   julia -i docs/activate.jl
#
using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)
