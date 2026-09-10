# Activates and instantiates the benchmark environment silently; the package
# resolves by path via [sources]. Included by run_benchmarks.jl; for an
# interactive session:
#
#   julia -i bench/activate.jl
#
using Pkg
Pkg.activate(@__DIR__; io = devnull)
Pkg.instantiate(; io = devnull)
