# Activates the test environment interactively: the sandbox Pkg.test builds
# (the package by path plus test/Project.toml), reproduced by TestEnv.jl.
# TestEnv lives in the shared "testenv" environment, installed on first use
# and appended to the load path.
#
#   julia -i test/activate.jl
#
using Pkg
const TESTENV_SHARED = "testenv"
if Base.find_package("TestEnv") === nothing
    push!(LOAD_PATH, "@" * TESTENV_SHARED)
    if Base.find_package("TestEnv") === nothing
        Pkg.activate(TESTENV_SHARED; shared = true, io = devnull)
        Pkg.add("TestEnv"; io = devnull)
    end
end
Pkg.activate(dirname(@__DIR__); io = devnull)
Pkg.instantiate(; io = devnull)
using TestEnv
TestEnv.activate()
