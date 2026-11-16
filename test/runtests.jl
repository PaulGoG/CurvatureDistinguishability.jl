using CurvatureDistinguishability
using Test
using Aqua
using ExplicitImports
using JET
using CSV
using DataFrames
using ForwardDiff
using KernelAbstractions
using LinearAlgebra: I, istriu, norm, svdvals
using Random: Xoshiro
using TOML

include("common.jl")

# test groups in the order they run; `Pkg.test(test_args = ["physics"])` (or
# `julia test/runtests.jl physics geometry` in the test environment) runs a
# subset
const TEST_GROUPS = (
    "qa",
    "physics",
    "geometry",
    "inference",
    "config",
    "fitting_plotting",
    "orchestrator",
    "e2e",
    "supervision",
)
const SELECTED_GROUPS = isempty(ARGS) ? TEST_GROUPS : Tuple(ARGS)
for group in SELECTED_GROUPS
    group in TEST_GROUPS ||
        error("unknown test group '$group'; groups: $(join(TEST_GROUPS, ", "))")
end

@testset "CurvatureDistinguishability.jl" begin
    for group in TEST_GROUPS
        group in SELECTED_GROUPS || continue
        include(group == "supervision" ? "supervision_tests.jl" : "$group.jl")
    end
end
