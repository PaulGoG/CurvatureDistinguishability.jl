using Pkg
# Dynamically resolve project root relative to this script
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io=devnull)

ENV["GKSwstype"] = "100" # Headless mode for Plots.jl to prevent windows from popping up

using Distributed
using ArgParse

function parse_commandline()
    s = ArgParseSettings(description="Two-Waveform Distinguishability Unified Pipeline")
    @add_arg_table! s begin
        "--config"
            help = "Path to a TOML configuration file"
            default = "config.toml"
        "--output-dir"
            help = "Directory to save outputs (relative to project root)"
            default = "data/outputs"
        "--workers"
            help = "Number of distributed worker processes to spawn (0 for local multi-threading only)"
            arg_type = Int
            default = 0
    end
    return parse_args(s)
end

args = parse_commandline()

if args["workers"] > 0
    println("Spawning $(args["workers"]) distributed worker processes...")
    addprocs(args["workers"]; exeflags="--project=$PROJECT_ROOT")
end

@everywhere begin
    using Pkg
    Pkg.activate($PROJECT_ROOT; io=devnull)
    
    # Attempt to load GPU packages if available to enable Hardware.jl detection
    try using CUDA catch end
    try using AMDGPU catch end
    try using Metal catch end
    try using oneAPI catch end
    
    using TwoWaveformDistinguishability
end

# -----------------------------------------------------------------------------
# MAIN ORCHESTRATOR ENTRY POINT
# -----------------------------------------------------------------------------
function main()
    # The entire pipeline logic has been modularized and shifted into src/Orchestrator.jl
    # for strict adherence to Julian software engineering best practices.
    config_path = joinpath(PROJECT_ROOT, args["config"])
    run_pipeline(config_path, PROJECT_ROOT, args["output-dir"])
end

main()