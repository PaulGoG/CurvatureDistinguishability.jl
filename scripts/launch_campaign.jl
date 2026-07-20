using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io=devnull)
using Dates

const PIPELINE_SCRIPT = joinpath(@__DIR__, "pipeline.jl")
const LOG_DIR = joinpath(PROJECT_ROOT, "data", "logs")

mkpath(LOG_DIR)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
log_file = joinpath(LOG_DIR, "campaign_$timestamp.log")

println("================================================================================")
println("  🚀 Launching Two-Waveform Distinguishability Campaign")
println("================================================================================")
println("  ▶ Project Root : $PROJECT_ROOT")
println("  ▶ Log Vault    : $log_file")

# Spawn the pipeline process asynchronously using Julia's native run()
cmd = `julia --threads auto $PIPELINE_SCRIPT`
process = run(pipeline(cmd, stdout=log_file, stderr=log_file), wait=false)

println("  ▶ Status       : RUNNING (PID: $(getpid(process)))")
println("================================================================================")
println("To monitor the live progress, run:")
println("tail -f $log_file")
flush(stdout)
