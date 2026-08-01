using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)
using Dates

const PIPELINE_SCRIPT = joinpath(@__DIR__, "run_pipeline.jl")
const LOG_DIR = joinpath(PROJECT_ROOT, "data", "logs")
mkpath(LOG_DIR)

timestamp = Dates.format(now(), "yyyymmdd_HHMMSS")
console_log = joinpath(LOG_DIR, "run_$(timestamp).log")

println("=" ^ 78)
println("  Launching CurvatureDistinguishability pipeline run (detached)")
println("=" ^ 78)
println("  project root : $PROJECT_ROOT")
println("  console log  : $console_log")
println("  (progress bars are TTY-gated: the detached log stays ANSI-free;")
println("   the run directory additionally receives a structured run.log)")

# Same Julia binary and flags as this session; detached child process.
cmd = `$(Base.julia_cmd()) --project=$PROJECT_ROOT --threads=auto $PIPELINE_SCRIPT $ARGS`
process = run(pipeline(cmd, stdout = console_log, stderr = console_log), wait = false)

println("  status       : RUNNING (PID $(getpid(process)))")
println("  monitor with : tail -f $console_log")
println("=" ^ 78)
