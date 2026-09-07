using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)
using Dates
using CurvatureDistinguishability: effective_config, DEFAULT_CONFIG

const PIPELINE_SCRIPT = joinpath(@__DIR__, "run_pipeline.jl")
const LOG_DIR = joinpath(PROJECT_ROOT, "data", "logs")
mkpath(LOG_DIR)

"""
`--heap-size-hint` flag for the child process from `[hardware]
.heap_size_hint_gb` of the launched configuration (empty at the default 0 —
no hint). An explicit hint makes the host garbage collector fire before the
process nears system memory limits; on GPU campaigns earlier host
collections also release device buffers, whose host handles are too small
to trigger collection on their own.
"""
function heap_hint_flags(argv)
    config_rel = DEFAULT_CONFIG
    for (i, a) in enumerate(argv)
        a == "--config" && i < length(argv) && (config_rel = argv[i+1])
    end
    config_path = joinpath(PROJECT_ROOT, config_rel)
    isfile(config_path) || return String[]
    hw = get(effective_config(config_path), "hardware", Dict{String,Any}())
    gb = get(hw, "heap_size_hint_gb", 0.0)
    gb isa Real && gb > 0 || return String[]
    return gb >= 1 ? ["--heap-size-hint=$(round(Int, gb))G"] :
           ["--heap-size-hint=$(max(1, round(Int, 1024gb)))M"]
end

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
cmd = `$(Base.julia_cmd()) --project=$PROJECT_ROOT --threads=auto $(heap_hint_flags(ARGS)) $PIPELINE_SCRIPT $ARGS`
process = run(pipeline(cmd, stdout = console_log, stderr = console_log), wait = false)

println("  status       : RUNNING (PID $(getpid(process)))")
println("  monitor with : tail -f $console_log")
println("=" ^ 78)
