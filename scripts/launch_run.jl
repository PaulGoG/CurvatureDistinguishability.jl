# Launch supervised runs detached from the terminal: starts
# scripts/run_supervised.jl in its own process group, with its console output in
# data/logs/, and returns at once. Arguments are those of run_supervised.jl.
#
#   julia scripts/launch_run.jl [--output-dir DIR] [--threads SPEC] [--fresh] <config.toml> ...
const PROJECT_ROOT = dirname(@__DIR__)
using Dates

const SUPERVISOR_SCRIPT = joinpath(@__DIR__, "run_supervised.jl")

isempty(ARGS) && error("no configuration given; arguments are those of run_supervised.jl")
output_dir = "data"
for (i, arg) in enumerate(ARGS)
    arg == "--output-dir" && i < length(ARGS) && (global output_dir = ARGS[i+1])
end
log_dir = joinpath(PROJECT_ROOT, output_dir, "logs")
mkpath(log_dir)
console_log = joinpath(log_dir, "launch_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")

# the child activates its own environment, survives the launching shell, and
# writes both streams through one handle so neither overwrites the other
cmd = `$(Base.julia_cmd().exec[1]) --startup-file=no $SUPERVISOR_SCRIPT $ARGS`
process = open(console_log, "a") do io
    run(pipeline(detach(cmd); stdout = io, stderr = io); wait = false)
end

println("Supervisor started (PID $(getpid(process)))")
println("  console log : $console_log")
println("  run state   : <run directory>/supervision.toml, run.log, heartbeat.toml")
