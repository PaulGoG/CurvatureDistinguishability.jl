# Run one or more configurations under the process supervisor: each worker
# (scripts/run_pipeline.jl) is watched through its heartbeat and CPU time,
# terminated when it hangs and relaunched into the same run directory within the
# retry budget of the configuration's [supervision] table. This process never
# loads a GPU package, so a lost device cannot stall it.
#
#   julia scripts/run_supervised.jl [--output-dir DIR] [--threads SPEC] [--fresh] <config.toml> ...
#
# Configurations run in the order given; re-running the same command continues
# whatever is unfinished and skips what is complete. Exit status: 0 every run
# complete; 2 configuration error; 3 otherwise (see supervision.toml in the run
# directories).
const PROJECT_ROOT = dirname(@__DIR__)
include(joinpath(PROJECT_ROOT, "activate.jl"))

using CurvatureDistinguishability
using Dates
using Logging
using LoggingExtras: FormatLogger, MinLevelLogger, TeeLogger

const WORKER_SCRIPT = joinpath(@__DIR__, "run_pipeline.jl")
const USAGE =
    "Usage: julia scripts/run_supervised.jl [--output-dir DIR] " *
    "[--threads SPEC] [--fresh] <config.toml> ..."

function parse_arguments(argv)
    output_dir, threads, fresh = "data", "auto", false
    configs = String[]
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg in ("-h", "--help")
            println(USAGE)
            exit(0)
        elseif arg == "--fresh"
            fresh = true
            i += 1
        elseif arg in ("--output-dir", "--threads")
            i < length(argv) || error("$arg requires a value\n$USAGE")
            arg == "--output-dir" ? (output_dir = argv[i+1]) : (threads = argv[i+1])
            i += 2
        elseif startswith(arg, "--")
            error("Unknown argument '$arg'\n$USAGE")
        else
            push!(configs, joinpath(PROJECT_ROOT, arg))
            i += 1
        end
    end
    isempty(configs) && error("no configuration given\n$USAGE")
    return (; output_dir, threads, fresh, configs)
end

function main(argv)
    args = parse_arguments(argv)
    log_dir = joinpath(PROJECT_ROOT, args.output_dir, "logs")
    mkpath(log_dir)
    log_path =
        joinpath(log_dir, "supervisor_$(Dates.format(now(), "yyyymmdd_HHMMSS")).log")
    statuses = Pair{String,Symbol}[]
    open(log_path, "a") do io
        file_logger = FormatLogger(io) do stream, record
            println(stream, "[", Dates.format(now(), "yyyy-mm-dd HH:MM:SS"), "] ",
                uppercase(string(record.level)), " ", record.message)
            flush(stream)
        end
        with_logger(
            TeeLogger(global_logger(), MinLevelLogger(file_logger, Logging.Info)),
        ) do
            for config in args.configs
                @info "Supervising $config"
                status = try
                    run_dir, result = supervise_pipeline(config, PROJECT_ROOT,
                        args.output_dir; worker_script = WORKER_SCRIPT,
                        threads = args.threads, fresh = args.fresh)
                    @info "$(basename(run_dir)): $(result.status) after " *
                          "$(length(result.attempts)) attempt(s)" *
                          (
                              isempty(result.abandoned) ? "" :
                              "; abandoned: $(join(result.abandoned, ", "))"
                          )
                    result.status
                catch err
                    err isa ArgumentError || rethrow()
                    @error "Configuration error in $config" exception = err
                    :config_error
                end
                push!(statuses, config => status)
                # a device that survived SIGKILL takes no further work
                status === :wedged && break
            end
        end
    end
    println("Supervisor log: $log_path")
    foreach(((config, status),) -> println(rpad(String(status), 18), config), statuses)
    all(==(:complete), last.(statuses)) && length(statuses) == length(args.configs) &&
        return 0
    return any(==(:config_error), last.(statuses)) ? 2 : 3
end

exit(main(ARGS))
