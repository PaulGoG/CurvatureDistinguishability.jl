const PROJECT_ROOT = dirname(@__DIR__)
include(joinpath(PROJECT_ROOT, "activate.jl"))

using CurvatureDistinguishability

const USAGE = """
Curvature-Distinguishability Unified Pipeline

    julia --threads=auto scripts/run_pipeline.jl [--config PATH] [--output-dir DIR]
                                                 [--resume] [--run-dir DIR]
                                                 [--heartbeat-period SECONDS]

    --config PATH       TOML configuration, relative to the project root
                        (shipped scenarios live in configs/; the default is
                        the minutes-scale configs/quickstart.toml —
                        production campaigns are selected explicitly, e.g.
                        configs/production_cpu.toml)
    --output-dir DIR    output directory relative to the project root
                        (default: data)
    --resume            continue the unfinished run of this configuration
                        instead of starting a new run_<id>_rN sibling:
                        complete stages and checkpointed work items are
                        skipped
    --run-dir DIR       use exactly this run directory (implies --resume);
                        set by scripts/run_supervised.jl
    --heartbeat-period SECONDS
                        publish heartbeat.toml in the run directory at this
                        period (default 0: no heartbeat)

Exit status: 0 every stage completed; 2 configuration or resource-budget
error (nothing was computed); 3 one or more stages failed (the others ran;
see failed_stages in metadata.toml); 1 any other error.
"""

const EXIT_CONFIG_ERROR = 2
const EXIT_STAGE_FAILURE = 3

function parse_commandline(argv)
    options = Dict{String,Any}("config" => CurvatureDistinguishability.DEFAULT_CONFIG,
        "output-dir" => "data", "resume" => false, "run-dir" => nothing,
        "heartbeat-period" => "0")
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg in ("-h", "--help")
            print(USAGE)
            exit(0)
        elseif arg == "--resume"
            options["resume"] = true
            i += 1
        elseif arg in ("--config", "--output-dir", "--run-dir", "--heartbeat-period")
            i < length(argv) || error("$arg requires a value\n$USAGE")
            options[arg[3:end]] = argv[i+1]
            i += 2
        else
            error("Unknown argument '$arg'\n$USAGE")
        end
    end
    return options
end

"""
Report a configuration or resource-budget error and end with `EXIT_CONFIG_ERROR`;
any other exception propagates.
"""
function exit_on_config_error(err)
    err isa Union{ArgumentError,CurvatureDistinguishability.ResourceBudgetError} || return
    showerror(stderr, err)
    println(stderr)
    exit(EXIT_CONFIG_ERROR)
end

args = parse_commandline(ARGS)
config_path = joinpath(PROJECT_ROOT, args["config"])
# validated before any GPU package loads, so an unusable file costs seconds
try
    load_and_validate_config(config_path)
catch err
    exit_on_config_error(err)
    rethrow()
end

# Load a GPU package only when the configuration asks for one AND it is
# installed in this environment — no blind try/catch, loud diagnostics. The
# package extension activates whichever of the two is loaded last, so the
# GPU package may follow the package itself; the [hardware] table is read
# from the effective configuration (an overlay merged onto its base).
const GPU_PACKAGES = Dict("cuda" => "CUDA", "amdgpu" => "AMDGPU",
    "metal" => "Metal", "oneapi" => "oneAPI")
let hw = get(effective_config(config_path), "hardware", Dict{String,Any}())
    requested = lowercase(String(get(hw, "gpu_backend", "auto")))
    wanted =
        requested == "auto" ? collect(keys(GPU_PACKAGES)) :
        haskey(GPU_PACKAGES, requested) ? [requested] : String[]
    if requested != "none"
        for key in wanted
            pkgname = GPU_PACKAGES[key]
            if Base.find_package(pkgname) === nothing
                requested == key &&
                    @warn "Requested GPU backend '$key' but package $pkgname is not installed " *
                          "in the default (stacked) environment. Install it with: julia -e " *
                          "'using Pkg; Pkg.add(\"$pkgname\")'"
                continue
            end
            @info "Loading GPU package $pkgname (activates the $pkgname extension)…"
            Base.require(Main, Symbol(pkgname))
        end
    end
end

heartbeat_period = something(tryparse(Float64, args["heartbeat-period"]), -1.0)
heartbeat_period >= 0 || error("--heartbeat-period must be a number >= 0\n$USAGE")
run_dir = args["run-dir"] === nothing ? nothing : joinpath(PROJECT_ROOT, args["run-dir"])

try
    run_pipeline(config_path, PROJECT_ROOT, args["output-dir"]; resume = args["resume"],
        run_dir = run_dir, heartbeat_period_s = heartbeat_period)
catch err
    # stage failures are already in run.log with their backtraces
    err isa CurvatureDistinguishability.PipelineStageError && exit(EXIT_STAGE_FAILURE)
    exit_on_config_error(err)
    rethrow()
end
