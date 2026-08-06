using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)

using TOML

const USAGE = """
Curvature-Distinguishability Unified Pipeline

    julia --project scripts/run_pipeline.jl [--config PATH] [--output-dir DIR]

    --config PATH       TOML configuration, relative to the project root
                        (shipped scenarios live in configs/; the default is
                        the minutes-scale configs/quickstart.toml —
                        production campaigns are selected explicitly, e.g.
                        configs/production_cpu.toml)
    --output-dir DIR    output directory relative to the project root
                        (default: data)
"""

function parse_commandline(argv)
    options = Dict("config" => joinpath("configs", "quickstart.toml"),
        "output-dir" => "data")
    i = 1
    while i <= length(argv)
        arg = argv[i]
        if arg in ("-h", "--help")
            print(USAGE)
            exit(0)
        elseif arg in ("--config", "--output-dir")
            i < length(argv) || error("$arg requires a value\n$USAGE")
            options[arg[3:end]] = argv[i+1]
            i += 2
        else
            error("Unknown argument '$arg'\n$USAGE")
        end
    end
    return options
end

args = parse_commandline(ARGS)
config_path = joinpath(PROJECT_ROOT, args["config"])
isfile(config_path) || error("Configuration file not found: $config_path")

# Load a GPU package only when the configuration asks for one AND it is
# installed in this environment — no blind try/catch, loud diagnostics.
const GPU_PACKAGES = Dict("cuda" => "CUDA", "amdgpu" => "AMDGPU",
    "metal" => "Metal", "oneapi" => "oneAPI")
let hw = get(TOML.parsefile(config_path), "hardware", Dict{String,Any}())
    requested = lowercase(String(get(hw, "gpu_backend", "auto")))
    wanted =
        requested == "auto" ? collect(keys(GPU_PACKAGES)) :
        haskey(GPU_PACKAGES, requested) ? [requested] : String[]
    if lowercase(String(get(hw, "gpu_backend", "auto"))) != "none"
        for key in wanted
            pkgname = GPU_PACKAGES[key]
            if Base.find_package(pkgname) === nothing
                requested == key &&
                    @warn "Requested GPU backend '$key' but package $pkgname is not installed " *
                          "in this environment. Install it with: julia --project -e " *
                          "'using Pkg; Pkg.add(\"$pkgname\")'"
                continue
            end
            @info "Loading GPU package $pkgname (activates the $pkgname extension)…"
            Base.require(Main, Symbol(pkgname))
        end
    end
end

using CurvatureDistinguishability

run_pipeline(config_path, PROJECT_ROOT, args["output-dir"])
