using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io = devnull)
Pkg.instantiate(; io = devnull)

using ArgParse
using TOML

function parse_commandline()
    s = ArgParseSettings(description = "Two-Waveform Distinguishability Unified Pipeline")
    @add_arg_table! s begin
        "--config"
            help = "Path to a TOML configuration file (relative to project root)"
            default = "config.toml"
        "--output-dir"
            help = "Directory to save outputs (relative to project root)"
            default = "data/outputs"
    end
    return parse_args(s)
end

args = parse_commandline()
config_path = joinpath(PROJECT_ROOT, args["config"])
isfile(config_path) || error("Configuration file not found: $config_path")

# Load a GPU package only when the configuration asks for one AND it is
# installed in this environment — no blind try/catch, loud diagnostics.
const GPU_PACKAGES = Dict("cuda" => "CUDA", "amdgpu" => "AMDGPU",
                          "metal" => "Metal", "oneapi" => "oneAPI")
let hw = get(TOML.parsefile(config_path), "hardware", Dict{String,Any}())
    requested = lowercase(String(get(hw, "gpu_backend", "auto")))
    wanted = requested == "auto" ? collect(keys(GPU_PACKAGES)) :
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

using TwoWaveformDistinguishability

run_pipeline(config_path, PROJECT_ROOT, args["output-dir"])
