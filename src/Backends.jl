"""
Compute-backend registry: the multi-threaded CPU fallback plus GPU probes
registered by the package extensions.
"""
module Backends

using DocStringExtensions: TYPEDSIGNATURES
using KernelAbstractions: KernelAbstractions, CPU

export get_best_backend, to_backend, backend_name
public register_backend!, reclaim_device_memory!, cpu_model, device_fingerprint

"""
Registry of GPU backend probes, populated by the package extensions
(`CurvatureDistinguishability{CUDA,AMDGPU,Metal,oneAPI}Ext`) when the
corresponding GPU package is loaded in the session. Each entry maps a
backend name to a zero-argument probe returning a functional
`KernelAbstractions.Backend` or `nothing`.
"""
const BACKEND_PROBES = Vector{Pair{Symbol,Function}}()

"""
$(TYPEDSIGNATURES)

Register a GPU backend probe. Called from the package extensions' `__init__`;
not intended for direct use.
"""
function register_backend!(name::Symbol, probe::Function)
    any(p -> first(p) === name, BACKEND_PROBES) || push!(BACKEND_PROBES, name => probe)
    return nothing
end

"""
$(TYPEDSIGNATURES)

Return the best available compute backend. GPU backends become available by
loading their package (e.g. `using CUDA`) in the session, which activates the
corresponding package extension.
`prefer` may name a specific backend (`:cuda`, `:amdgpu`, `:metal`,
`:oneapi`), request `:none` to force the CPU, or `:auto` to take the first
functional GPU. Falls back to the multi-threaded `CPU()` backend.
"""
function get_best_backend(; prefer::Symbol = :auto)
    prefer === :none && return CPU()
    for (name, probe) in BACKEND_PROBES
        (prefer === :auto || prefer === name) || continue
        backend = probe()
        backend !== nothing && return backend
    end
    if prefer ∉ (:auto, :none)
        @warn "Requested GPU backend :$prefer is not available (package not loaded " *
              "or device not functional); falling back to CPU." registered =
            first.(BACKEND_PROBES)
    end
    return CPU()
end

"""
$(TYPEDSIGNATURES)

Move an array to the given backend. The CPU method materializes a standard
`Array`; package extensions add methods for their device array types.
"""
to_backend(data::AbstractArray, ::CPU) = Array(data)

"""
$(TYPEDSIGNATURES)

Human-readable backend description for logs and banners, carrying the exact
chip model for hardware provenance (timing attribution across heterogeneous
campaign hosts requires the microarchitecture, not just the vendor).
"""
function backend_name(::CPU)
    model = cpu_model()
    return isempty(model) ? "$(Threads.nthreads())-thread CPU" :
           "$(Threads.nthreads())-thread CPU ($model)"
end
backend_name(b) = string(nameof(typeof(b)))

"""
$(TYPEDSIGNATURES)

Host CPU model string from the system information API, or `""` when the
query yields nothing (exotic platforms).
"""
function cpu_model()
    info = Sys.cpu_info()
    return isempty(info) ? "" : strip(info[1].model)
end

"""
$(TYPEDSIGNATURES)

Release cached device memory pools back to the driver, as inter-stage
maintenance on long campaigns. Package extensions override this per backend
(CUDA pool reclaim); the CPU method and backends without a pool-reclaim API
are no-ops.
"""
reclaim_device_memory!(_) = nothing

"""
$(TYPEDSIGNATURES)

Device and runtime fingerprint for hardware provenance: the GPU package's
`versioninfo` report (driver/runtime versions, device inventory with
memory), captured as a string for the run's hardware sidecar. Package
extensions override this per backend; the CPU method returns the empty
string (the host is fingerprinted separately).
"""
device_fingerprint(_) = ""

end # module
