module Hardware

using KernelAbstractions

export get_best_backend, to_backend, backend_name, register_backend!

"""
Registry of GPU backend probes, populated by the package extensions
(`TWDCUDAExt`, `TWDAMDGPUExt`, `TWDMetalExt`, `TWDoneAPIExt`) when the
corresponding GPU package is loaded in the session. Each entry maps a
backend name to a zero-argument probe returning a functional
`KernelAbstractions.Backend` or `nothing`.
"""
const BACKEND_PROBES = Vector{Pair{Symbol,Function}}()

"""
    register_backend!(name::Symbol, probe)

Register a GPU backend probe. Called from the package extensions' `__init__`;
not intended for direct use.
"""
function register_backend!(name::Symbol, probe::Function)
    any(p -> first(p) === name, BACKEND_PROBES) || push!(BACKEND_PROBES, name => probe)
    return nothing
end

"""
    get_best_backend(; prefer::Symbol = :auto)

Return the best available compute backend. GPU backends become available by
loading their package (e.g. `using CUDA`) in the session, which activates the
corresponding package extension — there is no `Main`-reflection involved.
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
              "or device not functional); falling back to CPU." registered = first.(BACKEND_PROBES)
    end
    return CPU()
end

"""
    to_backend(data, backend)

Move an array to the given backend. The CPU method materializes a standard
`Array`; package extensions add methods for their device array types.
"""
to_backend(data::AbstractArray, ::CPU) = Array(data)

"""
    backend_name(backend)

Human-readable backend description for logs and banners.
"""
backend_name(::CPU) = "$(Threads.nthreads())-thread CPU"
backend_name(b) = string(nameof(typeof(b)))

end # module
