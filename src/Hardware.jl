module Hardware

using KernelAbstractions
using Adapt

export get_best_backend, @kernel, synchronize, to_backend

"""
    get_best_backend()

Dynamically determines and returns the best available compute backend (GPU or CPU) 
using KernelAbstractions.jl. 

It checks for the presence of loaded GPU packages (CUDA, AMDGPU, Metal, oneAPI) in the 
current Julia session. If none are available or functional, it safely falls back to 
the multi-threaded CPU backend.
"""
function get_best_backend()
    # Check for NVIDIA GPUs (CUDA.jl)
    if isdefined(Main, :CUDA) && Main.CUDA.functional()
        return Main.CUDA.CUDABackend()
    end
    
    # Check for AMD GPUs (AMDGPU.jl)
    if isdefined(Main, :AMDGPU) && Main.AMDGPU.functional()
        return Main.AMDGPU.ROCBackend()
    end
    
    # Check for Apple Silicon (Metal.jl)
    if isdefined(Main, :Metal) && Main.Metal.functional()
        return Main.Metal.MetalBackend()
    end
    
    # Check for Intel GPUs (oneAPI.jl)
    if isdefined(Main, :oneAPI) && Main.oneAPI.functional()
        return Main.oneAPI.oneAPIBackend()
    end
    
    # Fallback to Multi-threaded CPU
    return CPU()
end

"""
    to_backend(data, backend)

Moves `data` (e.g., an Array) to the specific hardware `backend`.
"""
function to_backend(data, backend::Backend)
    if backend isa CPU
        return Array(data) # Ensure it's a standard CPU array
    end
    
    # Adapt.jl handles the conversion to the specific GPU array type
    # provided the corresponding GPU package is loaded.
    if isdefined(Main, :CUDA) && backend isa Main.CUDA.CUDABackend
        return Adapt.adapt(Main.CUDA.CuArray, data)
    elseif isdefined(Main, :AMDGPU) && backend isa Main.AMDGPU.ROCBackend
        return Adapt.adapt(Main.AMDGPU.ROCArray, data)
    elseif isdefined(Main, :Metal) && backend isa Main.Metal.MetalBackend
        return Adapt.adapt(Main.Metal.MtlArray, data)
    elseif isdefined(Main, :oneAPI) && backend isa Main.oneAPI.oneAPIBackend
        return Adapt.adapt(Main.oneAPI.oneArray, data)
    end
    
    error("Unsupported backend or GPU package not loaded in Main.")
end

end # module