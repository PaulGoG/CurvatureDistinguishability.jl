module CurvatureDistinguishabilityCUDAExt

using CUDA
using CurvatureDistinguishability.Backends

function __init__()
    Backends.register_backend!(:cuda, () -> CUDA.functional() ? CUDABackend() : nothing)
end

Backends.to_backend(data::AbstractArray, ::CUDABackend) = CuArray(data)
Backends.backend_name(::CUDABackend) =
    CUDA.functional() ? "NVIDIA CUDA GPU ($(CUDA.name(CUDA.device())))" : "NVIDIA CUDA GPU"

end
