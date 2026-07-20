module TWDCUDAExt

using CUDA
using TwoWaveformDistinguishability.Hardware

function __init__()
    Hardware.register_backend!(:cuda, () -> CUDA.functional() ? CUDABackend() : nothing)
end

Hardware.to_backend(data::AbstractArray, ::CUDABackend) = CuArray(data)
Hardware.backend_name(::CUDABackend) =
    CUDA.functional() ? "NVIDIA CUDA GPU ($(CUDA.name(CUDA.device())))" : "NVIDIA CUDA GPU"

end
