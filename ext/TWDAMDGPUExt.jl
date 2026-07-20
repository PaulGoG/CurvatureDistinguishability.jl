module TWDAMDGPUExt

using AMDGPU
using TwoWaveformDistinguishability.Hardware

function __init__()
    Hardware.register_backend!(:amdgpu, () -> AMDGPU.functional() ? ROCBackend() : nothing)
end

Hardware.to_backend(data::AbstractArray, ::ROCBackend) = ROCArray(data)
Hardware.backend_name(::ROCBackend) = "AMD ROCm GPU"

end
