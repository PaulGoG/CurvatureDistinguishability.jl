module TWDAMDGPUExt

using AMDGPU
using TwoWaveformDistinguishability.Backends

function __init__()
    Backends.register_backend!(:amdgpu, () -> AMDGPU.functional() ? ROCBackend() : nothing)
end

Backends.to_backend(data::AbstractArray, ::ROCBackend) = ROCArray(data)
Backends.backend_name(::ROCBackend) = "AMD ROCm GPU"

end
