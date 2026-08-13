module CurvatureDistinguishabilityAMDGPUExt

using AMDGPU
using CurvatureDistinguishability.Backends

function __init__()
    Backends.register_backend!(:amdgpu, () -> AMDGPU.functional() ? ROCBackend() : nothing)
end

Backends.to_backend(data::AbstractArray, ::ROCBackend) = ROCArray(data)
Backends.backend_name(::ROCBackend) =
    AMDGPU.functional() ? "AMD ROCm GPU ($(AMDGPU.HIP.name(AMDGPU.device())))" :
    "AMD ROCm GPU"

end
