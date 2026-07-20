module TWDoneAPIExt

using oneAPI
using TwoWaveformDistinguishability.Hardware

function __init__()
    Hardware.register_backend!(:oneapi, () -> oneAPI.functional() ? oneAPIBackend() : nothing)
end

Hardware.to_backend(data::AbstractArray, ::oneAPIBackend) = oneArray(data)
Hardware.backend_name(::oneAPIBackend) = "Intel oneAPI GPU"

end
