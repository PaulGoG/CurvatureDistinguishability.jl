module CurvatureDistinguishabilityoneAPIExt

using oneAPI
using CurvatureDistinguishability.Backends

function __init__()
    Backends.register_backend!(:oneapi, () -> oneAPI.functional() ? oneAPIBackend() : nothing)
end

Backends.to_backend(data::AbstractArray, ::oneAPIBackend) = oneArray(data)
Backends.backend_name(::oneAPIBackend) = "Intel oneAPI GPU"

end
