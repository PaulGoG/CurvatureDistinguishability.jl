module CurvatureDistinguishabilityoneAPIExt

using oneAPI
using CurvatureDistinguishability.Backends

function __init__()
    Backends.register_backend!(
        :oneapi,
        () -> oneAPI.functional() ? oneAPIBackend() : nothing,
    )
end

Backends.to_backend(data::AbstractArray, ::oneAPIBackend) = oneArray(data)
Backends.backend_name(::oneAPIBackend) =
    oneAPI.functional() ?
    "Intel oneAPI GPU ($(strip(oneAPI.oneL0.properties(oneAPI.device()).name)))" :
    "Intel oneAPI GPU"

end
