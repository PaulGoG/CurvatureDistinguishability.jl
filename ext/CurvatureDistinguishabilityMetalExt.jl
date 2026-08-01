module CurvatureDistinguishabilityMetalExt

using Metal
using CurvatureDistinguishability.Backends

function __init__()
    Backends.register_backend!(:metal, () -> Metal.functional() ? MetalBackend() : nothing)
end

Backends.to_backend(data::AbstractArray, ::MetalBackend) = MtlArray(data)
Backends.backend_name(::MetalBackend) = "Apple Metal GPU"

end
