module TWDMetalExt

using Metal
using TwoWaveformDistinguishability.Hardware

function __init__()
    Hardware.register_backend!(:metal, () -> Metal.functional() ? MetalBackend() : nothing)
end

Hardware.to_backend(data::AbstractArray, ::MetalBackend) = MtlArray(data)
Hardware.backend_name(::MetalBackend) = "Apple Metal GPU"

end
