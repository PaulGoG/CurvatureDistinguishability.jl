module CurvatureDistinguishabilityMetalExt

using Metal: Metal
using CurvatureDistinguishability.Backends

# The probe always declines (the warning gives the reason): `auto` falls back
# to the CPU, a run that names the Metal backend stops, and with `require_gpu`
# it stops before any output.
function metal_probe()
    @warn "Metal has no Float64 type and the device path of " *
          "CurvatureDistinguishability is double precision: the Metal backend " *
          "is not used." maxlog = 1
    return nothing
end

function __init__()
    Backends.register_backend!(:metal, metal_probe)
end

end
