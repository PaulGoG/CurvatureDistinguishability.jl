"""
Hard physical parameter bounds: validated deviation boxes, polar ray-box
capping and interior clamping.
"""
module Bounds

using DocStringExtensions: TYPEDSIGNATURES
using ..Physics: N_PARAMS
export ParameterBounds,
    default_bounds, bounds_from_config, deviation_box,
    ray_box_crossing, clamp_interior
public PARAM_KEYS, PHASE_INDEX, SPIN_INDICES

"""
Index of the coalescence phase in the parameter vector, the one periodic
coordinate (period `4π`, deviations capped at `±π`; see [`default_bounds`](@ref)).
"""
const PHASE_INDEX = 4

"""
Indices of the two aligned spins in the parameter vector; the only pair of
parameters sharing a unit, and the plane of the exact `χ_a` degeneracy.
"""
const SPIN_INDICES = (5, 6)

"""
Canonical TOML keys for the six model parameters, in index order.
"""
const PARAM_KEYS = ("luminosity_distance", "chirp_mass", "coalescence_time",
    "coalescence_phase", "spin1", "spin2")

"""
    ParameterBounds

Hard physical bounds on the O(1)-scaled parameter vector
`θ = [A, M_c, t_c, φ_c, χ₁, χ₂]`. `lower`/`upper` are absolute bounds used by
the constrained optimizer (`±Inf` allowed); `periodic` marks topological axes
(the phase) with their `period`, and `deviation_cap` limits the deviation
from any base point on those axes to `±deviation_cap` regardless of the
absolute value (`0 < deviation_cap ≤ period/2`). The four-argument
constructor sets the cap to half the period, the antipode on the circle.
"""
struct ParameterBounds
    lower::NTuple{N_PARAMS,Float64}
    upper::NTuple{N_PARAMS,Float64}
    periodic::NTuple{N_PARAMS,Bool}
    period::NTuple{N_PARAMS,Float64}
    deviation_cap::NTuple{N_PARAMS,Float64}
    function ParameterBounds(lower, upper, periodic, period, deviation_cap)
        for i in 1:N_PARAMS
            periodic[i] || continue
            0 < deviation_cap[i] <= period[i] / 2 || throw(
                ArgumentError(
                    "periodic component $i: the deviation cap ($(deviation_cap[i])) " *
                    "must lie in (0, period/2] with period $(period[i])",
                ),
            )
        end
        return new(lower, upper, periodic, period, deviation_cap)
    end
end

ParameterBounds(lower, upper, periodic, period) =
    ParameterBounds(lower, upper, periodic, period, period ./ 2)

"""
$(TYPEDSIGNATURES)

Physical defaults: `D_L ≥ 0`, `M_c ≥ 0`, `t_c ≥ 0`, spins in `[−1, 1]`, and the
phase unbounded for the optimizer with period `4π`: harmonic `k` carries
`−(k/2)φ_c`, so a `2π` shift reverses the sign of the odd harmonics `k = 1, 3`
(at equal mass they vanish and `2π` is already a period). Phase deviations
from a base point are capped at `±π`, half the period of the dominant `k = 2`
harmonic: along the pure-phase ray its contribution to `D²`, `∝ sin²(Δφ_c/2)`,
grows monotonically up to `|Δφ_c| = π` and decreases beyond, where the radial
confusion-zone construction no longer applies.
"""
function default_bounds()
    return ParameterBounds((0.0, 0.0, 0.0, -Inf, -1.0, -1.0),
        (Inf, Inf, Inf, Inf, 1.0, 1.0),
        (false, false, false, true, false, false),
        (0.0, 0.0, 0.0, 4π, 0.0, 0.0),
        (0.0, 0.0, 0.0, π, 0.0, 0.0))
end

"""
$(TYPEDSIGNATURES)

Build bounds from a `[parameter_bounds]` TOML table mapping parameter names
(`luminosity_distance`, `chirp_mass`, `coalescence_time`, `coalescence_phase`,
`spin1`, `spin2`) to `[lower, upper]` pairs (`inf`/`-inf` allowed). Missing
keys keep the defaults; the phase axis stays periodic regardless.
"""
function bounds_from_config(cfg::AbstractDict)
    b = default_bounds()
    lower = collect(b.lower)
    upper = collect(b.upper)
    for (i, key) in enumerate(PARAM_KEYS)
        if haskey(cfg, key)
            pair = cfg[key]
            (pair isa AbstractVector && length(pair) == 2) ||
                throw(
                    ArgumentError("[parameter_bounds].$key must be a [lower, upper] pair"),
                )
            lower[i] = Float64(pair[1])
            upper[i] = Float64(pair[2])
            lower[i] < upper[i] ||
                throw(ArgumentError("[parameter_bounds].$key: lower must be < upper"))
        end
    end
    return ParameterBounds(Tuple(lower), Tuple(upper), b.periodic, b.period,
        b.deviation_cap)
end

"""
$(TYPEDSIGNATURES)

Deviation-space prior box for the 2D map plane `(px, py)` around the base
point `theta0`: `[lower_i − θ0_i, upper_i − θ0_i]` per axis, or
`±deviation_cap` for periodic axes. Errors if `theta0` does not lie strictly inside the
bounds (the box must contain the origin for the polar capping to be exact).
"""
function deviation_box(b::ParameterBounds, theta0::AbstractVector, px::Integer, py::Integer)
    lims = map((px, py)) do idx
        if b.periodic[idx]
            cap = b.deviation_cap[idx]
            (-cap, cap)
        else
            lo = b.lower[idx] - theta0[idx]
            hi = b.upper[idx] - theta0[idx]
            (lo < 0.0 && hi > 0.0) ||
                throw(
                    ArgumentError(
                        "Base point component $idx (= $(theta0[idx])) is not strictly inside " *
                        "its physical bounds [$(b.lower[idx]), $(b.upper[idx])]; " *
                        "the confusion-zone capping requires an interior base point.",
                    ),
                )
            (lo, hi)
        end
    end
    return lims[1][1], lims[1][2], lims[2][1], lims[2][2]
end

"""
$(TYPEDSIGNATURES)

Distance from the origin to the boundary of the axis-aligned box
`[x_lower, x_upper] × [y_lower, y_upper]` (which must contain the origin)
along the ray with direction `(cos_phi, sin_phi)`. Returns `Inf` when both
bounds along the ray are infinite.
"""
function ray_box_crossing(
    cos_phi::Real,
    sin_phi::Real,
    x_lower::Real,
    x_upper::Real,
    y_lower::Real,
    y_upper::Real,
)
    tx = cos_phi > 0 ? x_upper / cos_phi : (cos_phi < 0 ? x_lower / cos_phi : Inf)
    ty = sin_phi > 0 ? y_upper / sin_phi : (sin_phi < 0 ? y_lower / sin_phi : Inf)
    return min(tx, ty)
end

"""
$(TYPEDSIGNATURES)

Clamp `theta` strictly inside the finite bounds (interior-point optimizers
require a strictly feasible start). `margin` is relative to the bound width
(or absolute where one side is infinite). Periodic axes are left untouched.
"""
function clamp_interior(theta::AbstractVector, b::ParameterBounds; margin::Real = 1e-8)
    out = collect(float.(theta))
    for i in eachindex(out)
        b.periodic[i] && continue
        lo, hi = b.lower[i], b.upper[i]
        m =
            isfinite(lo) && isfinite(hi) ? margin * (hi - lo) :
            margin * max(1.0, abs(out[i]))
        isfinite(lo) && out[i] <= lo && (out[i] = lo + m)
        isfinite(hi) && out[i] >= hi && (out[i] = hi - m)
    end
    return out
end

end # module
