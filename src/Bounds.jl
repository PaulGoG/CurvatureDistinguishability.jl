"""
Hard physical parameter bounds: validated deviation boxes, polar ray-box
capping and interior clamping.
"""
module Bounds

using DocStringExtensions: TYPEDSIGNATURES
export ParameterBounds, default_bounds, bounds_from_config, deviation_box,
    ray_box_crossing, clamp_interior, PARAM_KEYS

"""
Canonical TOML keys for the six model parameters, in index order.
"""
const PARAM_KEYS = ("amplitude", "chirp_mass", "time", "phase", "spin1", "spin2")

"""
    ParameterBounds

Hard physical bounds on the O(1)-scaled parameter vector
`θ = [A, M_c, t_c, φ_c, χ₁, χ₂]`. `lower`/`upper` are absolute bounds used by
the constrained optimizer (`±Inf` allowed); `periodic` marks topological axes
(the phase), whose *deviation* from any base point is limited to
`±period/2` regardless of the absolute value.
"""
struct ParameterBounds
    lower::NTuple{6,Float64}
    upper::NTuple{6,Float64}
    periodic::NTuple{6,Bool}
    period::NTuple{6,Float64}
end

"""
$(TYPEDSIGNATURES)

Physical defaults: `A ≥ 0`, `M_c ≥ 0`, `t_c ≥ 0`, phase periodic with period
`2π` (unbounded for the optimizer, deviation-limited to ±π), spins in `[−1, 1]`.
"""
function default_bounds()
    return ParameterBounds((0.0, 0.0, 0.0, -Inf, -1.0, -1.0),
        (Inf, Inf, Inf, Inf, 1.0, 1.0),
        (false, false, false, true, false, false),
        (0.0, 0.0, 0.0, 2π, 0.0, 0.0))
end

"""
$(TYPEDSIGNATURES)

Build bounds from a `[parameter_bounds]` TOML table mapping parameter names
(`amplitude`, `chirp_mass`, `time`, `phase`, `spin1`, `spin2`) to
`[lower, upper]` pairs (`inf`/`-inf` allowed). Missing keys keep the
defaults; the phase axis stays periodic regardless.
"""
function bounds_from_config(cfg::AbstractDict)
    b = default_bounds()
    lower = collect(b.lower)
    upper = collect(b.upper)
    for (i, key) in enumerate(PARAM_KEYS)
        if haskey(cfg, key)
            pair = cfg[key]
            (pair isa AbstractVector && length(pair) == 2) ||
                error("[parameter_bounds].$key must be a [lower, upper] pair")
            lower[i] = Float64(pair[1])
            upper[i] = Float64(pair[2])
            lower[i] < upper[i] || error("[parameter_bounds].$key: lower must be < upper")
        end
    end
    return ParameterBounds(Tuple(lower), Tuple(upper), b.periodic, b.period)
end

"""
$(TYPEDSIGNATURES)

Deviation-space prior box for the 2D map plane `(px, py)` around the base
point `theta0`: `[lower_i − θ0_i, upper_i − θ0_i]` per axis, or `±period/2`
for periodic axes. Errors if `theta0` does not lie strictly inside the
bounds (the box must contain the origin for the polar capping to be exact).
"""
function deviation_box(b::ParameterBounds, theta0::AbstractVector, px::Integer, py::Integer)
    lims = map((px, py)) do idx
        if b.periodic[idx]
            half = b.period[idx] / 2
            (-half, half)
        else
            lo = b.lower[idx] - theta0[idx]
            hi = b.upper[idx] - theta0[idx]
            (lo < 0.0 && hi > 0.0) ||
                error(
                    "Base point component $idx (= $(theta0[idx])) is not strictly inside " *
                    "its physical bounds [$(b.lower[idx]), $(b.upper[idx])]; " *
                    "the confusion-zone capping requires an interior base point.",
                )
            (lo, hi)
        end
    end
    return lims[1][1], lims[1][2], lims[2][1], lims[2][2]
end

"""
$(TYPEDSIGNATURES)

Distance from the origin to the boundary of the axis-aligned box
`[lox, hix] × [loy, hiy]` (which must contain the origin) along the ray with
direction `(cphi, sphi)`. Returns `Inf` when both bounds along the ray are
infinite.
"""
function ray_box_crossing(
    cphi::Real,
    sphi::Real,
    lox::Real,
    hix::Real,
    loy::Real,
    hiy::Real,
)
    tx = cphi > 0 ? hix / cphi : (cphi < 0 ? lox / cphi : Inf)
    ty = sphi > 0 ? hiy / sphi : (sphi < 0 ? loy / sphi : Inf)
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
