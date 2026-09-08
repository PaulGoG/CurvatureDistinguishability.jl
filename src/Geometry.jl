"""
Differential geometry of the signal manifold: noise-weighted inner
products, the orthonormal tangent basis, and directional extrinsic
curvature from fused nested-dual automatic differentiation.
"""
module Geometry

using DocStringExtensions: TYPEDSIGNATURES
using ForwardDiff: ForwardDiff
using ..Physics
using ..Detector

export inner_product, multi_channel_inner_product, compute_tangent_basis,
    compute_extrinsic_curvature_from_basis, compute_extrinsic_curvature,
    boundary_radius, cap_unbounded_radii!, cap_at_prior
public flat_response, GS_NORM_TOL, K_UNDERFLOW, WALL_RTOL, value_and_directional_derivs

"""
Gram–Schmidt drop tolerance, relative to the candidate's norm before
orthogonalization: a tangent vector whose orthogonalized noise-weighted norm
falls below this fraction of its original norm is treated as linearly
dependent (a degenerate parameter direction) and excluded from the basis.
Relative, because the Jacobian columns span several orders of magnitude in
norm and an absolute threshold would compare round-off of a large column
with the genuine residual of a small one.
"""
const GS_NORM_TOL = 1e-10

"""
Curvature underflow guard: `K` values at or below this are treated as
exactly flat directions (infinite mathematical boundary radius) instead of
dividing into the `(16ρ²/K)^{1/4}` radius.
"""
const K_UNDERFLOW = 1e-300

"""
$(TYPEDSIGNATURES)

Mathematical boundary radius `(16 ρ²/K)^{1/4}` of the leading-order law
`D² = K r⁴/16` at the threshold `rho_sq`; `Inf` for curvatures at or below
[`K_UNDERFLOW`](@ref) (exactly flat directions).
"""
boundary_radius(K::Real, rho_sq::Real) =
    K > K_UNDERFLOW ? (16.0 * rho_sq / K)^(1 / 4) : Inf

"""
Relative tolerance of the prior-wall test: a direction whose mathematical
radius reaches the prior box within this fraction is capped exactly at the
wall and flagged prior-limited. The crossover vertices located by
bisection satisfy `r_math = r_box` only to round-off, and the plain
comparison `r_math >= r_box` left them on the curvature side, which cut
the drawn wall segment short of its corners.
"""
const WALL_RTOL = 1e-9

"""
$(TYPEDSIGNATURES)

Cap the mathematical radius `r_math` of one direction at its prior-box
distance `r_box`: returns `(r_capped, prior_limited)` with
`r_capped = r_box` and `prior_limited = true` whenever `r_math` reaches the
wall within [`WALL_RTOL`](@ref) (finite `r_box`), otherwise
`(r_math, false)`.
"""
function cap_at_prior(r_math::Real, r_box::Real)
    if isfinite(r_box) && r_math >= r_box * (1 - WALL_RTOL)
        return float(r_box), true
    end
    return float(r_math), false
end

"""
$(TYPEDSIGNATURES)

Close the boundary polygon of a map with unbounded directions (no curvature
limit and no finite prior wall): every non-finite entry of `r_cap` is
replaced by `cap_factor` times the largest finite capped radius. Returns
`(n_capped, polygon_cap)`, with `polygon_cap = NaN` when nothing was capped.
"""
function cap_unbounded_radii!(r_cap::AbstractVector, cap_factor::Real)
    unbounded = .!isfinite.(r_cap)
    n_capped = count(unbounded)
    n_capped == 0 && return 0, NaN
    polygon_cap = cap_factor * maximum(filter(isfinite, r_cap); init = 1.0)
    r_cap[unbounded] .= polygon_cap
    return n_capped, polygon_cap
end

"""
$(TYPEDSIGNATURES)

Noise-weighted inner product `4 df Σ Re(h1* h2)/Sn` over the frequency grid.
"""
function inner_product(
    h1::AbstractVector,
    h2::AbstractVector,
    Sn_vals::AbstractVector,
    df::Real,
)
    return 4.0 * df * mapreduce((x, y, s) -> real(conj(x) * y) / s, +, h1, h2, Sn_vals)
end

"""
$(TYPEDSIGNATURES)

Sum of [`inner_product`](@ref) over corresponding channels of the tuples
`H1`, `H2` (generic over 2-channel A/E and 3-channel A/E/T configurations).
"""
function multi_channel_inner_product(
    H1::Tuple,
    H2::Tuple,
    Sn_vals::AbstractVector,
    df::Real,
)
    total_ip = 0.0
    for (h1, h2) in zip(H1, H2)
        total_ip += inner_product(h1, h2, Sn_vals, df)
    end
    return total_ip
end

"""
$(TYPEDSIGNATURES)

Full detector response at parameters `p`, flattened to a real vector
`[re(C₁); im(C₁); re(C₂); im(C₂); …]` over the active channels — the map
whose Jacobian defines the signal-manifold tangent space. Generic over dual
numbers.
"""
function flat_response(p::AbstractVector, freqs::AbstractVector, wp::WaveformParams)
    chans = channel_strain(p, freqs, wp)
    n = length(freqs)
    out = Vector{real(eltype(chans[1]))}(undef, 2 * length(chans) * n)
    off = 0
    for c in chans
        @inbounds for i in 1:n
            out[off+i] = real(c[i])
            out[off+n+i] = imag(c[i])
        end
        off += 2n
    end
    return out
end

"""
$(TYPEDSIGNATURES)

Inverse of the [`flat_response`](@ref) layout. The channel count is a `Val`
so the tuple length — and therefore the return type — is known to the
compiler.
"""
function unflatten_channels(v::AbstractVector, n_bins::Integer, ::Val{NCH}) where {NCH}
    return ntuple(Val(NCH)) do c
        off = 2 * n_bins * (c - 1)
        complex.(view(v, (off+1):(off+n_bins)), view(v, (off+n_bins+1):(off+2n_bins)))
    end
end

"""
$(TYPEDSIGNATURES)

Noise-weighted orthonormal tangent basis of the signal manifold at `theta_0`,
via one ForwardDiff Jacobian of [`flat_response`](@ref) followed by modified
Gram–Schmidt. Directions whose orthogonalized norm falls below
[`GS_NORM_TOL`](@ref) (exactly degenerate combinations, e.g. the equal-mass
spin difference χ_a) are dropped; the returned basis may have fewer than 6
elements. Each element is an `n_ch`-tuple of complex channel vectors.
"""
function compute_tangent_basis(theta_0::AbstractVector, freqs::AbstractVector,
    Sn_vals::AbstractVector, df::Real,
    wp::WaveformParams{T,NCH}) where {T,NCH}
    n_bins = length(freqs)

    J_flat = ForwardDiff.jacobian(p -> flat_response(p, freqs, wp), theta_0)
    n_params = length(theta_0)

    basis = Vector{NTuple{NCH,Vector{ComplexF64}}}()
    for i in 1:n_params
        w = map(c -> collect(ComplexF64, c),
            unflatten_channels(view(J_flat, :, i), n_bins, Val(NCH)))
        norm_w0 = sqrt(multi_channel_inner_product(w, w, Sn_vals, df))
        for e in basis
            proj = multi_channel_inner_product(w, e, Sn_vals, df)
            for c in 1:NCH
                w[c] .-= proj .* e[c]
            end
        end
        norm_w = sqrt(multi_channel_inner_product(w, w, Sn_vals, df))
        if norm_w > GS_NORM_TOL * norm_w0
            push!(basis, map(c -> c ./ norm_w, w))
        end
    end

    return basis
end

# Explicit tag types for the fused nested-dual directional derivatives.
struct DirDerivInner end
struct DirDerivOuter end

"""
$(TYPEDSIGNATURES)

Value, first and second directional derivative of the vector map `g` at `s0`
from a single evaluation with nested dual numbers (no separate AD pass for
the first derivative).
"""
function value_and_directional_derivs(g, s0::Float64)
    Ti = ForwardDiff.Tag{DirDerivInner,Float64}
    To = ForwardDiff.Tag{DirDerivOuter,Float64}
    x = ForwardDiff.Dual{To}(ForwardDiff.Dual{Ti}(s0, 1.0), ForwardDiff.Dual{Ti}(1.0, 0.0))
    y = g(x)
    h = map(e -> ForwardDiff.value(ForwardDiff.value(e)), y)
    dh = map(e -> ForwardDiff.partials(ForwardDiff.value(e), 1), y)
    d2h = map(e -> ForwardDiff.partials(ForwardDiff.partials(e, 1), 1), y)
    return h, dh, d2h
end

"""
$(TYPEDSIGNATURES)

Directional extrinsic curvature `K(u) = ‖P⊥ ∂²_u h‖²` and Fisher norm
`g(u,u) = ‖∂_u h‖²` at `theta_0` along `u_dir`, projecting the directional
second derivative against the precomputed tangent `basis`. Both first and
second derivatives come from one fused nested-dual evaluation. `K` and `g`
are exactly even in `u_dir`.
"""
function compute_extrinsic_curvature_from_basis(theta_0::AbstractVector,
    u_dir::AbstractVector,
    basis::Vector{<:NTuple}, freqs::AbstractVector,
    Sn_vals::AbstractVector, df::Real,
    wp::WaveformParams{T,NCH}) where {T,NCH}
    n_bins = length(freqs)

    _, dh_flat, d2h_flat = value_and_directional_derivs(
        s -> flat_response(theta_0 .+ s .* u_dir, freqs, wp), 0.0)

    raw_curv = unflatten_channels(d2h_flat, n_bins, Val(NCH))

    tan_comp = ntuple(_ -> zeros(ComplexF64, n_bins), Val(NCH))
    for e in basis
        coeff = multi_channel_inner_product(raw_curv, e, Sn_vals, df)
        for c in 1:NCH
            tan_comp[c] .+= coeff .* e[c]
        end
    end

    normal_curv = ntuple(c -> raw_curv[c] .- tan_comp[c], Val(NCH))
    K_u = multi_channel_inner_product(normal_curv, normal_curv, Sn_vals, df)

    dh_u = unflatten_channels(dh_flat, n_bins, Val(NCH))
    g_uu = multi_channel_inner_product(dh_u, dh_u, Sn_vals, df)

    return K_u, g_uu
end

"""
$(TYPEDSIGNATURES)

Convenience wrapper computing the tangent basis and the directional curvature
in one call. For repeated directions at a fixed base point (2D mapping), use
[`compute_tangent_basis`](@ref) once and
[`compute_extrinsic_curvature_from_basis`](@ref) per direction instead.
"""
function compute_extrinsic_curvature(theta_0::AbstractVector, u_dir::AbstractVector,
    freqs::AbstractVector, Sn_vals::AbstractVector, df::Real,
    wp::WaveformParams)
    basis = compute_tangent_basis(theta_0, freqs, Sn_vals, df, wp)
    return compute_extrinsic_curvature_from_basis(
        theta_0,
        u_dir,
        basis,
        freqs,
        Sn_vals,
        df,
        wp,
    )
end

end # module
