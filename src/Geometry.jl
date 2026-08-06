module Geometry

using ForwardDiff: ForwardDiff
using ..Physics
using ..Detector

export inner_product, multi_channel_inner_product, compute_tangent_basis,
       compute_extrinsic_curvature_from_basis, compute_extrinsic_curvature,
       flat_response, GS_NORM_TOL

"""
Gram–Schmidt drop tolerance: candidate tangent vectors whose orthogonalized
noise-weighted norm falls below this are treated as linearly dependent
(degenerate parameter directions) and excluded from the basis.
"""
const GS_NORM_TOL = 1e-14

"""
    inner_product(h1, h2, Sn_vals, df)

Noise-weighted inner product `4 df Σ Re(h1* h2)/Sn` over the frequency grid.
"""
function inner_product(h1::AbstractVector, h2::AbstractVector, Sn_vals::AbstractVector, df::Real)
    return 4.0 * df * mapreduce((x, y, s) -> real(conj(x) * y) / s, +, h1, h2, Sn_vals)
end

"""
    multi_channel_inner_product(H1, H2, Sn_vals, df)

Sum of [`inner_product`](@ref) over corresponding channels of the tuples
`H1`, `H2` (generic over 2-channel A/E and 3-channel A/E/T configurations).
"""
function multi_channel_inner_product(H1, H2, Sn_vals::AbstractVector, df::Real)
    total_ip = 0.0
    for (h1, h2) in zip(H1, H2)
        total_ip += inner_product(h1, h2, Sn_vals, df)
    end
    return total_ip
end

"""
    flat_response(p, freqs, wp) -> Vector

Full detector response at parameters `p`, flattened to a real vector
`[re(C₁); im(C₁); re(C₂); im(C₂); …]` over the active channels — the map
whose Jacobian defines the signal-manifold tangent space. Generic over dual
numbers.
"""
function flat_response(p::AbstractVector, freqs::AbstractVector, wp::WaveformParams)
    h = scaled_waveform_model(p, freqs, wp)
    chans = project_to_tdi(h, freqs, p, wp)
    n = length(freqs)
    out = Vector{real(eltype(chans[1]))}(undef, 2 * length(chans) * n)
    off = 0
    for c in chans
        @inbounds for i in 1:n
            out[off + i] = real(c[i])
            out[off + n + i] = imag(c[i])
        end
        off += 2n
    end
    return out
end

"""
    unflatten_channels(v, n_bins, ::Val{NCH}) -> NTuple{NCH} of complex Vectors

Inverse of the [`flat_response`](@ref) layout. The channel count is a `Val`
so the tuple length — and therefore the return type — is known to the
compiler.
"""
function unflatten_channels(v::AbstractVector, n_bins::Integer, ::Val{NCH}) where {NCH}
    return ntuple(Val(NCH)) do c
        off = 2 * n_bins * (c - 1)
        complex.(view(v, off+1:off+n_bins), view(v, off+n_bins+1:off+2n_bins))
    end
end

"""
    compute_tangent_basis(theta_0, freqs, Sn_vals, df, wp::WaveformParams)

Noise-weighted orthonormal tangent basis of the signal manifold at `theta_0`,
via one ForwardDiff Jacobian of [`flat_response`](@ref) followed by modified
Gram–Schmidt. Directions whose orthogonalized norm falls below
[`GS_NORM_TOL`](@ref) (exactly degenerate combinations, e.g. the equal-mass
spin difference χ_a) are dropped; the returned basis may have fewer than 6
elements. Each element is an `nch`-tuple of complex channel vectors.
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
        for e in basis
            proj = multi_channel_inner_product(w, e, Sn_vals, df)
            for c in 1:NCH
                w[c] .-= proj .* e[c]
            end
        end
        norm_w = sqrt(multi_channel_inner_product(w, w, Sn_vals, df))
        if norm_w > GS_NORM_TOL
            push!(basis, map(c -> c ./ norm_w, w))
        end
    end

    return basis
end

# Explicit tag types for the fused nested-dual directional derivatives.
struct DirDerivInner end
struct DirDerivOuter end

"""
    value_and_directional_derivs(g, s0) -> (h, dh, d2h)

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
    compute_extrinsic_curvature_from_basis(theta_0, u_dir, basis, freqs, Sn_vals, df, wp)

Directional extrinsic curvature `K(u) = ‖P⊥ ∂²_u h‖²` and Fisher norm
`g(u,u) = ‖∂_u h‖²` at `theta_0` along `u_dir`, projecting the directional
second derivative against the precomputed tangent `basis`. Both first and
second derivatives come from one fused nested-dual evaluation. `K` and `g`
are exactly even in `u_dir`.
"""
function compute_extrinsic_curvature_from_basis(theta_0::AbstractVector, u_dir::AbstractVector,
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
    compute_extrinsic_curvature(theta_0, u_dir, freqs, Sn_vals, df, wp)

Convenience wrapper computing the tangent basis and the directional curvature
in one call. For repeated directions at a fixed base point (2D mapping), use
[`compute_tangent_basis`](@ref) once and
[`compute_extrinsic_curvature_from_basis`](@ref) per direction instead.
"""
function compute_extrinsic_curvature(theta_0::AbstractVector, u_dir::AbstractVector,
                                     freqs::AbstractVector, Sn_vals::AbstractVector, df::Real,
                                     wp::WaveformParams)
    basis = compute_tangent_basis(theta_0, freqs, Sn_vals, df, wp)
    return compute_extrinsic_curvature_from_basis(theta_0, u_dir, basis, freqs, Sn_vals, df, wp)
end

end # module
