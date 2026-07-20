module Geometry

using LinearAlgebra
using ForwardDiff
using ..Physics
using ..Detector

export inner_product, compute_extrinsic_curvature, multi_channel_inner_product, compute_tangent_basis, compute_extrinsic_curvature_from_basis

"""
    inner_product(h1::AbstractVector, h2::AbstractVector, Sn_vals::AbstractVector, df::Real) -> Float64

Computes the standard noise-weighted inner product between two frequency-domain waveforms `h1` and `h2`.
The integral is evaluated discretely as a Riemann sum over the frequency grid.

# Arguments
- `h1`: First frequency-domain waveform.
- `h2`: Second frequency-domain waveform.
- `Sn_vals`: The one-sided power spectral density of the detector noise.
- `df`: Frequency bin resolution (Hz).

# Returns
- A `Float64` representing the scalar inner product.
"""
function inner_product(h1::AbstractVector, h2::AbstractVector, Sn_vals::AbstractVector, df::Real)
    return 4.0 * df * mapreduce((x, y, s) -> real(conj(x) * y) / s, +, h1, h2, Sn_vals)
end

"""
    multi_channel_inner_product(H1, H2, Sn_vals::AbstractVector, df::Real) -> Float64

Computes the total noise-weighted inner product across multiple orthogonal detector channels.
Typically used for TDI configurations summing over the `A`, `E`, and `T` streams.

# Arguments
- `H1`: A tuple or array of frequency-domain waveforms for the first signal across multiple channels.
- `H2`: A tuple or array of frequency-domain waveforms for the second signal across multiple channels.
- `Sn_vals`: Noise PSD vector.
- `df`: Frequency resolution (Hz).
"""
function multi_channel_inner_product(H1, H2, Sn_vals::AbstractVector, df::Real)
    total_ip = 0.0
    for (h1, h2) in zip(H1, H2)
        total_ip += inner_product(h1, h2, Sn_vals, df)
    end
    return total_ip
end

"""
    compute_tangent_basis(theta_0::AbstractVector, freqs::AbstractVector, Sn_vals::AbstractVector, df::Real; kwargs...) -> Vector

Computes the Gram-Schmidt orthonormalized tangent basis for the signal manifold at parameter location `theta_0`.

This is the most computationally expensive operation in the geometry engine. By separating the Jacobian evaluation and orthogonalization out into this standalone function, the basis can be reused for thousands of different directional curvature evaluations (e.g., during 2D mapping) without hitting VRAM or RAM limits.

# Arguments
- `theta_0`: The base parameter vector (length 6).
- `freqs`: The frequency grid.
- `Sn_vals`: The one-sided power spectral density values.
- `df`: Frequency resolution.

# Returns
- A `Vector` of tuples. Each tuple contains three orthogonal `ComplexF64` arrays representing the tangent vector projected onto the A, E, and T channels.
"""
function compute_tangent_basis(theta_0::AbstractVector, freqs::AbstractVector, Sn_vals::AbstractVector, df::Real; kwargs...)
    n_bins = length(freqs)
    
    function waveform_flat(p)
        wf = scaled_waveform_model(p, freqs; kwargs...)
        A, E, T = project_to_tdi(wf, freqs, p; kwargs...)
        vA = vcat(real.(A), imag.(A))
        vE = vcat(real.(E), imag.(E))
        vT = vcat(real.(T), imag.(T))
        return vcat(vA, vE, vT)
    end

    J_flat = ForwardDiff.jacobian(waveform_flat, theta_0)
    n_params = length(theta_0)
    
    tangent_vectors = Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}}}(undef, n_params)
    offset_E = 2 * n_bins
    offset_T = 4 * n_bins
    
    for i in 1:n_params
        col = J_flat[:, i]
        A_r = col[1:n_bins]
        A_i = col[n_bins+1:offset_E]
        E_r = col[offset_E+1:offset_E+n_bins]
        E_i = col[offset_E+n_bins+1:offset_T]
        T_r = col[offset_T+1:offset_T+n_bins]
        T_i = col[offset_T+n_bins+1:end]
        
        tangent_vectors[i] = (complex.(A_r, A_i), complex.(E_r, E_i), complex.(T_r, T_i))
    end
    
    basis = Vector{Tuple{Vector{ComplexF64}, Vector{ComplexF64}, Vector{ComplexF64}}}()
    for v in tangent_vectors
        wA, wE, wT = copy(v[1]), copy(v[2]), copy(v[3])
        w = (wA, wE, wT)
        for e in basis
            proj = multi_channel_inner_product(w, e, Sn_vals, df)
            w[1] .-= proj .* e[1]
            w[2] .-= proj .* e[2]
            w[3] .-= proj .* e[3]
        end
        norm_w = sqrt(multi_channel_inner_product(w, w, Sn_vals, df))
        if norm_w > 1e-14
            push!(basis, (w[1] ./ norm_w, w[2] ./ norm_w, w[3] ./ norm_w))
        end
    end
    
    return basis
end

"""
    compute_extrinsic_curvature_from_basis(theta_0::AbstractVector, u_dir::AbstractVector, basis::Vector, freqs::AbstractVector, Sn_vals::AbstractVector, df::Real; kwargs...) -> Tuple{Float64, Float64}

Computes the Extrinsic Curvature \$K(u)\$ and the Fisher Information Matrix norm \$g_{uu}\$ in the direction `u_dir`.

This function calculates the Directional Hessian (second derivative) along `u_dir`, projects it onto the A, E, and T channels, and then orthogonalizes it against the pre-computed `basis`. The resulting vector represents the portion of the signal change that fundamentally cannot be absorbed by shifting parameters, thus dictating the true quartic scaling distance \$D^2 \\approx \\frac{1}{16} K \\delta^4\$.

# Arguments
- `theta_0`: The base parameter vector (length 6).
- `u_dir`: The normalized direction vector in parameter space.
- `basis`: The Gram-Schmidt orthonormalized tangent basis generated by `compute_tangent_basis`.
- `freqs`: Frequency grid.
- `Sn_vals`: Noise PSD vector.
- `df`: Frequency resolution.

# Returns
- A tuple `(K_u, g_uu)` representing the Extrinsic Curvature and the linear Fisher Norm.
"""
function compute_extrinsic_curvature_from_basis(theta_0::AbstractVector, u_dir::AbstractVector, 
                                                basis::Vector, freqs::AbstractVector, Sn_vals::AbstractVector, df::Real; kwargs...)
    n_bins = length(freqs)
    
    function waveform_flat(p)
        wf = scaled_waveform_model(p, freqs; kwargs...)
        A, E, T = project_to_tdi(wf, freqs, p; kwargs...)
        vA = vcat(real.(A), imag.(A))
        vE = vcat(real.(E), imag.(E))
        vT = vcat(real.(T), imag.(T))
        return vcat(vA, vE, vT)
    end

    function directional_h(s)
        p_shift = theta_0 .+ s .* u_dir
        return waveform_flat(p_shift)
    end
    
    d2h_flat = ForwardDiff.derivative(s -> ForwardDiff.derivative(directional_h, s), 0.0)
    
    offset_E = 2 * n_bins
    offset_T = 4 * n_bins
    
    raw_A_r = @view d2h_flat[1:n_bins]
    raw_A_i = @view d2h_flat[n_bins+1:offset_E]
    raw_E_r = @view d2h_flat[offset_E+1:offset_E+n_bins]
    raw_E_i = @view d2h_flat[offset_E+n_bins+1:offset_T]
    raw_T_r = @view d2h_flat[offset_T+1:offset_T+n_bins]
    raw_T_i = @view d2h_flat[offset_T+n_bins+1:end]
    
    raw_curv = (complex.(raw_A_r, raw_A_i), complex.(raw_E_r, raw_E_i), complex.(raw_T_r, raw_T_i))
    
    tan_comp_A = zeros(ComplexF64, n_bins)
    tan_comp_E = zeros(ComplexF64, n_bins)
    tan_comp_T = zeros(ComplexF64, n_bins)
    
    for e in basis
        coeff = multi_channel_inner_product(raw_curv, e, Sn_vals, df)
        tan_comp_A .+= coeff .* e[1]
        tan_comp_E .+= coeff .* e[2]
        tan_comp_T .+= coeff .* e[3]
    end
    
    norm_curv_A = raw_curv[1] .- tan_comp_A
    norm_curv_E = raw_curv[2] .- tan_comp_E
    norm_curv_T = raw_curv[3] .- tan_comp_T
    
    normal_curv_vec = (norm_curv_A, norm_curv_E, norm_curv_T)
    
    K_u = multi_channel_inner_product(normal_curv_vec, normal_curv_vec, Sn_vals, df)
    
    dh_flat = ForwardDiff.derivative(directional_h, 0.0)
    dh_A_r = @view dh_flat[1:n_bins]
    dh_A_i = @view dh_flat[n_bins+1:offset_E]
    dh_E_r = @view dh_flat[offset_E+1:offset_E+n_bins]
    dh_E_i = @view dh_flat[offset_E+n_bins+1:offset_T]
    dh_T_r = @view dh_flat[offset_T+1:offset_T+n_bins]
    dh_T_i = @view dh_flat[offset_T+n_bins+1:end]
    
    dh_u = (complex.(dh_A_r, dh_A_i), complex.(dh_E_r, dh_E_i), complex.(dh_T_r, dh_T_i))
    g_uu = multi_channel_inner_product(dh_u, dh_u, Sn_vals, df)
    
    return K_u, g_uu
end

"""
    compute_extrinsic_curvature(theta_0::AbstractVector, u_dir::AbstractVector, freqs::AbstractVector, Sn_vals::AbstractVector, df::Real; kwargs...) -> Tuple{Float64, Float64}

Convenience wrapper that computes the tangent basis and the extrinsic curvature in one step. 
Warning: This recalculates the massive Jacobian every time it is called. For 2D mapping tasks with multiple angles, use `compute_tangent_basis` and `compute_extrinsic_curvature_from_basis` directly to save memory and compute time.
"""
function compute_extrinsic_curvature(theta_0::AbstractVector, u_dir::AbstractVector, 
                                     freqs::AbstractVector, Sn_vals::AbstractVector, df::Real; kwargs...)
    basis = compute_tangent_basis(theta_0, freqs, Sn_vals, df; kwargs...)
    return compute_extrinsic_curvature_from_basis(theta_0, u_dir, basis, freqs, Sn_vals, df; kwargs...)
end

end # module
