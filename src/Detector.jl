"""
Long-wavelength LISA response on the analytic constellation orbits: the
Michelson antenna patterns of the three spacecraft combined into the
noise-orthogonal A and E channels, the orbital Doppler phase and the
finite-arm transfer roll-off, assembled per frequency bin and per harmonic
into the observed channel strains.
"""
module Detector

using DocStringExtensions: TYPEDSIGNATURES
using ..Physics: WaveformParams, ResponseGeometry, SECONDS_PER_YEAR, R_ORBIT_SEC,
    harmonic_amplitudes, harmonic_phase, spa_time, isco_frequency, inspiral_taper,
    observation_window,
    spin_beta, total_mass, mass_asymmetry, transfer_frequency

export channel_strain, channel_strain_bin, n_channels
public spacecraft_positions, channel_patterns, doppler_phase

"""
$(TYPEDSIGNATURES)

Number of active TDI channels: 2 (A, E) by default, 3 when the identically
zero null channel T is explicitly requested through the `include_t_channel`
keyword of [`WaveformParams`](@ref). Read from the type parameter, so it
constant-folds in specialized code.

```jldoctest
julia> n_channels(waveform_params())
2
```
"""
n_channels(::WaveformParams{T,NCH}) where {T,NCH} = NCH

@inline dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]

@inline function unit_arm(from, to)
    d = (to[1] - from[1], to[2] - from[2], to[3] - from[3])
    inv_norm = 1 / sqrt(dot3(d, d))
    return (d[1] * inv_norm, d[2] * inv_norm, d[3] * inv_norm)
end

"""
$(TYPEDSIGNATURES)

Spacecraft positions [light-seconds, ecliptic SSB frame] on the analytic
equal-arm orbits of Rubbo, Cornish & Poujade (2004, Eq. 1) to second order
in the orbital eccentricity `e = L/(2√3 R)`: with the orbital phase
`α = Ωt + κ` given through `(cos α, sin α)` and the constellation phases
`β_n = 2π(n−1)/3 + λ_c` from `geometry`,
`x_n = R cos α + ½eR[cos(2α − β_n) − 3 cos β_n] + ⅛e²R[3 cos(3α − 2β_n) − 10 cos α − 5 cos(α − 2β_n)]`,
`y_n = R sin α + ½eR[sin(2α − β_n) − 3 sin β_n] + ⅛e²R[3 sin(3α − 2β_n) − 10 sin α + 5 sin(α − 2β_n)]`,
`z_n = −√3 eR cos(α − β_n) + √3 e²R[cos²(α − β_n) + 2 sin²(α − β_n)]`.
The constellation plane is inclined by 60° to the ecliptic and cartwheels
once per orbit. Only `cos α`, `sin α` are transcendental inputs; every
other angle follows by angle addition from the stored constellation phases.
"""
@inline function spacecraft_positions(cos_alpha, sin_alpha, geometry::ResponseGeometry)
    R = R_ORBIT_SEC
    e = geometry.eccentricity
    cos_2a = cos_alpha^2 - sin_alpha^2
    sin_2a = 2 * sin_alpha * cos_alpha
    cos_3a = cos_2a * cos_alpha - sin_2a * sin_alpha
    sin_3a = sin_2a * cos_alpha + cos_2a * sin_alpha
    return ntuple(Val(3)) do n
        cb, sb = geometry.cos_beta[n], geometry.sin_beta[n]
        c2b, s2b = geometry.cos_2beta[n], geometry.sin_2beta[n]
        cos_2a_b = cos_2a * cb + sin_2a * sb        # cos(2α − β)
        sin_2a_b = sin_2a * cb - cos_2a * sb        # sin(2α − β)
        cos_3a_2b = cos_3a * c2b + sin_3a * s2b     # cos(3α − 2β)
        sin_3a_2b = sin_3a * c2b - cos_3a * s2b     # sin(3α − 2β)
        cos_a_2b = cos_alpha * c2b + sin_alpha * s2b # cos(α − 2β)
        sin_a_2b = sin_alpha * c2b - cos_alpha * s2b # sin(α − 2β)
        cos_a_b = cos_alpha * cb + sin_alpha * sb   # cos(α − β)
        sin_a_b = sin_alpha * cb - cos_alpha * sb   # sin(α − β)
        x =
            R * cos_alpha + 0.5 * e * R * (cos_2a_b - 3 * cb) +
            0.125 * e^2 * R * (3 * cos_3a_2b - 10 * cos_alpha - 5 * cos_a_2b)
        y =
            R * sin_alpha + 0.5 * e * R * (sin_2a_b - 3 * sb) +
            0.125 * e^2 * R * (3 * sin_3a_2b - 10 * sin_alpha + 5 * sin_a_2b)
        z = -sqrt(3.0) * e * R * cos_a_b + sqrt(3.0) * e^2 * R * (cos_a_b^2 + 2 * sin_a_b^2)
        (x, y, z)
    end
end

"""
$(TYPEDSIGNATURES)

Orbital Doppler phase `2πf n̂·R_c(t)` of a wave from direction `n̂` received
at the constellation centre `R_c = R(cos α, sin α, 0)` — a wavefront reaches
the centre `n̂·R_c` light-seconds before the solar-system barycentre, which
in the `e^{-2πift}` Fourier convention multiplies the barycentric strain by
`e^{+2πif n̂·R_c}`.
"""
@inline function doppler_phase(f, cos_alpha, sin_alpha, geometry::ResponseGeometry)
    n = geometry.source_direction
    return 2 * π * f * R_ORBIT_SEC * (n[1] * cos_alpha + n[2] * sin_alpha)
end

"""
$(TYPEDSIGNATURES)

Antenna patterns of the noise-orthogonal channels at time `t` [s]:
`(F⁺_A, F×_A, F⁺_E, F×_E)`. Each spacecraft `i` synthesises a Michelson
interferometer with detector tensor `D_i = ½(â⊗â − b̂⊗b̂)` from its two arm
unit vectors, giving `F⁺_i = D_i : e⁺` and `F×_i = D_i : e×` with the
wave-frame polarization tensors of [`ResponseGeometry`](@ref); the three
Michelson responses are combined into `A = (Z − X)/√2` and
`E = (X − 2Y + Z)/√6` (Baghi et al. 2026, Eq. 40) and rescaled by `√(2/3)` so
that both channels carry the single-Michelson noise PSD the noise model
provides: `F_A = (F_Z − F_X)/√3`, `F_E = (F_X − 2F_Y + F_Z)/3`. Their
sky-and-polarization average of `F⁺² + F×²` is `3/10`, the long-wavelength
response of Robson et al. (2019). The null combination `X + Y + Z`
vanishes identically in this limit.
"""
@inline function channel_patterns(t, geometry::ResponseGeometry, orbit_phase)
    alpha = 2 * π / SECONDS_PER_YEAR * t + orbit_phase
    cos_alpha, sin_alpha = cos(alpha), sin(alpha)
    r1, r2, r3 = spacecraft_positions(cos_alpha, sin_alpha, geometry)
    u12 = unit_arm(r1, r2)
    u23 = unit_arm(r2, r3)
    u31 = unit_arm(r3, r1)
    p, q = geometry.p, geometry.q
    p12, q12 = dot3(u12, p), dot3(u12, q)
    p23, q23 = dot3(u23, p), dot3(u23, q)
    p31, q31 = dot3(u31, p), dot3(u31, q)
    # Michelson at spacecraft 1 (arms 1→2, 1→3), 2 (2→3, 2→1), 3 (3→1, 3→2);
    # arm orientation drops out of the quadratic forms
    plus_1 = 0.5 * ((p12^2 - q12^2) - (p31^2 - q31^2))
    cross_1 = p12 * q12 - p31 * q31
    plus_2 = 0.5 * ((p23^2 - q23^2) - (p12^2 - q12^2))
    cross_2 = p23 * q23 - p12 * q12
    plus_3 = 0.5 * ((p31^2 - q31^2) - (p23^2 - q23^2))
    cross_3 = p31 * q31 - p23 * q23
    inv_sqrt3 = 1 / sqrt(3.0)
    return ((plus_3 - plus_1) * inv_sqrt3, (cross_3 - cross_1) * inv_sqrt3,
        (plus_1 - 2 * plus_2 + plus_3) / 3, (cross_1 - 2 * cross_2 + cross_3) / 3,
        cos_alpha, sin_alpha)
end

"""
$(TYPEDSIGNATURES)

Observed strains `(h̃_A, h̃_E)` at frequency `f` [Hz] of a source with
luminosity distance `distance` [s], chirp mass `chirp_mass` [s],
coalescence time `coalescence_time` [s], coalescence phase and aligned spins
`chi1` (heavier body), `chi2`, in the `e^{-2πift}` Fourier convention:
`h̃_C(f) = Σ_k w_k(f) 𝒯(f) [a₊_k F⁺_C(t_k) − i a×_k F×_C(t_k)] e^{-iΨ_k(f)} e^{+iΔ_D(t_k)}`
over the harmonics `k` of [`harmonic_amplitudes`](@ref), each with its own
stationary-phase phase [`harmonic_phase`](@ref), its own emission time
`t_k = t(2f/k)` ([`spa_time`](@ref)) at which the antenna patterns
[`channel_patterns`](@ref) and the Doppler phase [`doppler_phase`](@ref) are
evaluated, the finite-observation weight ([`observation_window`](@ref), when
`wp.observation_time > 0`), the innermost-stable-orbit window `w_k` ([`inspiral_taper`](@ref)
at the harmonic's (2,2) frequency) and the transfer roll-off
`𝒯 = [1 + 0.6 (f/f★)²]^{-1/2}`. The 0.5PN harmonics are skipped at equal
mass, where their amplitudes vanish. Generic over `Real` (including
`ForwardDiff.Dual`); the single scalar core shared by the broadcast model,
the CPU loss loop and the GPU kernels.
"""
@inline function channel_strain_bin(f, distance, chirp_mass, coalescence_time,
    coalescence_phase, chi1, chi2, wp::WaveformParams)
    eta = wp.eta
    beta = spin_beta(chi1, chi2, eta)
    f_isco = isco_frequency(total_mass(chirp_mass, eta))
    transfer = 1 / sqrt(1 + 0.6 * (f / transfer_frequency(wp.arm_length))^2)
    geometry = wp.geometry
    hA, hE = harmonic_channel_strain(f, 2, distance, chirp_mass, coalescence_time,
        coalescence_phase, beta, f_isco, transfer, geometry, wp)
    if mass_asymmetry(eta) > 0
        for k in (1, 3)
            dA, dE = harmonic_channel_strain(f, k, distance, chirp_mass, coalescence_time,
                coalescence_phase, beta, f_isco, transfer, geometry, wp)
            hA += dA
            hE += dE
        end
    end
    return hA, hE
end

@inline function harmonic_channel_strain(f, k, distance, chirp_mass, coalescence_time,
    coalescence_phase, beta, f_isco, transfer, geometry, wp)
    F = 2 * f / k
    taper = inspiral_taper(F, f_isco, wp.cutoff_width)
    a_plus, a_cross = harmonic_amplitudes(f, k, chirp_mass, wp.eta, wp.inclination,
        distance)
    t_k = spa_time(F, chirp_mass, coalescence_time)
    # finite observation: the harmonic counts only while it is emitted inside
    # [0, observation_time]; without a window the factor is an exact one
    window =
        wp.observation_time > 0 ?
        taper * observation_window(t_k, wp.observation_time, wp.window_edge_time) :
        taper * one(t_k)
    FA_plus, FA_cross, FE_plus, FE_cross, cos_alpha, sin_alpha =
        channel_patterns(t_k, geometry, wp.orbit_phase)
    phase =
        doppler_phase(f, cos_alpha, sin_alpha, geometry) -
        harmonic_phase(f, k, chirp_mass, wp.eta, coalescence_time, coalescence_phase,
            beta)
    factor = (window * transfer) * cis(phase)
    return (a_plus * FA_plus - im * a_cross * FA_cross) * factor,
    (a_plus * FE_plus - im * a_cross * FE_cross) * factor
end

"""
$(TYPEDSIGNATURES)

Observed channel strains over `freqs` for the O(1)-scaled parameter vector
`theta = [D_L, 𝓜, t_c, φ_c, χ₁, χ₂]` (physical units restored through the
scales in `wp`), as `(A, E)` or `(A, E, T)` with `T` identically zero
(long-wavelength null channel) according to the channel count of `wp`. One
fused pass over the grid, one allocation per channel.
"""
function channel_strain(theta::AbstractVector, freqs::AbstractVector, wp::WaveformParams)
    distance = theta[1] * wp.distance_scale
    chirp_mass = theta[2] * wp.mass_scale
    coalescence_time = theta[3] * wp.time_scale
    hA1, _ = channel_strain_bin(freqs[1], distance, chirp_mass, coalescence_time,
        theta[4], theta[5], theta[6], wp)
    CT = typeof(hA1)
    A = Vector{CT}(undef, length(freqs))
    E = Vector{CT}(undef, length(freqs))
    @inbounds for i in eachindex(freqs)
        A[i], E[i] = channel_strain_bin(freqs[i], distance, chirp_mass, coalescence_time,
            theta[4], theta[5], theta[6], wp)
    end
    if n_channels(wp) == 3 # constant-folds: NCH is a type parameter
        return A, E, zeros(CT, length(freqs))
    end
    return A, E
end

end # module
