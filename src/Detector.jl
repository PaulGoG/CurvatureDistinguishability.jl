"""
Low-frequency TDI A/E response: per-bin orbital-Doppler modulation and the
fused strain-to-channel projection.
"""
module Detector

using DocStringExtensions: TYPEDSIGNATURES
using ..Physics: WaveformParams, SECONDS_PER_YEAR

export tdi_modulation_bin, project_to_tdi, n_channels

const R_ORBIT_SEC = 499.00478383615643 # 1 AU in light-seconds

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

"""
$(TYPEDSIGNATURES)

Scalar per-bin complex modulation of the A and E TDI channels: orbital
Doppler phase and antenna patterns evaluated at the Newtonian SPA
time–frequency map `t(f) = t_c − 5𝓜/(256 v⁸)`, and the long-wavelength
transfer roll-off `[1 + 0.6 (f/f★)²]^{-1/2}` (Robson et al. 2019, Eq. 13;
`f★ = wp.transfer_frequency`). The antenna patterns are those of a 90°
detector rotating once per year in the ecliptic plane, evaluated at the
source azimuth `φ_orb − φ_sky` in the detector frame — the same azimuth
the Doppler term uses; A and E carry the identical `√3/2` normalisation
of the long-wavelength limit, E being A rotated by 45° in polarisation.
`chirp_mass`, `coalescence_time` in physical units. Generic over `Real`
(including `ForwardDiff.Dual`); safe inside GPU kernels.
"""
@inline function tdi_modulation_bin(f::Real, chirp_mass, coalescence_time,
    wp::WaveformParams)
    omega_orbit = 2 * π / SECONDS_PER_YEAR

    pn_velocity = (π * chirp_mass * f)^(1 / 3)
    t_f = coalescence_time - 5.0 * chirp_mass / (256.0 * pn_velocity^8)

    phi_orb = omega_orbit * t_f
    azimuth = phi_orb - wp.sky_phi # source azimuth in the rotating detector frame
    doppler_phase = 2 * π * f * R_ORBIT_SEC * sin(wp.sky_theta) * cos(azimuth)

    F_plus =
        0.5 * (1 + cos(wp.sky_theta)^2) * cos(2 * azimuth) * cos(2 * wp.polarization) -
        cos(wp.sky_theta) * sin(2 * azimuth) * sin(2 * wp.polarization)
    F_cross =
        0.5 * (1 + cos(wp.sky_theta)^2) * cos(2 * azimuth) * sin(2 * wp.polarization) +
        cos(wp.sky_theta) * sin(2 * azimuth) * cos(2 * wp.polarization)

    h_plus_amp = 0.5 * (1 + cos(wp.inclination)^2)
    h_cross_amp = cos(wp.inclination)

    # long-wavelength normalisation √3/2 of both channels, Doppler phase and
    # the finite-arm transfer roll-off
    response = sqrt(3 / 4) / sqrt(1 + 0.6 * (f / wp.transfer_frequency)^2)
    phase_shift = response * cis(doppler_phase)

    mod_A = (F_plus * h_plus_amp - 1im * F_cross * h_cross_amp) * phase_shift
    mod_E = (F_cross * h_plus_amp + 1im * F_plus * h_cross_amp) * phase_shift

    return mod_A, mod_E
end

"""
$(TYPEDSIGNATURES)

Project a frequency-domain strain into the TDI response channels in a single
fused pass (one loop, one allocation per channel). Returns `(A, E)` or
`(A, E, T)` depending on the channel count of `wp`; `T` is identically zero.
"""
function project_to_tdi(
    h_strain::AbstractVector,
    freqs::AbstractVector,
    p::AbstractVector,
    wp::WaveformParams,
)
    chirp_mass = p[2] * wp.mass_scale
    coalescence_time = p[3] * wp.time_scale

    mA1, _ = tdi_modulation_bin(freqs[1], chirp_mass, coalescence_time, wp)
    CT = typeof(mA1 * h_strain[1])
    A = Vector{CT}(undef, length(freqs))
    E = Vector{CT}(undef, length(freqs))
    @inbounds for i in eachindex(freqs, h_strain)
        mod_A, mod_E = tdi_modulation_bin(freqs[i], chirp_mass, coalescence_time, wp)
        A[i] = mod_A * h_strain[i]
        E[i] = mod_E * h_strain[i]
    end

    if n_channels(wp) == 3 # constant-folds: NCH is a type parameter
        return A, E, zeros(CT, length(freqs))
    end
    return A, E
end

end # module
