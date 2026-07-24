module Detector

using ..Physics: WaveformParams, waveform_params, SECONDS_PER_YEAR

export tdi_modulation_bin, project_to_tdi, n_channels

const R_ORBIT_SEC = 499.00478383615643 # 1 AU in light-seconds

"""
    n_channels(wp::WaveformParams)

Number of active TDI channels: 2 (A, E) by default, 3 when the identically
zero null channel T is explicitly requested via `wp.include_t_channel`.
"""
n_channels(wp::WaveformParams) = wp.include_t_channel ? 3 : 2

"""
    tdi_modulation_bin(f, Mc, tc, wp) -> (mod_A, mod_E)

Scalar per-bin complex modulation of the A and E TDI channels: orbital
Doppler phase (via the SPA time-frequency map `t(f) = t_c − 5M_c/(256 v⁸)`)
and low-frequency antenna patterns. `Mc`, `tc` in physical units. Generic
over `Real` (including `ForwardDiff.Dual`); safe inside GPU kernels.
"""
@inline function tdi_modulation_bin(f::Real, Mc, tc, wp::WaveformParams)
    omega_orbit = 2 * π / SECONDS_PER_YEAR

    v = (π * Mc * f)^(1 / 3)
    t_f = tc - 5.0 * Mc / (256.0 * v^8)

    phi_orb = omega_orbit * t_f
    doppler_phase = 2 * π * f * R_ORBIT_SEC * sin(wp.sky_theta) * cos(phi_orb - wp.sky_phi)

    F_plus = 0.5 * (1 + cos(wp.sky_theta)^2) * cos(2 * phi_orb) * cos(2 * wp.polarization) -
             cos(wp.sky_theta) * sin(2 * phi_orb) * sin(2 * wp.polarization)
    F_cross = 0.5 * (1 + cos(wp.sky_theta)^2) * cos(2 * phi_orb) * sin(2 * wp.polarization) +
              cos(wp.sky_theta) * sin(2 * phi_orb) * cos(2 * wp.polarization)

    h_plus_amp = 0.5 * (1 + cos(wp.inclination)^2)
    h_cross_amp = cos(wp.inclination)

    phase_shift = cis(doppler_phase)

    mod_A = sqrt(3 / 4) * (F_plus * h_plus_amp - 1im * F_cross * h_cross_amp) * phase_shift
    mod_E = sqrt(1 / 4) * (F_cross * h_plus_amp + 1im * F_plus * h_cross_amp) * phase_shift

    return mod_A, mod_E
end

"""
    project_to_tdi(h_strain, freqs, p, wp::WaveformParams)

Project a frequency-domain strain into the TDI response channels in a single
fused pass (one loop, one allocation per channel — instead of a
four-broadcast tuple-unpacking implementation). Returns `(A, E)` or
`(A, E, T)` depending on `wp.include_t_channel`; `T` is identically zero.
"""
function project_to_tdi(h_strain::AbstractVector, freqs::AbstractVector, p::AbstractVector, wp::WaveformParams)
    Mc = p[2] * wp.mass_scale
    tc = p[3] * wp.time_scale

    mA1, _ = tdi_modulation_bin(freqs[1], Mc, tc, wp)
    CT = typeof(mA1 * h_strain[1])
    A = Vector{CT}(undef, length(freqs))
    E = Vector{CT}(undef, length(freqs))
    @inbounds for i in eachindex(freqs, h_strain)
        mod_A, mod_E = tdi_modulation_bin(freqs[i], Mc, tc, wp)
        A[i] = mod_A * h_strain[i]
        E[i] = mod_E * h_strain[i]
    end

    if wp.include_t_channel
        return A, E, zeros(CT, length(freqs))
    end
    return A, E
end

end # module
