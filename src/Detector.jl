module Detector

export project_to_tdi, tdi_modulation

"""
    tdi_modulation(f, p; kwargs...)

Computes the frequency-dependent complex modulation factors for A, E, and T channels
due to the orbital motion of the detector (Doppler phase and antenna pattern amplitude).
"""
function tdi_modulation(f::Real, p::AbstractVector; 
                        mass_scale::Real=10.0, time_scale::Real=1000.0, 
                        sky_theta::Real=1.047, sky_phi::Real=0.0, 
                        inclination::Real=0.523, polarization::Real=0.0, kwargs...)
    Mc = p[2] * mass_scale
    tc = p[3] * time_scale
    
    R_sec = 499.00478383615643 # 1 AU in seconds
    omega_orbit = 2 * π / 3.15576e7 # 1 year orbital angular velocity
    
    # Stationary Phase Approximation (SPA) time-frequency relation
    v = (π * Mc * f)^(1/3)
    t_f = tc - 5.0 * Mc / (256.0 * v^8)
    
    phi_orb = omega_orbit * t_f
    doppler_phase = 2 * π * f * R_sec * sin(sky_theta) * cos(phi_orb - sky_phi)
    
    # Simplified low-frequency antenna patterns for A and E
    F_plus = 0.5 * (1 + cos(sky_theta)^2) * cos(2 * phi_orb) * cos(2 * polarization) - cos(sky_theta) * sin(2 * phi_orb) * sin(2 * polarization)
    F_cross = 0.5 * (1 + cos(sky_theta)^2) * cos(2 * phi_orb) * sin(2 * polarization) + cos(sky_theta) * sin(2 * phi_orb) * cos(2 * polarization)
    
    h_plus_amp = 0.5 * (1 + cos(inclination)^2)
    h_cross_amp = cos(inclination)
    
    phase_shift = exp(1im * doppler_phase)
    
    mod_A = sqrt(3/4) * (F_plus * h_plus_amp - 1im * F_cross * h_cross_amp) * phase_shift
    mod_E = sqrt(1/4) * (F_cross * h_plus_amp + 1im * F_plus * h_cross_amp) * phase_shift
    mod_T = zero(mod_A) # Null channel at low frequencies
    
    return mod_A, mod_E, mod_T
end

"""
    project_to_tdi(h_strain::AbstractVector, freqs::AbstractVector, p::AbstractVector; kwargs...) -> Tuple{AbstractVector, AbstractVector, AbstractVector}

Projects a raw, frequency-domain astrophysical strain \$h(f)\$ into the orthogonal `A`, `E`, and `T` Time Delay Interferometry (TDI) response channels.

It dynamically applies orbital modulation across the entire frequency grid by broadcasting `tdi_modulation`. This is mathematically equivalent to taking the raw signal and "flying" it through the detector's orbital path.

# Arguments
- `h_strain`: The raw complex astrophysical waveform array.
- `freqs`: The frequency grid.
- `p`: The 6-parameter source vector.

# Returns
- A tuple `(A_channel, E_channel, T_channel)` of the modulated complex response arrays.
"""
function project_to_tdi(h_strain::AbstractVector, freqs::AbstractVector, p::AbstractVector; kwargs...)
    mods = tdi_modulation.(freqs, Ref(p); kwargs...)
    mod_A = first.(mods)
    mod_E = getindex.(mods, 2)
    mod_T = last.(mods)
    
    return mod_A .* h_strain, mod_E .* h_strain, mod_T .* h_strain
end

end # module