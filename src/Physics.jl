module Physics

export analytic_noise_psd, scaled_waveform_model

const L_ARM = 2.5e9
const C_LIGHT = 2.99792458e8
const F_STAR = C_LIGHT / (2 * π * L_ARM)

"""
    analytic_noise_psd(f)

Calculates an analytic noise Power Spectral Density (PSD) at frequency `f` typical for space-based interferometers.
This includes both instrumental noise approximations and confusion noise background.
"""
function analytic_noise_psd(f::Real)
    if f <= 0.0
        return 1e-30
    end
    
    # Optical Metrology Noise
    p_oms = (1.5e-11)^2 * (1 + (2e-3/f)^4)
    
    # Acceleration Noise
    p_acc = (3e-15)^2 * (1 + (0.4e-3/f)^2) * (1 + (f/8e-3)^4)
    
    # Total Instrumental Noise
    s_inst = (p_oms / L_ARM^2) + (2 * p_acc / ( (2*π*f)^4 * L_ARM^2 )) * (1 + cos(f/F_STAR)^2)

    # Galactic Binary Confusion Noise (approximate fit)
    A_gal, fk, B, C, D = 1.8e-44, 1.0e-4, 292.0, 10.0^(-3.5), 10.0^(-4.5)
    s_gal = A_gal * f^(-7/3) * exp(-(f/fk)^B) * (1 + tanh((C-f)/D))

    return s_inst + s_gal
end

"""
    scaled_waveform_model(theta, freq_grid)

Generates a frequency-domain inspiral waveform including spin-orbit coupling 
and the first sub-dominant higher harmonic (l=3, m=3).
Crucially, `theta` is an array of scaled parameters of O(1) to ensure the Fisher matrix 
and optimization landscape are well-conditioned.

Parameters in `theta`:
1. A_scaled   (O(1) value mapping to Amplitude ~ 1e-21)
2. Mc_scaled  (O(1) value mapping to Chirp Mass ~ 10 seconds)
3. tc_scaled  (O(1) value mapping to Coalescence Time ~ 1000 seconds)
4. phic       (Phase at coalescence, O(1) radians)
5. chi1       (Dimensionless spin of primary mass, [-1, 1])
6. chi2       (Dimensionless spin of secondary mass, [-1, 1])
"""
function scaled_waveform_model(theta::AbstractVector, freq_grid::AbstractVector; 
                               mass_scale::Real=10.0, time_scale::Real=1000.0, 
                               amp_scale::Real=1e-21, eta::Real=0.25, 
                               amp_33_factor::Real=0.1, kwargs...)
    A_scale, Mc_scale, tc_scale, phic, chi1, chi2 = theta
    
    # Restore physical units internally using configurable scales
    A = A_scale * amp_scale
    Mc = Mc_scale * mass_scale
    tc = tc_scale * time_scale
    
    # Effective spin parameter (simplified for aligned spins)
    chi_eff = 0.5 * (chi1 + chi2) # Exact for equal masses
    
    # --- Dominant Harmonic (l=2, m=2) ---
    amp_22 = @. A * (freq_grid ^ (-7/6))
    
    # Phase includes the leading order spin-orbit coupling term (1.5PN)
    # The term is proportional to beta = (113/3 - 76*eta/3) * chi_eff / 4
    beta = (113.0/3.0 - 76.0*eta/3.0) * chi_eff / 4.0
    v_param = @. (π * Mc * freq_grid)^(1/3)
    
    phase_22 = @. 2 * π * freq_grid * tc - phic - (3/128) * (v_param^(-5)) * (1.0 - 4.0 * beta * (v_param^3))
    
    h_22 = @. amp_22 * exp(1im * phase_22)
    
    # --- Higher Harmonic (l=3, m=3) ---
    amp_33 = @. (amp_33_factor * A) * (freq_grid ^ (-7/6)) * v_param
    
    # The frequency of the 33 mode is 1.5x the 22 mode frequency
    phase_33 = @. 1.5 * phase_22
    
    h_33 = @. amp_33 * exp(1im * phase_33)
    
    return h_22 .+ h_33
end

end # module