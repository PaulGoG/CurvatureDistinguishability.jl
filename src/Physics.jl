module Physics

export NoiseParams, robson_confusion_params, analytic_noise_psd,
       WaveformParams, waveform_params, spin_beta, strain_bin,
       scaled_waveform_model, SECONDS_PER_YEAR

const C_LIGHT = 2.99792458e8

"""
Seconds in one Julian year (365.25 d). Single source of truth for the
detector's orbital period and the default observation time.
"""
const SECONDS_PER_YEAR = 3.15576e7

# Robson, Cornish & Liu (2019), arXiv:1803.01944, Table 1: galactic
# confusion-noise fit coefficients (α, β, κ, γ, f_k) per mission duration.
const ROBSON_TABLE = (
    (tobs = 0.5 * SECONDS_PER_YEAR, alpha = 0.133, beta = 243.0, kappa = 482.0, gamma = 917.0, fk = 0.00258),
    (tobs = 1.0 * SECONDS_PER_YEAR, alpha = 0.171, beta = 292.0, kappa = 1020.0, gamma = 1680.0, fk = 0.00215),
    (tobs = 2.0 * SECONDS_PER_YEAR, alpha = 0.165, beta = 299.0, kappa = 611.0, gamma = 1340.0, fk = 0.00173),
    (tobs = 4.0 * SECONDS_PER_YEAR, alpha = 0.138, beta = -221.0, kappa = 521.0, gamma = 1680.0, fk = 0.00113),
)

"""
    NoiseParams(; kwargs...)

Coefficients of the analytic LISA noise model. The instrumental part is the
Michelson-channel PSD of Robson et al. (2019) Eq. 12 (strain-referred, *not*
sky-averaged — antenna response is applied explicitly by `Detector`), built
from the Eq. 10 optical-metrology noise `oms_amplitude` [m Hz⁻¹ᐟ²] with its
`oms_reddening_freq` [Hz] low-frequency term, the Eq. 11 acceleration noise
`acc_amplitude` [m s⁻² Hz⁻¹ᐟ²] with shoulders `acc_knee_low`/`acc_knee_high`
[Hz], and the `arm_length` [m] (which also sets the transfer frequency
`f★ = c/(2π L)`). The galactic confusion part is Eq. 14,
`S_c(f) = A f^{-7/3} e^{-f^α + β f sin(κf)} [1 + tanh(γ(f_k - f))]`,
defaulting to the 1-yr column of Table 1.

All fields are configurable through the `[noise]` section of the run configuration;
the defaults reproduce the published Robson et al. (2019) LISA model.
"""
Base.@kwdef struct NoiseParams
    confusion_enabled::Bool = true
    confusion_amp::Float64 = 9.0e-45
    confusion_alpha::Float64 = 0.171
    confusion_beta::Float64 = 292.0
    confusion_kappa::Float64 = 1020.0
    confusion_gamma::Float64 = 1680.0
    confusion_fk::Float64 = 0.00215
    arm_length::Float64 = 2.5e9
    oms_amplitude::Float64 = 1.5e-11
    oms_reddening_freq::Float64 = 2.0e-3
    acc_amplitude::Float64 = 3.0e-15
    acc_knee_low::Float64 = 0.4e-3
    acc_knee_high::Float64 = 8.0e-3
end

"""
    robson_confusion_params(T_obs; enabled = true, amp = 9.0e-45)

Build a [`NoiseParams`](@ref) whose confusion coefficients are the Robson
et al. (2019) Table 1 column nearest to the observation time `T_obs` [s].
"""
function robson_confusion_params(T_obs::Real; enabled::Bool = true, amp::Real = 9.0e-45)
    row = argmin(r -> abs(log(T_obs / r.tobs)), ROBSON_TABLE)
    return NoiseParams(confusion_enabled = enabled, confusion_amp = amp,
                       confusion_alpha = row.alpha, confusion_beta = row.beta,
                       confusion_kappa = row.kappa, confusion_gamma = row.gamma,
                       confusion_fk = row.fk)
end

"""
    analytic_noise_psd(f; noise = NoiseParams())

One-sided noise PSD at frequency `f` [Hz]: Robson et al. (2019) Eq. 12
instrumental noise plus the Eq. 14 galactic confusion fit (togglable via
`noise.confusion_enabled`). Returns a positive floor value for `f <= 0`.
"""
function analytic_noise_psd(f::Real; noise::NoiseParams = NoiseParams())
    if f <= 0.0
        return 1e-30
    end

    # Optical Metrology Noise (Robson Eq. 10)
    p_oms = noise.oms_amplitude^2 * (1 + (noise.oms_reddening_freq / f)^4)

    # Acceleration Noise (Robson Eq. 11)
    p_acc = noise.acc_amplitude^2 * (1 + (noise.acc_knee_low / f)^2) *
            (1 + (f / noise.acc_knee_high)^4)

    # Total Instrumental Noise (Robson Eq. 12); f★ = c/(2πL)
    L = noise.arm_length
    f_star = C_LIGHT / (2 * π * L)
    s_inst = (p_oms / L^2) + (2 * p_acc / ((2 * π * f)^4 * L^2)) * (1 + cos(f / f_star)^2)

    if !noise.confusion_enabled
        return s_inst
    end

    # Galactic binary confusion noise (Robson Eq. 14)
    s_gal = noise.confusion_amp * f^(-7 / 3) *
            exp(-(f^noise.confusion_alpha) + noise.confusion_beta * f * sin(noise.confusion_kappa * f)) *
            (1 + tanh(noise.confusion_gamma * (noise.confusion_fk - f)))

    return s_inst + s_gal
end

"""
    WaveformParams(; kwargs...)

Immutable, isbits container for every physical parameter of the waveform and
detector-response model; the single source of parameter defaults, safe to
pass into GPU kernels. Values are overridden by the `[physics]` section of the run
configuration. The active channel count (2, or 3 with the identically zero
T channel) is carried as the type parameter `NCH`, so channel-dependent
tuple types are inferable throughout the geometry and inference paths.
"""
struct WaveformParams{T<:Real,NCH}
    mass_scale::T
    time_scale::T
    amp_scale::T
    eta::T
    amp_33_factor::T
    sky_theta::T
    sky_phi::T
    inclination::T
    polarization::T
    include_t_channel::Bool
end

function WaveformParams(; mass_scale::Real = 10.0, time_scale::Real = 1000.0,
                        amp_scale::Real = 1e-21, eta::Real = 0.25,
                        amp_33_factor::Real = 0.1, sky_theta::Real = 1.047,
                        sky_phi::Real = 0.0, inclination::Real = 0.523,
                        polarization::Real = 0.0, include_t_channel::Bool = false)
    fields = promote(float(mass_scale), float(time_scale), float(amp_scale),
                     float(eta), float(amp_33_factor), float(sky_theta),
                     float(sky_phi), float(inclination), float(polarization))
    return WaveformParams{typeof(fields[1]),include_t_channel ? 3 : 2}(fields...,
                                                                       include_t_channel)
end

"""
    waveform_params(; kwargs...) -> WaveformParams

Build a [`WaveformParams`](@ref) from keyword arguments, silently ignoring
any keys that are not fields (so pipeline call sites can splat a mixed
configuration NamedTuple through the keyword APIs).
"""
function waveform_params(; kwargs...)
    known = filter(p -> first(p) in fieldnames(WaveformParams), pairs(kwargs))
    return WaveformParams(; known...)
end

"""
    spin_beta(chi1, chi2, eta)

Leading-order (1.5PN) spin-orbit phase coefficient
`β = (113/3 − 76η/3) χ_eff / 4` with `χ_eff = (χ₁ + χ₂)/2`.
"""
@inline function spin_beta(chi1, chi2, eta)
    chi_eff = 0.5 * (chi1 + chi2)
    return (113.0 / 3.0 - 76.0 * eta / 3.0) * chi_eff / 4.0
end

"""
    strain_bin(f, A, Mc, tc, phic, beta, amp_33_factor)

Scalar per-bin frequency-domain strain: dominant (2,2) mode with 1.5PN
spin-orbit phasing plus the (3,3) harmonic at Newtonian phase ratio
`Ψ₃₃ = 1.5 Ψ₂₂`. `A`, `Mc`, `tc` are in physical units (s-based geometrised
units for `Mc`, `tc`). This is the single scalar core shared by the broadcast
model, the CPU inference loop and the GPU kernel — generic over `Real`
(including `ForwardDiff.Dual`).
"""
@inline function strain_bin(f::Real, A, Mc, tc, phic, beta, amp_33_factor)
    v_param = (π * Mc * f)^(1 / 3)

    amp_22 = A * (f^(-7 / 6))
    phase_22 = 2 * π * f * tc - phic - (3 / 128) * (v_param^(-5)) * (1.0 - 4.0 * beta * (v_param^3))
    h_22 = amp_22 * cis(phase_22)

    amp_33 = (amp_33_factor * A) * (f^(-7 / 6)) * v_param
    h_33 = amp_33 * cis(1.5 * phase_22)

    return h_22 + h_33
end

"""
    scaled_waveform_model(theta, freq_grid, wp::WaveformParams)

Frequency-domain inspiral waveform over `freq_grid` for the O(1)-scaled
6-parameter vector `theta = [A, M_c, t_c, φ_c, χ₁, χ₂]`. Physical units are
restored internally via the scales in `wp`. Broadcasts [`strain_bin`](@ref).
"""
function scaled_waveform_model(theta::AbstractVector, freq_grid::AbstractVector, wp::WaveformParams)
    A = theta[1] * wp.amp_scale
    Mc = theta[2] * wp.mass_scale
    tc = theta[3] * wp.time_scale
    phic = theta[4]
    beta = spin_beta(theta[5], theta[6], wp.eta)
    return strain_bin.(freq_grid, A, Mc, tc, phic, beta, wp.amp_33_factor)
end

end # module
