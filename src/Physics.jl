"""
Analytic LISA noise model (Robson et al. 2019, Eq. 10–14, Table 1) and the
post-Newtonian inspiral model shared by the CPU loss loop and the GPU
kernels: TaylorF2 phasing for aligned spins and unequal masses to 1.5PN
order or, on request, to any order through 4.5PN, 0.5PN-amplitude harmonics,
an innermost-stable-orbit taper and the physical source parameters (luminosity
distance, chirp mass, coalescence time and phase, aligned spins).
"""
module Physics

using DocStringExtensions: TYPEDSIGNATURES
export NoiseParams, robson_confusion_params, analytic_noise_psd, sky_averaged_response,
    WaveformParams, waveform_params, spin_beta, spin_sigma, PhaseCoefficients,
    phase_coefficients, pn_phase, harmonic_phase,
    harmonic_amplitudes, total_mass, isco_frequency, second_source,
    SECONDS_PER_YEAR
public N_PARAMS, LISA_ARM_LENGTH, GIGAPARSEC_SEC, HARMONICS, C_LIGHT, R_ORBIT_SEC,
    transfer_frequency, mass_asymmetry, phase_coefficient_2pn, spa_time, inspiral_taper,
    observation_window, ResponseGeometry, PHASE_PN_ORDERS

"""
Speed of light [m s⁻¹]; converts the arm length and the distance unit to
light-seconds.
"""
const C_LIGHT = 2.99792458e8

"""
LISA arm length [m] (Robson et al. 2019); the single default behind the
instrumental noise (`NoiseParams.arm_length`) and the response transfer
frequency of `WaveformParams`.
"""
const LISA_ARM_LENGTH = 2.5e9

"""
$(TYPEDSIGNATURES)

Transfer frequency `f★ = c/(2π L)` [Hz] of an arm of length `L` [m]: the
scale above which the long-wavelength response rolls off (Robson et al.
2019, Eq. 13).
"""
transfer_frequency(arm_length::Real) = C_LIGHT / (2 * π * arm_length)

"""
$(TYPEDSIGNATURES)

Sky- and polarization-averaged signal response of the two low-frequency LISA
channels, Robson et al. (2019) Eq. 9: `R(f) = (3/10) / (1 + 0.6 (f/f★)²)`
at `f` [Hz], with `f★ = c/(2π L)` from `arm_length` [m]. It relates a two-channel
sensitivity-level spectral density to the per-channel PSD (their Eq. 2,
`S_n = P_n / R`), and its roll-off is the square of the transfer factor
`Detector` applies to every harmonic. Used by [`analytic_noise_psd`](@ref)
to bring the Eq. 14 confusion fit to the level of the Eq. 12 channel noise.
"""
function sky_averaged_response(f::Real, arm_length::Real)
    return (3 / 10) / (1 + 0.6 * (f / transfer_frequency(arm_length))^2)
end

"""
One gigaparsec in light-seconds, the default luminosity-distance unit.
"""
const GIGAPARSEC_SEC = 3.0856775814913673e25 / C_LIGHT

"""
Solar mass in seconds, G M⊙/c³, from the nominal solar mass parameter
(GM)⊙ = 1.3271244 × 10²⁰ m³ s⁻² of IAU 2015 Resolution B3 (Prša et al. 2016,
doi:10.3847/0004-6256/152/2/41). Used to quote chirp masses in solar masses on
figure axes; the computation keeps masses in seconds.
"""
const SOLAR_MASS_SEC = 1.3271244e20 / C_LIGHT^3

"""
Orbital radius of the constellation center, 1 AU in light-seconds.
"""
const R_ORBIT_SEC = 499.00478383615643

"""
Harmonics of the orbital phase carried by the waveform: `k = 2` at leading
amplitude order, `k = 1` and `k = 3` at 0.5PN amplitude order (both
proportional to the mass asymmetry).
"""
const HARMONICS = (1, 2, 3)

"""
Dimension of the waveform parameter vector `θ = (D_L, 𝓜, t_c, φ_c, χ₁, χ₂)`.
Every parameter-count-dependent structure (bounds tuples, kernel parameter
tuples, work-item validation) is sized from it.
"""
const N_PARAMS = 6

"""
Post-Newtonian orders of the phase that `phase_pn_order` accepts. Each order adds
the terms of the TaylorF2 phase at that order to those of the orders below it;
1.5 is the default.
"""
const PHASE_PN_ORDERS = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5)

# positive PSD floor returned for non-positive frequencies, so the noise
# weighting never divides by zero
const PSD_FLOOR = 1e-30

"""
Seconds in one Julian year (365.25 d). The detector's orbital period and the
default observation time are both taken from it.
"""
const SECONDS_PER_YEAR = 3.15576e7

# Robson, Cornish & Liu (2019), arXiv:1803.01944, Table 1: galactic
# confusion-noise fit coefficients (α, β, κ, γ, f_k) per mission duration.
const ROBSON_TABLE = (
    (
        T_obs = 0.5 * SECONDS_PER_YEAR,
        alpha = 0.133,
        beta = 243.0,
        kappa = 482.0,
        gamma = 917.0,
        f_knee = 0.00258,
    ),
    (
        T_obs = 1.0 * SECONDS_PER_YEAR,
        alpha = 0.171,
        beta = 292.0,
        kappa = 1020.0,
        gamma = 1680.0,
        f_knee = 0.00215,
    ),
    (
        T_obs = 2.0 * SECONDS_PER_YEAR,
        alpha = 0.165,
        beta = 299.0,
        kappa = 611.0,
        gamma = 1340.0,
        f_knee = 0.00173,
    ),
    (
        T_obs = 4.0 * SECONDS_PER_YEAR,
        alpha = 0.138,
        beta = -221.0,
        kappa = 521.0,
        gamma = 1680.0,
        f_knee = 0.00113,
    ),
)

# Robson et al. (2019) Eq. 14 galactic-confusion amplitude A_c [Hz^-1]
const GALACTIC_CONFUSION_AMP = 9.0e-45

"""
    NoiseParams(; kwargs...)

Coefficients of the analytic LISA noise model. The instrumental part is the
Michelson-channel PSD of Robson et al. (2019) Eq. 12 (strain-referred and not
sky-averaged; the antenna response is applied explicitly by `Detector`), built
from the Eq. 10 optical-metrology noise `oms_amplitude` [m Hz⁻¹ᐟ²] with its
`oms_reddening_freq` [Hz] low-frequency term, the Eq. 11 acceleration noise
`acc_amplitude` [m s⁻² Hz⁻¹ᐟ²] with shoulders `acc_knee_low`/`acc_knee_high`
[Hz], and the `arm_length` [m] (which also sets the transfer frequency
`f★ = c/(2π L)`). The low-frequency factor `1 + (f_red/f)⁴` of the
optical-metrology noise and the factor `1 + (f/f_high)⁴` of the acceleration noise
are those of the arXiv version of the paper (arXiv:1803.01944, Eqs. 10–11); the
journal version prints the two equations without them. The galactic confusion part
is Eq. 14,
`S_c(f) = A f^{-7/3} e^{-f^α + β f sin(κf)} [1 + tanh(γ(f_k - f))]`,
defaulting to the 1-yr column of Table 1. Eq. 14 is calibrated on the
two-channel sensitivity curve (their Eq. 1, `S_n = P_n/R + S_c`), so
[`analytic_noise_psd`](@ref) multiplies it by the Eq. 9 response
[`sky_averaged_response`](@ref) before adding it to the channel noise.

All fields are configurable through the `[noise]` section of the run configuration;
the defaults reproduce the Robson et al. (2019) LISA model in its arXiv form.
"""
Base.@kwdef struct NoiseParams
    confusion_enabled::Bool = true
    confusion_amp::Float64 = GALACTIC_CONFUSION_AMP
    confusion_alpha::Float64 = 0.171
    confusion_beta::Float64 = 292.0
    confusion_kappa::Float64 = 1020.0
    confusion_gamma::Float64 = 1680.0
    confusion_knee_freq::Float64 = 0.00215
    arm_length::Float64 = LISA_ARM_LENGTH
    oms_amplitude::Float64 = 1.5e-11
    oms_reddening_freq::Float64 = 2.0e-3
    acc_amplitude::Float64 = 3.0e-15
    acc_knee_low::Float64 = 0.4e-3
    acc_knee_high::Float64 = 8.0e-3
end

"""
$(TYPEDSIGNATURES)

Build a [`NoiseParams`](@ref) whose confusion coefficients are the Robson
et al. (2019) Table 1 column nearest in `log T_obs` to the observation time `T_obs` [s].
"""
function robson_confusion_params(T_obs::Real; enabled::Bool = true,
    confusion_amp::Real = GALACTIC_CONFUSION_AMP)
    row = argmin(r -> abs(log(T_obs / r.T_obs)), ROBSON_TABLE)
    return NoiseParams(confusion_enabled = enabled, confusion_amp = confusion_amp,
        confusion_alpha = row.alpha, confusion_beta = row.beta,
        confusion_kappa = row.kappa, confusion_gamma = row.gamma,
        confusion_knee_freq = row.f_knee)
end

"""
$(TYPEDSIGNATURES)

One-sided noise PSD [Hz⁻¹] of one channel at frequency `f` [Hz]: the Robson et
al. (2019) Eq. 12 instrumental noise `P_n(f)` plus the Eq. 14 galactic
confusion fit `S_c(f)` brought to the same level by the Eq. 9 response,
`P_n(f) + R(f) S_c(f)` (togglable via `noise.confusion_enabled`). Dividing
the result by `R(f)` reproduces their sensitivity curve, Eq. 13 + Eq. 14,
so the sky- and polarization-averaged A + E signal-to-noise ratio of the
explicit response equals the one computed from that curve. Returns a
positive floor value for `f <= 0`.
"""
function analytic_noise_psd(f::Real; noise::NoiseParams = NoiseParams())
    if f <= 0.0
        return PSD_FLOOR
    end

    # optical metrology noise (Robson Eq. 10)
    p_oms = noise.oms_amplitude^2 * (1 + (noise.oms_reddening_freq / f)^4)

    # acceleration noise (Robson Eq. 11)
    p_acc =
        noise.acc_amplitude^2 * (1 + (noise.acc_knee_low / f)^2) *
        (1 + (f / noise.acc_knee_high)^4)

    # total instrumental noise (Robson Eq. 12); f★ = c/(2πL)
    L = noise.arm_length
    f_star = C_LIGHT / (2 * π * L)
    s_inst = (p_oms / L^2) + (2 * p_acc / ((2 * π * f)^4 * L^2)) * (1 + cos(f / f_star)^2)

    if !noise.confusion_enabled
        return s_inst
    end

    # galactic binary confusion noise (Robson Eq. 14): a two-channel
    # sensitivity-level fit, converted to the per-channel PSD by Eq. 9
    s_gal =
        noise.confusion_amp * f^(-7 / 3) *
        exp(
            -(f^noise.confusion_alpha) +
            noise.confusion_beta * f * sin(noise.confusion_kappa * f),
        ) *
        (1 + tanh(noise.confusion_gamma * (noise.confusion_knee_freq - f)))

    return s_inst + sky_averaged_response(f, L) * s_gal
end

"""
Constellation and source constants the per-bin response needs, derived once
from a [`WaveformParams`](@ref): the orbital eccentricity `e = L/(2√3 R)` of
the analytic equal-arm orbits, the cosines and sines of the three spacecraft
constellation phases `β_n = 2π(n−1)/3 + λ_c` (and of `2β_n`), the direction
`n̂` to the source, and the wave-frame polarization vectors `(p, q)`.
"""
struct ResponseGeometry{T<:Real}
    eccentricity::T
    cos_beta::NTuple{3,T}
    sin_beta::NTuple{3,T}
    cos_2beta::NTuple{3,T}
    sin_2beta::NTuple{3,T}
    source_direction::NTuple{3,T}
    p::NTuple{3,T}
    q::NTuple{3,T}
end

"""
    WaveformParams(; kwargs...)

Immutable, isbits container for every fixed parameter of the waveform and
detector-response model, safe to pass into GPU kernels. Parameter units:
`mass_scale` (chirp mass, geometrized seconds), `time_scale` (coalescence
time, s) and `distance_scale` (luminosity distance, s; default one
gigaparsec) restore the physical values of the O(1) parameter vector. The
symmetric mass ratio `eta` ∈ (0, 1/4] is fixed per configuration (the
heavier body carries `chi1`). The source sits at ecliptic longitude
`ecliptic_longitude` and latitude `ecliptic_latitude` [rad], with
inclination `inclination` [rad] (angle between the propagation direction and the
orbital angular momentum) and polarization angle `polarization` [rad] of the wave-frame
vectors of the LISA science ground segment conventions (Baghi et al. 2026). The
polarization amplitudes are those of Blanchet et al. (1996), whose vectors lie along
and across the line of nodes, a quarter turn from those of the conventions: the
polarization angle of the conventions is `polarization + π/2`. `orbit_phase` and
`constellation_phase` are the orbital and cartwheel phases [rad] of the constellation
at `t = 0`, `arm_length` [m] sets the orbital eccentricity of the spacecraft
and the transfer frequency, and `cutoff_width` is the relative width of the
innermost-stable-orbit taper. `observation_time` [s] is the duration of the
observation that starts at `t = 0`: when positive, every harmonic is weighted
by [`observation_window`](@ref) at its stationary-phase emission time, with
edges of time scale `window_edge_time` [s]; at the default `0` the signal
fills the whole frequency band whenever it was emitted. `phase_pn_order` is the
post-Newtonian order of the phase, one of [`PHASE_PN_ORDERS`](@ref) (default `1.5`);
the coefficients are those of [`phase_coefficients`](@ref). With `pn_spa_time`
the stationary-phase times at which the response is evaluated follow from that
phase ([`spa_time`](@ref) with coefficients); by default they are Newtonian. The
active channel count (2, or 3 with the identically zero T channel, requested through
the `include_t_channel` keyword) is carried only as the type parameter `NCH`.
`geometry` holds the derived response constants.
"""
struct WaveformParams{T<:Real,NCH}
    mass_scale::T
    time_scale::T
    distance_scale::T
    eta::T
    ecliptic_longitude::T
    ecliptic_latitude::T
    inclination::T
    polarization::T
    orbit_phase::T
    constellation_phase::T
    arm_length::T
    cutoff_width::T
    observation_time::T
    window_edge_time::T
    phase_pn_order::T
    geometry::ResponseGeometry{T}
    pn_spa_time::Bool
end

"""
$(TYPEDSIGNATURES)

Response constants of a source and constellation (see
[`ResponseGeometry`](@ref)). The reference polarization vectors of the
ecliptic frame are `u = −e_λ` and `v = e_β`; the wave-frame vectors are
their rotation by the polarization angle about the propagation direction,
`p = u cos ψ + v sin ψ`, `q = −u sin ψ + v cos ψ`, so that `(p, q, k)` with
`k = −n̂` is right-handed and `e⁺ = p⊗p − q⊗q`, `e× = p⊗q + q⊗p`.
"""
function response_geometry(ecliptic_longitude::T, ecliptic_latitude::T, polarization::T,
    constellation_phase::T, arm_length::T) where {T<:Real}
    λ, β, ψ = ecliptic_longitude, ecliptic_latitude, polarization
    u = (sin(λ), -cos(λ), zero(T))
    v = (-sin(β) * cos(λ), -sin(β) * sin(λ), cos(β))
    p = ntuple(i -> u[i] * cos(ψ) + v[i] * sin(ψ), Val(3))
    q = ntuple(i -> -u[i] * sin(ψ) + v[i] * cos(ψ), Val(3))
    n = (cos(β) * cos(λ), cos(β) * sin(λ), sin(β))
    betas = ntuple(k -> 2 * T(π) * (k - 1) / 3 + constellation_phase, Val(3))
    eccentricity = arm_length / C_LIGHT / (2 * sqrt(T(3)) * R_ORBIT_SEC)
    return ResponseGeometry(eccentricity, cos.(betas), sin.(betas), cos.(2 .* betas),
        sin.(2 .* betas), n, p, q)
end

function WaveformParams(; mass_scale::Real = 1.0, time_scale::Real = 1.0e6,
    distance_scale::Real = GIGAPARSEC_SEC, eta::Real = 0.25,
    ecliptic_longitude::Real = 0.0, ecliptic_latitude::Real = π / 6,
    inclination::Real = π / 6, polarization::Real = 0.0,
    orbit_phase::Real = 0.0, constellation_phase::Real = 0.0,
    arm_length::Real = LISA_ARM_LENGTH, cutoff_width::Real = 0.1,
    observation_time::Real = 0.0, window_edge_time::Real = 1.0e6,
    phase_pn_order::Real = 1.5, pn_spa_time::Bool = false,
    include_t_channel::Bool = false)
    phase_pn_order in PHASE_PN_ORDERS || throw(
        ArgumentError(
            "phase_pn_order must be one of $(join(PHASE_PN_ORDERS, ", ")), " *
            "got $phase_pn_order",
        ),
    )
    fields = promote(float(mass_scale), float(time_scale), float(distance_scale),
        float(eta), float(ecliptic_longitude), float(ecliptic_latitude),
        float(inclination), float(polarization), float(orbit_phase),
        float(constellation_phase), float(arm_length), float(cutoff_width),
        float(observation_time), float(window_edge_time), float(phase_pn_order))
    T = typeof(fields[1])
    geometry = response_geometry(fields[5], fields[6], fields[8], fields[10], fields[11])
    return WaveformParams{T,include_t_channel ? 3 : 2}(fields..., geometry, pn_spa_time)
end

# keywords accepted by the WaveformParams constructor: every stored parameter
# plus the channel-count switch, which lives in the type parameter
const WAVEFORM_KEYWORDS = (
    :mass_scale, :time_scale, :distance_scale, :eta, :ecliptic_longitude,
    :ecliptic_latitude, :inclination, :polarization, :orbit_phase,
    :constellation_phase, :arm_length, :cutoff_width, :observation_time,
    :window_edge_time, :phase_pn_order, :pn_spa_time, :include_t_channel,
)

"""
$(TYPEDSIGNATURES)

Build a [`WaveformParams`](@ref) from keyword arguments, rejecting unknown
keys with an `ArgumentError` naming the offending keyword.
"""
function waveform_params(; kwargs...)
    unknown = setdiff(keys(kwargs), WAVEFORM_KEYWORDS)
    isempty(unknown) || throw(
        ArgumentError(
            "Unknown waveform parameter(s): $(join(unknown, ", ")). Valid keys: " *
            "$(join(WAVEFORM_KEYWORDS, ", ")).",
        ),
    )
    return WaveformParams(; kwargs...)
end

"""
$(TYPEDSIGNATURES)

Total mass `M = 𝓜 η^{-3/5}` from the chirp mass and the symmetric mass ratio.
`M` is in the unit of `chirp_mass` ([s] throughout the package).
"""
@inline total_mass(chirp_mass, eta) = chirp_mass * eta^(-3 / 5)

"""
$(TYPEDSIGNATURES)

Mass asymmetry `δ = (m₁ − m₂)/M = √(1 − 4η)` (non-negative: `m₁ ≥ m₂`).
"""
@inline mass_asymmetry(eta) = sqrt(max(1 - 4 * eta, zero(eta)))

"""
$(TYPEDSIGNATURES)

Gravitational-wave frequency [Hz] of the innermost stable circular orbit of a
Schwarzschild binary of total mass `total_mass` [s]: `1/(6^{3/2} π M)`.

```jldoctest
julia> round(isco_frequency(4.925) * 1e3; digits = 3) # M = 10⁶ M⊙, in mHz
4.398
```
"""
@inline isco_frequency(total_mass) = 1 / (6 * sqrt(6.0) * π * total_mass)

"""
$(TYPEDSIGNATURES)

1.5PN spin–orbit phase coefficient of Poisson & Will (1995) for aligned
spins, `β = (1/12) Σᵢ [113 (mᵢ/M)² + 75η] χᵢ`, written with the symmetric and
antisymmetric spins `χ_s = (χ₁ + χ₂)/2`, `χ_a = (χ₁ − χ₂)/2` and the mass
asymmetry `δ`: `β = [(113 − 76η) χ_s + 113 δ χ_a]/12`. `chi1` is the spin
of the heavier body.

```jldoctest
julia> spin_beta(0.5, 0.3, 0.25)
3.1333333333333333
```
"""
@inline function spin_beta(chi1, chi2, eta)
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    return ((113.0 - 76.0 * eta) * chi_s + 113.0 * mass_asymmetry(eta) * chi_a) / 12.0
end

"""
$(TYPEDSIGNATURES)

2PN spin–spin phase coefficient for black holes with aligned spins,
`σ = (79/8) η χ₁χ₂ + (81/16) Σᵢ (mᵢ/M)² χᵢ²`: the spin1–spin2 coupling plus,
for each body, the quadrupole–monopole and self-spin terms (Arun et al. 2009,
with its erratum; Khan et al. 2016, Appendix B). Written with the symmetric
and antisymmetric spins and the mass asymmetry `δ`:
`σ = (81/16 − η/4) χ_s² + (81/16 − 20η) χ_a² + (81/8) δ χ_s χ_a`.
`chi1` is the spin of the heavier body.

```jldoctest
julia> spin_sigma(0.5, 0.5, 0.25)
1.25
```
"""
@inline function spin_sigma(chi1, chi2, eta)
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    return (81 / 16 - eta / 4) * chi_s^2 + (81 / 16 - 20 * eta) * chi_a^2 +
           (81 / 8) * mass_asymmetry(eta) * chi_s * chi_a
end

"""
$(TYPEDSIGNATURES)

2PN coefficient of the TaylorF2 phase bracket,
`ψ₄ = 15293365/508032 + 27145η/504 + 3085η²/72 − 10σ`, with the spin–spin
coefficient `sigma` from [`spin_sigma`](@ref) (Poisson & Will 1995;
Arun et al. 2009).
"""
@inline function phase_coefficient_2pn(eta, sigma)
    return 15293365 / 508032 + 27145 * eta / 504 + 3085 * eta^2 / 72 - 10 * sigma
end

"""
    PhaseCoefficients{T}

Coefficients of the bracket of the TaylorF2 phase,
`ψ_PN(F) = (3/(128 η v⁵)) [1 + p2 v² + p3 v³ + p4 v⁴ + (p5 + p5L ln v) v⁵ +
(p6 + p6L ln v) v⁶ + p7 v⁷ + (p8L ln v + p8L2 ln²v) v⁸ + (p9 + p9L ln v) v⁹]`,
`v = (π M F)^{1/3}`, from [`phase_coefficients`](@ref). Isbits for real `T`, so it
passes into GPU kernels.
"""
struct PhaseCoefficients{T<:Real}
    p2::T
    p3::T
    p4::T
    p5::T
    p5L::T
    p6::T
    p6L::T
    p7::T
    p8L::T
    p8L2::T
    p9::T
    p9L::T
end

"""
$(TYPEDSIGNATURES)

2.5PN coefficient `p5` of the TaylorF2 phase bracket ([`PhaseCoefficients`](@ref))
for symmetric mass ratio `eta`, mass asymmetry `delta` and the symmetric and
antisymmetric spins `chi_s`, `chi_a`: the non-spinning term (Buonanno et al. 2009,
doi:10.1103/PhysRevD.80.084043; Blanchet 2024, doi:10.1007/s41114-024-00050-z) and the
next-to-leading-order spin–orbit term (Arun et al. 2009, doi:10.1103/PhysRevD.79.104023,
with its erratum, doi:10.1103/PhysRevD.84.049901). The coefficient of `v⁵ ln v` is
`p5L = 3 p5`.
"""
@inline function phase_coefficient_25pn(eta, delta, chi_s, chi_a)
    return (
        38645 * π / 756 - 65 * π * eta / 9
        + (-732985 / 2268 + 24260 * eta / 81 + 340 * eta^2 / 9) * chi_s
        + (-732985 / 2268 - 140 * eta / 9) * delta * chi_a
    )
end

"""
$(TYPEDSIGNATURES)

3PN coefficients `(p6, p6L)` of the TaylorF2 phase bracket
([`PhaseCoefficients`](@ref)), arguments as for [`phase_coefficient_25pn`](@ref): the
non-spinning terms and their logarithm (Buonanno et al. 2009,
doi:10.1103/PhysRevD.80.084043; Blanchet 2024, doi:10.1007/s41114-024-00050-z), the
leading spin–orbit tail (Bohé, Marsat & Blanchet 2013,
doi:10.1088/0264-9381/30/13/135009) and the next-to-leading-order spin–spin terms
(Bohé, Faye, Marsat & Porter 2015, doi:10.1088/0264-9381/32/19/195010).
"""
@inline function phase_coefficients_3pn(eta, delta, chi_s, chi_a)
    p6 = (
        11583231236531 / 4694215680 - 15737765635 * eta / 3048192
        + 76055 * eta^2 / 1728 - 127825 * eta^3 / 1296
        -
        13696 * log(2.0) / 21 - 6848 * Base.MathConstants.eulergamma / 21
        -
        640 * π^2 / 3 + 2255 * π^2 * eta / 12
        + (2270 * π / 3 - 520 * π * eta) * chi_s
        + (2270 * π / 3) * delta * chi_a
        + (75515 / 288 - 232415 * eta / 504 + 1255 * eta^2 / 9) * chi_s^2
        + (75515 / 144 - 8225 * eta / 18) * delta * chi_s * chi_a
        + (75515 / 288 - 263245 * eta / 252 - 480 * eta^2) * chi_a^2
    )
    p6L = -6848 / 21
    return p6, p6L
end

"""
$(TYPEDSIGNATURES)

3.5PN coefficient `p7` of the TaylorF2 phase bracket ([`PhaseCoefficients`](@ref)),
arguments as for [`phase_coefficient_25pn`](@ref): the non-spinning term (Buonanno et
al. 2009, doi:10.1103/PhysRevD.80.084043; Blanchet 2024,
doi:10.1007/s41114-024-00050-z), the next-to-next-to-leading-order spin–orbit terms
(Bohé, Marsat, Faye & Blanchet 2013, doi:10.1088/0264-9381/30/7/075017; Bohé, Marsat &
Blanchet 2013, doi:10.1088/0264-9381/30/13/135009), the spin–spin tail (Cho, Porto &
Yang 2022, doi:10.1103/PhysRevD.106.L101501) and the leading cubic-in-spin terms
(Marsat 2015, doi:10.1088/0264-9381/32/8/085008).
"""
@inline function phase_coefficient_35pn(eta, delta, chi_s, chi_a)
    return (
        77096675 * π / 254016 + 378515 * π * eta / 1512 - 74045 * π * eta^2 / 756
        +
        (
            -25150083775 / 3048192 + 10566655595 * eta / 762048
            -
            1042165 * eta^2 / 3024 + 5345 * eta^3 / 36
        ) * chi_s
        +
        (
            -25150083775 / 3048192 + 26804935 * eta / 6048 - 1985 * eta^2 / 48
        ) * delta * chi_a
        + (-815 * π / 2 + 30 * π * eta) * chi_s^2
        + (-815 * π) * delta * chi_s * chi_a
        + (-815 * π / 2 + 1600 * π * eta) * chi_a^2
        + (14585 / 24 - 475 * eta / 6 + 100 * eta^2 / 3) * chi_s^3
        + (14585 / 8 - 215 * eta / 2) * delta * chi_s^2 * chi_a
        + (14585 / 8 - 7270 * eta + 80 * eta^2) * chi_s * chi_a^2
        + (14585 / 24 - 2380 * eta) * delta * chi_a^3
    )
end

"""
$(TYPEDSIGNATURES)

4PN coefficients `(p8L, p8L2)` of `v⁸ ln v` and `v⁸ ln²v` in the TaylorF2 phase
bracket ([`PhaseCoefficients`](@ref)), arguments as for
[`phase_coefficient_25pn`](@ref): the non-spinning terms (Blanchet, Faye, Henry,
Larrouturou & Trestini 2023, doi:10.1103/PhysRevLett.131.121402; Blanchet 2024,
doi:10.1007/s41114-024-00050-z), the next-to-leading-order spin–orbit tail (Marsat et
al. 2014, doi:10.1088/0264-9381/31/2/025023), the next-to-next-to-leading-order
spin–spin terms (Cho, Porto & Yang 2022, doi:10.1103/PhysRevD.106.L101501) and the
leading quartic-in-spin terms for black holes (Siemonsen, Steinhoff & Vines 2018,
doi:10.1103/PhysRevD.97.124046). These papers give the binding energy and the flux;
the spin terms of `p8L` follow from them by the stationary-phase reduction, as
they are not printed in frequency-domain form there. The spin–spin flux at this
order rests on a single computation (Cho, Porto & Yang 2022). The constant 4PN
coefficient is a shift of the coalescence time and is not carried.
"""
@inline function phase_coefficients_4pn(eta, delta, chi_s, chi_a)
    p8L = (
        -2550713843998885153 / 276808510218240 + 680712846248317 * eta / 42247941120
        -
        7510073635 * eta^2 / 3048192 - 1292395 * eta^3 / 12096 + 5975 * eta^4 / 96
        + 78975 * log(3.0) / 196 - 78975 * log(3.0) * eta / 49
        + 1011020 * log(2.0) / 1323 + 9964112 * log(2.0) * eta / 1323
        + 36812 * Base.MathConstants.eulergamma / 63
        + 3911888 * Base.MathConstants.eulergamma * eta / 1323
        + 90490 * π^2 / 189 - 109295 * π^2 * eta / 224 + 11275 * π^2 * eta^2 / 144
        + (-233915 * π / 56 + 3970375 * π * eta / 756 - 19655 * π * eta^2 / 63) * chi_s
        + (-233915 * π / 56 + 99185 * π * eta / 84) * delta * chi_a
        +
        (
            4023088585 / 6096384 + 1926964265 * eta / 1524096 + 222955 * eta^2 / 6048
            -
            46495 * eta^3 / 216
        ) * chi_s^2
        +
        (
            4023088585 / 3048192 + 4412285 * eta / 2016 + 31535 * eta^2 / 432
        ) * delta * chi_s * chi_a
        +
        (
            4023088585 / 6096384 - 62246795 * eta / 36288 - 18751735 * eta^2 / 6048
            +
            950 * eta^3 / 3
        ) * chi_a^2
        + (12005 / 96 - 245 * eta / 12 + 5 * eta^2 / 6) * chi_s^4
        + (12005 / 24 - 245 * eta / 6) * delta * chi_s^3 * chi_a
        + (12005 / 16 - 12005 * eta / 4 + 80 * eta^2) * chi_s^2 * chi_a^2
        + (12005 / 24 - 1960 * eta) * delta * chi_s * chi_a^3
        + (12005 / 96 - 980 * eta + 1920 * eta^2) * chi_a^4
    )
    p8L2 = 18406 / 63 + 1955944 * eta / 1323
    return p8L, p8L2
end

"""
$(TYPEDSIGNATURES)

4.5PN coefficients `(p9, p9L)` of the TaylorF2 phase bracket
([`PhaseCoefficients`](@ref)), arguments as for [`phase_coefficient_25pn`](@ref): the
non-spinning terms (Blanchet, Faye, Henry, Larrouturou & Trestini 2023,
doi:10.1103/PhysRevLett.131.121402; Blanchet 2024, doi:10.1007/s41114-024-00050-z).
Only the non-spinning term is known at this order, so at 4.5PN spinning sources carry a
truncation that omits terms of order χ v⁹; `delta`, `chi_s` and `chi_a` are unused.
"""
@inline function phase_coefficients_45pn(eta, delta, chi_s, chi_a)
    p9 = (
        105344279473163 * π / 18776862720 - 1492917260735 * π * eta / 134120448
        + 45293335 * π * eta^2 / 127008 + 10323755 * π * eta^3 / 199584
        -
        27392 * π * log(2.0) / 21 - 13696 * π * Base.MathConstants.eulergamma / 21
        -
        640 * π^3 / 3 + 2255 * π^3 * eta / 6
    )
    p9L = -13696 * π / 21
    return p9, p9L
end

"""
$(TYPEDSIGNATURES)

Coefficients of the TaylorF2 phase bracket ([`PhaseCoefficients`](@ref)) for two
black holes with aligned spins `chi1` (heavier body), `chi2` and symmetric mass
ratio `eta`, through the post-Newtonian `order` (one of
[`PHASE_PN_ORDERS`](@ref)); every coefficient above that order is an exact zero.
Order 1.5 carries the Newtonian and 1PN terms, the 1.5PN tail and the leading
spin–orbit term ([`spin_beta`](@ref)). Order 2 adds the 2PN term with the spin–spin
coupling ([`phase_coefficient_2pn`](@ref)). Order 2.5 adds the non-spinning and
next-to-leading-order spin–orbit terms with their logarithm
(`phase_coefficient_25pn`). Order 3 adds the non-spinning terms with their
logarithm, the spin–orbit tail and the next-to-leading-order spin–spin terms
(`phase_coefficients_3pn`). Order 3.5 adds the non-spinning term, the
next-to-next-to-leading-order spin–orbit terms, the spin–spin tail and the leading
cubic-in-spin terms (`phase_coefficient_35pn`). Order 4 adds the `ln v` and
`ln²v` terms of the non-spinning sector, the spin–orbit tail, the
next-to-next-to-leading-order spin–spin terms and the leading quartic-in-spin terms,
the constant being a time shift set to zero (`phase_coefficients_4pn`).
Order 4.5 adds the non-spinning terms only (`phase_coefficients_45pn`).
`order` is not checked here, so that the function stays free of exceptions inside
GPU kernels: a value outside [`PHASE_PN_ORDERS`](@ref) acts as the highest order
not above it, or as 1.5 when it is below 1.5 or NaN. Validated orders enter through
[`WaveformParams`](@ref) and the configuration.
Sources: doi:10.1007/s41114-024-00050-z, doi:10.1103/PhysRevD.80.084043,
doi:10.1088/0264-9381/30/13/135009, doi:10.1088/0264-9381/30/7/075017,
doi:10.1088/0264-9381/32/19/195010, doi:10.1088/0264-9381/32/8/085008,
doi:10.1103/PhysRevD.106.L101501, doi:10.1088/0264-9381/31/2/025023,
doi:10.1103/PhysRevD.97.124046, doi:10.1103/PhysRevLett.131.121402,
doi:10.1103/PhysRevD.79.104023 (erratum doi:10.1103/PhysRevD.84.049901).
"""
@inline function phase_coefficients(chi1, chi2, eta, order::Real)
    beta = spin_beta(chi1, chi2, eta)
    z = zero(beta)
    T = typeof(z)
    p2 = T(3715 / 756 + 55 * eta / 9)
    p3 = 4 * beta - 16 * π
    p4 = order >= 2 ? T(phase_coefficient_2pn(eta, spin_sigma(chi1, chi2, eta))) : z
    delta = mass_asymmetry(eta)
    chi_s = 0.5 * (chi1 + chi2)
    chi_a = 0.5 * (chi1 - chi2)
    p5 = order >= 2.5 ? T(phase_coefficient_25pn(eta, delta, chi_s, chi_a)) : z
    p5L = 3 * p5
    if order >= 3
        q6, q6L = phase_coefficients_3pn(eta, delta, chi_s, chi_a)
        p6, p6L = T(q6), T(q6L)
    else
        p6, p6L = z, z
    end
    p7 = order >= 3.5 ? T(phase_coefficient_35pn(eta, delta, chi_s, chi_a)) : z
    if order >= 4
        q8L, q8L2 = phase_coefficients_4pn(eta, delta, chi_s, chi_a)
        p8L, p8L2 = T(q8L), T(q8L2)
    else
        p8L, p8L2 = z, z
    end
    if order >= 4.5
        q9, q9L = phase_coefficients_45pn(eta, delta, chi_s, chi_a)
        p9, p9L = T(q9), T(q9L)
    else
        p9, p9L = z, z
    end
    return PhaseCoefficients{T}(p2, p3, p4, p5, p5L, p6, p6L, p7, p8L, p8L2, p9, p9L)
end

"""
$(TYPEDSIGNATURES)

Post-Newtonian phase of the (2,2) harmonic at gravitational-wave frequency
`F`, TaylorF2 in the total-mass velocity `v = (π M F)^{1/3}`:
`(3/(128 η v⁵)) [1 + (3715/756 + 55η/9) v² + (4β − 16π) v³ + ψ₄ v⁴]`: the
Newtonian term, the 1PN term, the 1.5PN tail and spin–orbit term with `beta`
from [`spin_beta`](@ref), and the 2PN term with `psi4` from
[`phase_coefficient_2pn`](@ref). Without `psi4` the phase stops at 1.5PN
order. `F` [Hz], `chirp_mass` [s]; returns the phase [rad].
"""
@inline function pn_phase(F, chirp_mass, eta, beta, psi4 = zero(beta))
    v = (π * total_mass(chirp_mass, eta) * F)^(1 / 3)
    return (3 / (128 * eta * v^5)) *
           (1 + (3715 / 756 + 55 * eta / 9) * v^2 + (4 * beta - 16 * π) * v^3 + psi4 * v^4)
end

"""
$(TYPEDSIGNATURES)

Post-Newtonian phase of the (2,2) harmonic at gravitational-wave frequency `F`,
TaylorF2 in the total-mass velocity `v = (π M F)^{1/3}`, with the bracket of
[`PhaseCoefficients`](@ref) and the coefficients `c` from
[`phase_coefficients`](@ref). `order` selects which terms enter; the coefficients
above it are zero, and the branches skip their arithmetic and the logarithm.
As in [`phase_coefficients`](@ref), `order` is not checked. `F` [Hz], `chirp_mass`
[s]; returns the phase [rad].
"""
@inline function pn_phase(F, chirp_mass, eta, c::PhaseCoefficients, order::Real)
    v = (π * total_mass(chirp_mass, eta) * F)^(1 / 3)
    bracket = 1 + c.p2 * v^2 + c.p3 * v^3
    if order >= 2
        bracket += c.p4 * v^4
    end
    if order >= 2.5
        lv = log(v)
        bracket += (c.p5 + c.p5L * lv) * v^5
        order >= 3 && (bracket += (c.p6 + c.p6L * lv) * v^6)
        order >= 3.5 && (bracket += c.p7 * v^7)
        order >= 4 && (bracket += (c.p8L * lv + c.p8L2 * lv^2) * v^8)
        order >= 4.5 && (bracket += (c.p9 + c.p9L * lv) * v^9)
    end
    return (3 / (128 * eta * v^5)) * bracket
end

"""
$(TYPEDSIGNATURES)

Stationary-phase phase of harmonic `k` (`k` times the orbital phase) at
frequency `f`, in the Fourier convention `h̃(f) = ∫ h(t) e^{-2πift} dt`
(Baghi et al. 2026, Eq. 1), for which `h̃(f) ∝ e^{-iΨ_k(f)}` with
`Ψ_k(f) = 2πf t_c − (k/2) φ_c + (k/2) ψ_PN(2f/k) − π/4`: `ψ_PN` is
[`pn_phase`](@ref) of the (2,2) harmonic (to 1.5PN order without `psi4`) and
`2f/k` the (2,2) frequency at which harmonic `k` radiates at `f`; `φ_c` is the
(2,2) phase at coalescence. The arrival-time term is common to all harmonics,
and at Newtonian order `dΨ₂/df = 2π t(f)` with `t(f)` the Newtonian [`spa_time`](@ref).
`f` [Hz], `chirp_mass` and `coalescence_time` [s], `coalescence_phase` [rad];
returns the phase [rad]. The method with [`PhaseCoefficients`](@ref) `c` and the
post-Newtonian `order` evaluates `ψ_PN` with the `pn_phase` method of the same
arguments, to any order of [`PHASE_PN_ORDERS`](@ref).
"""
@inline function harmonic_phase(f, k::Integer, chirp_mass, eta, coalescence_time,
    coalescence_phase, beta, psi4 = zero(beta))
    half_k = k / 2
    return 2 * π * f * coalescence_time - half_k * coalescence_phase +
           half_k * pn_phase(f / half_k, chirp_mass, eta, beta, psi4) - π / 4
end

@inline function harmonic_phase(f, k::Integer, chirp_mass, eta, coalescence_time,
    coalescence_phase, c::PhaseCoefficients, order::Real)
    half_k = k / 2
    return 2 * π * f * coalescence_time - half_k * coalescence_phase +
           half_k * pn_phase(f / half_k, chirp_mass, eta, c, order) - π / 4
end

"""
$(TYPEDSIGNATURES)

Newtonian stationary-phase time at which the (2,2) harmonic radiates at
frequency `F` [Hz]: `t_c − 5𝓜/(256 v⁸)` with `v = (π 𝓜 F)^{1/3}`.
`chirp_mass` [s], `coalescence_time` [s]; returns a time [s].
"""
@inline function spa_time(F, chirp_mass, coalescence_time)
    v = (π * chirp_mass * F)^(1 / 3)
    return coalescence_time - 5 * chirp_mass / (256 * v^8)
end

"""
$(TYPEDSIGNATURES)

Stationary-phase time at which the (2,2) harmonic radiates at frequency `F`,
from the phase of the same order: `t(F) = t_c + (1/2π) dψ_PN/dF` with `ψ_PN`
of [`pn_phase`](@ref), coefficients `c` and `order` as there. With
`v = (π M F)^{1/3}` and the bracket `B(v)` of [`PhaseCoefficients`](@ref),
`dψ_PN/dF = (1/(128 η F v⁵)) (v B′ − 5B)`, so that
`t(F) = t_c + (v B′ − 5B)/(256 π η v⁵ F)`; with `B = 1` this is the Newtonian
[`spa_time`](@ref). Below the innermost stable orbit the result can decrease
with `F` and exceed `coalescence_time` (Waveform Physics, §8). `F` [Hz],
`chirp_mass` and `coalescence_time` [s]; returns a time [s].
"""
@inline function spa_time(F, chirp_mass, eta, coalescence_time, c::PhaseCoefficients,
    order::Real)
    v = (π * total_mass(chirp_mass, eta) * F)^(1 / 3)
    # v B′ − 5B term by term: v^i (a + b ln v + d ln²v) contributes
    # v^i [(i − 5)(a + b ln v + d ln²v) + b + 2d ln v]
    s = -5 - 3 * c.p2 * v^2 - 2 * c.p3 * v^3
    if order >= 2
        s -= c.p4 * v^4
    end
    if order >= 2.5
        lv = log(v)
        s += c.p5L * v^5
        order >= 3 && (s += (c.p6 + c.p6L * lv + c.p6L) * v^6)
        order >= 3.5 && (s += 2 * c.p7 * v^7)
        order >= 4 &&
            (s += (3 * (c.p8L * lv + c.p8L2 * lv^2) + c.p8L + 2 * c.p8L2 * lv) * v^8)
        order >= 4.5 && (s += (4 * (c.p9 + c.p9L * lv) + c.p9L) * v^9)
    end
    return coalescence_time + s / (256 * π * eta * v^5 * F)
end

"""
$(TYPEDSIGNATURES)

Dimensionless weight of the signal emitted at time `t` [s] for an observation over
`[0, observation_time]` [s]:
`W(t) = ½ [tanh(t/Δ) − tanh((t − T)/Δ)]` with `T = observation_time` and
`Δ = edge_time` [s]: unity inside the observation, ½ at either end, zero outside,
with edges of time scale `Δ`. In the stationary-phase approximation a window
that varies slowly against the local chirp time `1/√ḟ` multiplies each
harmonic at its emission time `t_k(f)` ([`spa_time`](@ref)); `Δ` must
therefore stay well above `1/√ḟ` at the edges (days for the sources of the
shipped configurations). Smooth in `t`, hence differentiable in the chirp mass
and the coalescence time, through which the edges move in frequency.
"""
@inline function observation_window(t, observation_time, edge_time)
    return (tanh(t / edge_time) - tanh((t - observation_time) / edge_time)) / 2
end

"""
$(TYPEDSIGNATURES)

Smooth end-of-inspiral window applied to a harmonic radiating at (2,2)
frequency `F` [Hz]: `½[1 − tanh((F/f_isco − 1)/width)]`, unity well below the
innermost stable orbit and zero well above it, differentiable in every
parameter (the cutoff moves with the total mass). `f_isco` [Hz]; `width` is
dimensionless.
"""
@inline inspiral_taper(F, f_isco, width) = 0.5 * (1 - tanh((F / f_isco - 1) / width))

"""
$(TYPEDSIGNATURES)

Stationary-phase amplitudes `(a₊, a×)` of harmonic `k` at frequency `f` [Hz] for a
source at luminosity `distance` [s]: the polarization amplitudes of
Blanchet, Iyer, Will & Wiseman (1996, doi:10.1088/0264-9381/13/4/002; leading order
for `k = 2`, 0.5PN order for `k = 1, 3`, both ∝ the mass asymmetry `δ`),
`h₊ = (2Mηx/D)[−(1 + c²) cos 2ψ − (sδ/8)((5 + c²) cos ψ − 9(1 + c²) cos 3ψ) x^{1/2}]`,
`h× = (2Mηx/D)[−2c sin 2ψ − (3/4) s c δ (sin ψ − 3 sin 3ψ) x^{1/2}]`
(`c = cos ι`, `s = sin ι`, `x = (πMF)^{2/3}`), converted by the stationary
phase approximation: harmonic `k` has phase `kψ`, so
`a = (A_k/2) √(2/(k Ḟ))` with `Ḟ = (96/5) π^{8/3} 𝓜^{5/3} F^{11/3}` the
Newtonian chirp rate at its (2,2) frequency `F = 2f/k`. The `×`
amplitudes multiply `sin(kψ)` and enter the strain with a factor `−i`. For
`k = 2` this reproduces `√(5/24) π^{-2/3} 𝓜^{5/6} f^{-7/6} (1 + c²)/(2D)`.
`chirp_mass` [s], `inclination` [rad]; the amplitudes are in [Hz⁻¹].
"""
@inline function harmonic_amplitudes(f, k::Integer, chirp_mass, eta, inclination, distance)
    F = 2 * f / k
    M = total_mass(chirp_mass, eta)
    x = (π * M * F)^(2 / 3)
    chirp_rate = (96 / 5) * π^(8 / 3) * chirp_mass^(5 / 3) * F^(11 / 3)
    prefactor = (M * eta * x / distance) * sqrt(2 / (k * chirp_rate))
    c = cos(inclination)
    s = sin(inclination)
    if k == 2
        return -(1 + c^2) * prefactor, -2 * c * prefactor
    end
    half_pn = sqrt(x) * mass_asymmetry(eta) * prefactor
    if k == 1
        return -(s / 8) * (5 + c^2) * half_pn, -(3 / 4) * s * c * half_pn
    end
    return (9 * s / 8) * (1 + c^2) * half_pn, (9 / 4) * s * c * half_pn
end

"""
$(TYPEDSIGNATURES)

Parameter vector of the second source of a two-source configuration:
`theta0` displaced by `delta` along `u_norm`, with the luminosity distance
set to `theta0[1] / amp_ratio`, an amplitude ratio `A₂/A₁ = amp_ratio` at
equal chirp mass, the amplitude separation being carried by the ratio and
never by the direction.
"""
function second_source(theta0::AbstractVector, u_norm::AbstractVector, delta::Real,
    amp_ratio::Real)
    p2 = theta0 .+ delta .* u_norm
    p2[1] = theta0[1] / amp_ratio
    return p2
end

end # module
