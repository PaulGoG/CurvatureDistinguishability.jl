"""
Analytic LISA noise model (Robson et al. 2019, Eq. 10–14, Table 1) and the
post-Newtonian inspiral model shared by the CPU loss loop and the GPU
kernels: 1.5PN TaylorF2 phasing for aligned spins and unequal masses,
0.5PN-amplitude harmonics, an innermost-stable-orbit taper and the physical
source parameters (luminosity distance, chirp mass, coalescence time and
phase, aligned spins).
"""
module Physics

using DocStringExtensions: TYPEDSIGNATURES
export NoiseParams, robson_confusion_params, analytic_noise_psd, sky_averaged_response,
    WaveformParams, waveform_params, spin_beta, pn_phase, harmonic_phase,
    harmonic_amplitudes, total_mass, isco_frequency, second_source,
    SECONDS_PER_YEAR
public N_PARAMS, LISA_ARM_LENGTH, GIGAPARSEC_SEC, HARMONICS, C_LIGHT, R_ORBIT_SEC,
    transfer_frequency, mass_asymmetry, spa_time, inspiral_taper, observation_window,
    ResponseGeometry

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
with `f★ = c/(2π L)` from `arm_length` [m]. It relates a two-channel
sensitivity-level spectral density to the per-channel PSD (their Eq. 1,
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
Orbital radius of the constellation centre, 1 AU in light-seconds.
"""
const R_ORBIT_SEC = 499.00478383615643

"""
Harmonics of the orbital phase carried by the waveform: `k = 2` at leading
amplitude order, `k = 1` and `k = 3` at 0.5PN amplitude order (both
proportional to the mass asymmetry).
"""
const HARMONICS = (1, 2, 3)

"""
Dimension of the waveform parameter vector θ = (A, 𝓜, t_c, Φ₀, χ₁, χ₂).
Single source of truth for every parameter-count-dependent structure
(bounds tuples, kernel parameter tuples, work-item validation).
"""
const N_PARAMS = 6

# positive PSD floor returned for non-positive frequencies, so the noise
# weighting never divides by zero
const PSD_FLOOR = 1e-30

"""
Seconds in one Julian year (365.25 d). Single source of truth for the
detector's orbital period and the default observation time.
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
Michelson-channel PSD of Robson et al. (2019) Eq. 12 (strain-referred, *not*
sky-averaged — antenna response is applied explicitly by `Detector`), built
from the Eq. 10 optical-metrology noise `oms_amplitude` [m Hz⁻¹ᐟ²] with its
`oms_reddening_freq` [Hz] low-frequency term, the Eq. 11 acceleration noise
`acc_amplitude` [m s⁻² Hz⁻¹ᐟ²] with shoulders `acc_knee_low`/`acc_knee_high`
[Hz], and the `arm_length` [m] (which also sets the transfer frequency
`f★ = c/(2π L)`). The galactic confusion part is Eq. 14,
`S_c(f) = A f^{-7/3} e^{-f^α + β f sin(κf)} [1 + tanh(γ(f_k - f))]`,
defaulting to the 1-yr column of Table 1. Eq. 14 is calibrated on the
two-channel sensitivity curve (their Eq. 1, `S_n = P_n/R + S_c`), so
[`analytic_noise_psd`](@ref) multiplies it by the Eq. 9 response
[`sky_averaged_response`](@ref) before adding it to the channel noise.

All fields are configurable through the `[noise]` section of the run configuration;
the defaults reproduce the published Robson et al. (2019) LISA model.
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
et al. (2019) Table 1 column nearest to the observation time `T_obs` [s].
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

One-sided noise PSD of one channel at frequency `f` [Hz]: the Robson et
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

    # Optical Metrology Noise (Robson Eq. 10)
    p_oms = noise.oms_amplitude^2 * (1 + (noise.oms_reddening_freq / f)^4)

    # Acceleration Noise (Robson Eq. 11)
    p_acc =
        noise.acc_amplitude^2 * (1 + (noise.acc_knee_low / f)^2) *
        (1 + (f / noise.acc_knee_high)^4)

    # Total Instrumental Noise (Robson Eq. 12); f★ = c/(2πL)
    L = noise.arm_length
    f_star = C_LIGHT / (2 * π * L)
    s_inst = (p_oms / L^2) + (2 * p_acc / ((2 * π * f)^4 * L^2)) * (1 + cos(f / f_star)^2)

    if !noise.confusion_enabled
        return s_inst
    end

    # Galactic binary confusion noise (Robson Eq. 14): a two-channel
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
`mass_scale` (chirp mass, geometrised seconds), `time_scale` (coalescence
time, s) and `distance_scale` (luminosity distance, s; default one
gigaparsec) restore the physical values of the O(1) parameter vector. The
symmetric mass ratio `eta` ∈ (0, 1/4] is fixed per configuration (the
heavier body carries `chi1`). The source sits at ecliptic longitude
`ecliptic_longitude` and latitude `ecliptic_latitude` [rad], with
inclination `inclination` (angle between the propagation direction and the
orbital angular momentum) and polarization angle `polarization` in the
Rosetta Stone conventions. `orbit_phase` and `constellation_phase` are the
orbital and cartwheel phases of the constellation at `t = 0`, `arm_length`
[m] sets the orbital eccentricity of the spacecraft and the transfer
frequency, and `cutoff_width` is the relative width of the
innermost-stable-orbit taper. `observation_time` [s] is the duration of the
observation that starts at `t = 0`: when positive, every harmonic is weighted
by [`observation_window`](@ref) at its stationary-phase emission time, with
edges of time scale `window_edge_time` [s]; at the default `0` the signal
fills the whole frequency band whenever it was emitted. The active channel count (2, or 3 with the
identically zero T channel, requested through the `include_t_channel`
keyword) is carried only as the type parameter `NCH`. `geometry` holds the
derived response constants.
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
    geometry::ResponseGeometry{T}
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
    include_t_channel::Bool = false)
    fields = promote(float(mass_scale), float(time_scale), float(distance_scale),
        float(eta), float(ecliptic_longitude), float(ecliptic_latitude),
        float(inclination), float(polarization), float(orbit_phase),
        float(constellation_phase), float(arm_length), float(cutoff_width),
        float(observation_time), float(window_edge_time))
    T = typeof(fields[1])
    geometry = response_geometry(fields[5], fields[6], fields[8], fields[10], fields[11])
    return WaveformParams{T,include_t_channel ? 3 : 2}(fields..., geometry)
end

# keywords accepted by the WaveformParams constructor: every stored parameter
# plus the channel-count switch, which lives in the type parameter
const WAVEFORM_KEYWORDS = (
    :mass_scale, :time_scale, :distance_scale, :eta, :ecliptic_longitude,
    :ecliptic_latitude, :inclination, :polarization, :orbit_phase,
    :constellation_phase, :arm_length, :cutoff_width, :observation_time,
    :window_edge_time, :include_t_channel,
)

"""
$(TYPEDSIGNATURES)

Build a [`WaveformParams`](@ref) from keyword arguments, rejecting unknown
keys with an `ArgumentError` naming the offending keyword (typo
protection at the physical-model boundary).
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
"""
@inline total_mass(chirp_mass, eta) = chirp_mass * eta^(-3 / 5)

"""
$(TYPEDSIGNATURES)

Mass asymmetry `δ = (m₁ − m₂)/M = √(1 − 4η)` (non-negative: `m₁ ≥ m₂`).
"""
@inline mass_asymmetry(eta) = sqrt(max(1 - 4 * eta, zero(eta)))

"""
$(TYPEDSIGNATURES)

Gravitational-wave frequency of the innermost stable circular orbit of a
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

Post-Newtonian phase of the (2,2) harmonic at gravitational-wave frequency
`F`, TaylorF2 to 1.5PN order in the total-mass velocity
`v = (π M F)^{1/3}`:
`(3/(128 η v⁵)) [1 + (3715/756 + 55η/9) v² + (4β − 16π) v³]` — the Newtonian
term, the 1PN term, the 1.5PN tail and the spin–orbit term with `beta` from
[`spin_beta`](@ref).
"""
@inline function pn_phase(F, chirp_mass, eta, beta)
    v = (π * total_mass(chirp_mass, eta) * F)^(1 / 3)
    return (3 / (128 * eta * v^5)) *
           (1 + (3715 / 756 + 55 * eta / 9) * v^2 + (4 * beta - 16 * π) * v^3)
end

"""
$(TYPEDSIGNATURES)

Stationary-phase phase of harmonic `k` (`k` times the orbital phase) at
frequency `f`, in the Fourier convention `h̃(f) = ∫ h(t) e^{-2πift} dt`
(LISA Rosetta Stone, Eq. 1), for which `h̃(f) ∝ e^{-iΨ_k(f)}` with
`Ψ_k(f) = 2πf t_c − (k/2) φ_c + (k/2) ψ_PN(2f/k) − π/4`: `ψ_PN` is
[`pn_phase`](@ref) of the (2,2) harmonic and `2f/k` the (2,2) frequency at
which harmonic `k` radiates at `f`; `φ_c` is the (2,2) phase at coalescence.
The arrival-time term is common to all harmonics, and
`dΨ₂/df = 2π t(f)` with the Newtonian map `t(f) = t_c − 5𝓜/(256 v⁸)`.
"""
@inline function harmonic_phase(f, k::Integer, chirp_mass, eta, coalescence_time,
    coalescence_phase, beta)
    half_k = k / 2
    return 2 * π * f * coalescence_time - half_k * coalescence_phase +
           half_k * pn_phase(f / half_k, chirp_mass, eta, beta) - π / 4
end

"""
$(TYPEDSIGNATURES)

Newtonian stationary-phase time at which the (2,2) harmonic radiates at
frequency `F`: `t_c − 5𝓜/(256 v⁸)` with `v = (π 𝓜 F)^{1/3}`.
"""
@inline function spa_time(F, chirp_mass, coalescence_time)
    v = (π * chirp_mass * F)^(1 / 3)
    return coalescence_time - 5 * chirp_mass / (256 * v^8)
end

"""
$(TYPEDSIGNATURES)

Weight of the signal emitted at time `t` for an observation over
`[0, observation_time]`:
`W(t) = ½ [tanh(t/Δ) − tanh((t − T)/Δ)]` with `T = observation_time` and
`Δ = edge_time` — unity inside the observation, ½ at either end, zero outside,
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
frequency `F`: `½[1 − tanh((F/f_isco − 1)/width)]`, unity well below the
innermost stable orbit and zero well above it, differentiable in every
parameter (the cutoff moves with the total mass).
"""
@inline inspiral_taper(F, f_isco, width) = 0.5 * (1 - tanh((F / f_isco - 1) / width))

"""
$(TYPEDSIGNATURES)

Stationary-phase amplitudes `(a₊, a×)` of harmonic `k` at frequency `f` for a
source at luminosity `distance` [s]: the polarization amplitudes of
Blanchet, Iyer, Will & Wiseman (leading order for `k = 2`, 0.5PN order for
`k = 1, 3`, both ∝ the mass asymmetry `δ`),
`h₊ = (2Mηx/D)[−(1 + c²) cos 2ψ − (sδ/8)((5 + c²) cos ψ − 9(1 + c²) cos 3ψ) x^{1/2}]`,
`h× = (2Mηx/D)[−2c sin 2ψ − (3/4) s c δ (sin ψ − 3 sin 3ψ) x^{1/2}]`
(`c = cos ι`, `s = sin ι`, `x = (πMF)^{2/3}`), converted by the stationary
phase approximation: harmonic `k` has phase `kψ`, so
`a = (A_k/2) √(2/(k Ḟ))` with `Ḟ = (96/5) π^{8/3} 𝓜^{5/3} F^{11/3}` the
Newtonian chirp rate at its (2,2) frequency `F = 2f/k`. The `×`
amplitudes multiply `sin(kψ)` and enter the strain with a factor `−i`. For
`k = 2` this reproduces `√(5/24) π^{-2/3} 𝓜^{5/6} f^{-7/6} (1 + c²)/(2D)`.
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
set to `theta0[1] / amp_ratio` — an amplitude ratio `A₂/A₁ = amp_ratio` at
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
