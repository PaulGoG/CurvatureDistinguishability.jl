# Shared constants of the test files: the 201-bin fixture grid and its
# physical setup. Included by runtests.jl.

const CD = CurvatureDistinguishability
const FIXDIR = joinpath(@__DIR__, "fixtures", "reference")

const FIX_DF = 1e-5
const FIX_FREQS = collect(1e-3:FIX_DF:3e-3)
const FIX_PHYS = (mass_scale = 1.0, time_scale = 100.0, eta = 0.25,
    ecliptic_longitude = 3.1415, ecliptic_latitude = 0.5238,
    inclination = 0.523, polarization = 0.785)
const FIX_WP = waveform_params(; FIX_PHYS...)
const THETA0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
const FIX_NOISE_OFF = NoiseParams(confusion_enabled = false)
const FIX_SN = analytic_noise_psd.(FIX_FREQS; noise = FIX_NOISE_OFF)
