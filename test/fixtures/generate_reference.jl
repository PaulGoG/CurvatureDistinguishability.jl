# Regenerate the committed reference fixtures on the 201-bin fixture grid
# with the current model (run from the package root: julia --project test/fixtures/generate_reference.jl).
using CurvatureDistinguishability
using CSV, DataFrames
const FIXDIR = joinpath(@__DIR__, "reference")
const FIX_DF = 1e-5
const FIX_FREQS = collect(1e-3:FIX_DF:3e-3)
const FIX_PHYS = (mass_scale = 1.0, time_scale = 100.0, eta = 0.25,
    ecliptic_longitude = 3.1415, ecliptic_latitude = 0.5238,
    inclination = 0.523, polarization = 0.785)
const FIX_WP = waveform_params(; FIX_PHYS...)
const THETA0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
const FIX_SN =
    analytic_noise_psd.(FIX_FREQS; noise = NoiseParams(confusion_enabled = false))

# psd.csv: header-less (f, Sn) — instrumental noise only
CSV.write(
    joinpath(FIXDIR, "psd.csv"),
    DataFrame(f = FIX_FREQS, Sn = FIX_SN);
    header = false,
)

# sweep.csv: the reference direction on the existing Delta grid
u_raw = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
K_u, g_uu = compute_extrinsic_curvature(THETA0, u_raw, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
K_norm = K_u / g_uu^2
u_norm = u_raw ./ sqrt(g_uu)
deltas = (16 / K_norm)^(1 / 4) .* 10 .^ range(-2.0, 0.3, length = 8) # δ/δ_min ∈ [0.01, 2]
c1 = channel_strain(THETA0, FIX_FREQS, FIX_WP)
d2_num = map(deltas) do d
    p2 = THETA0 .+ d .* u_norm
    c2 = channel_strain(p2, FIX_FREQS, FIX_WP)
    data = map((a, b) -> a .+ b, c1, c2)
    guess = THETA0 .+ (0.5 * d) .* u_norm
    guess[1] /= 2.0
    first(
        calculate_numerical_distance(data, guess, FIX_FREQS, FIX_SN, FIX_DF;
            optimizer = :ipnewton, wp = FIX_WP),
    )
end
CSV.write(joinpath(FIXDIR, "sweep.csv"),
    DataFrame(Delta = deltas, D2_Numerical = d2_num,
        D2_Theoretical = (1 / 16) .* K_norm .* deltas .^ 4,
        K_u_Norm = fill(K_norm, length(deltas))))

# map_<px>_<py>.csv: per-angle curvature and unit-threshold boundary
basis = compute_tangent_basis(THETA0, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
for (px, py) in ((2, 3), (5, 6), (2, 4))
    path = joinpath(FIXDIR, "map_$(px)_$(py).csv")
    angles = CSV.read(path, DataFrame).Angle
    rows = map(angles) do phi
        dir = zeros(6)
        dir[px] = cos(phi)
        dir[py] = sin(phi)
        K, g = compute_extrinsic_curvature_from_basis(THETA0, dir, basis,
            FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
        r = boundary_radius(K, 1.0)
        (Angle = phi, K_raw = K, G_uu = g, X_Bound = r * cos(phi), Y_Bound = r * sin(phi))
    end
    CSV.write(path, DataFrame(rows))
end
println("reference fixtures regenerated in $FIXDIR")
