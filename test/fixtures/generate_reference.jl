# Regenerate the committed reference fixtures on the 201-bin fixture grid
# with the current model: julia test/fixtures/generate_reference.jl
include(joinpath(@__DIR__, "..", "..", "activate.jl"))
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

# largest relative change of the numeric columns shared by a committed table and
# its regenerated values; rows with a zero committed value are skipped
function max_relative_change(old::DataFrame, new::DataFrame)
    worst = 0.0
    for name in intersect(names(old), names(new))
        for (a, b) in zip(old[!, name], new[!, name])
            (a isa Real && b isa Real && a != 0) || continue
            worst = max(worst, abs(b / a - 1))
        end
    end
    return worst
end

# psd.csv: header-less (f, Sn), instrumental noise only
CSV.write(
    joinpath(FIXDIR, "psd.csv"),
    DataFrame(f = FIX_FREQS, Sn = FIX_SN);
    header = false,
)

# sweep.csv: the reference direction on a log-spaced δ grid
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
# the optimized distances must follow the curvature prediction before they
# become reference values
d2_theo = (1 / 16) .* K_norm .* deltas .^ 4
worst = maximum(abs.(d2_num ./ d2_theo .- 1))
worst < 2e-3 || error(
    "sweep fixture not written: optimized and predicted D² differ by up to $worst")
sweep = DataFrame(Delta = deltas, D2_Numerical = d2_num, D2_Theoretical = d2_theo,
    K_u_Norm = fill(K_norm, length(deltas)))
sweep_path = joinpath(FIXDIR, "sweep.csv")
isfile(sweep_path) && println("sweep.csv: max relative change ",
    max_relative_change(CSV.read(sweep_path, DataFrame), sweep))
CSV.write(sweep_path, sweep)

# map_<px>_<py>.csv: per-angle curvature and unit-threshold boundary
basis = compute_tangent_basis(THETA0, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
for (px, py) in ((2, 3), (5, 6), (2, 4))
    path = joinpath(FIXDIR, "map_$(px)_$(py).csv")
    committed = CSV.read(path, DataFrame)
    angles = committed.Angle
    rows = map(angles) do phi
        dir = zeros(6)
        dir[px] = cos(phi)
        dir[py] = sin(phi)
        K, g = compute_extrinsic_curvature_from_basis(THETA0, dir, basis,
            FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
        r = boundary_radius(K, 1.0)
        (Angle = phi, K_raw = K, G_uu = g, X_Bound = r * cos(phi), Y_Bound = r * sin(phi))
    end
    regenerated = DataFrame(rows)
    println("map_$(px)_$(py).csv: max relative change ",
        max_relative_change(committed, regenerated))
    CSV.write(path, regenerated)
end
println("reference fixtures regenerated in $FIXDIR")
