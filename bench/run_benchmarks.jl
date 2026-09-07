# Performance benchmarks, kept out of the test suite by design (benchmarks
# measure, tests assert). Uses its own environment; the package resolves by
# path via [sources].
#
#   julia --threads=auto bench/run_benchmarks.jl
#
using Pkg
Pkg.activate(@__DIR__; io = devnull)
# the package resolves by path via [sources] (unregistered dependency)
Pkg.instantiate(; io = devnull)

using BenchmarkTools
using ForwardDiff
using KernelAbstractions
using CurvatureDistinguishability

# Moderate grid: large enough to be representative, small enough to finish fast.
T_obs = SECONDS_PER_YEAR
df = 1.0 / T_obs
freqs = collect(1.0e-3:df:2.0e-3)
noise = NoiseParams()
Sn_vals = analytic_noise_psd.(freqs; noise = noise)
wp = waveform_params(time_scale = 100.0, sky_phi = π, polarization = 0.785)

theta_0 = [1.0, 2.0, 1.0, 0.0, 0.5, 0.5]
u_dir = [0.0, 1.0, 0.1, 0.0, 0.0, 0.0]

h1 = scaled_waveform_model(theta_0, freqs, wp)
h2 = scaled_waveform_model(theta_0 .+ 1e-4 .* u_dir, freqs, wp)
H1 = project_to_tdi(h1, freqs, theta_0, wp)
H2 = project_to_tdi(h2, freqs, theta_0 .+ 1e-4 .* u_dir, wp)
data = map((a, b) -> a .+ b, H1, H2)

println("=====================================================")
println("  Benchmarks: CurvatureDistinguishability")
println("  grid: $(length(freqs)) bins, $(Threads.nthreads()) threads")
println("=====================================================")

println("\n[1] Multi-channel noise-weighted inner product")
display(@benchmark multi_channel_inner_product($H1, $H2, $Sn_vals, $df))
println()

println("\n[2] Tangent-basis generation (6-column Jacobian + MGS)")
display(
    @benchmark compute_tangent_basis($theta_0, $freqs, $Sn_vals, $df, $wp) samples = 5 evals =
        1
)
println()

basis = compute_tangent_basis(theta_0, freqs, Sn_vals, df, wp)

println("\n[3] Directional curvature (fused value+dh+d²h nested-dual pass)")
display(
    @benchmark compute_extrinsic_curvature_from_basis($theta_0, $u_dir, $basis,
        $freqs, $Sn_vals, $df, $wp) samples = 10 evals = 1
)
println()

loss = loss_function(data, freqs, Sn_vals, df, wp, CPU())
p0 = [2.0, 2.0, 1.0, 0.0, 0.5, 0.5]
G = zeros(CurvatureDistinguishability.Physics.N_PARAMS)

println("\n[4] Loss value: allocation-free CPU loop")
display(@benchmark $loss($p0))
println()

println("\n[5] Loss value: KernelAbstractions kernel on the CPU backend")
display(
    @benchmark CurvatureDistinguishability.Inference.device_loss($p0, $freqs, $Sn_vals,
        $(data[1]), $(data[2]),
        $df, $wp, $(CPU()))
)
println()

println("\n[6] ForwardDiff gradient of the loss (CPU loop)")
display(@benchmark ForwardDiff.gradient!($G, $loss, $p0))
println()

println("\n[7] Full bounded optimization: IPNewton vs Fminbox(LBFGS)")
for opt in (:ipnewton, :lbfgs_box)
    t0 = time()
    d2, _, res = calculate_numerical_distance(data, p0, freqs, Sn_vals, df;
        optimizer = opt, wp = wp)
    diag = optimization_diagnostics(
        res,
        ones(CurvatureDistinguishability.Physics.N_PARAMS),
        default_bounds(),
    )
    println(
        "  $(rpad(opt, 10)): D² = $(d2)  iters = $(diag.iterations)  " *
        "wall = $(round(time() - t0, digits = 2)) s",
    )
end
