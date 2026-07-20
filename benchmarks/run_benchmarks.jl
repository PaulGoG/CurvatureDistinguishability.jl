using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io=devnull)

using BenchmarkTools
using TwoWaveformDistinguishability

# Setup Grid
T_obs = 3.15e7
df = 1.0 / T_obs
freqs = collect(1.0e-3:df:0.01)
Sn_vals = analytic_noise_psd.(freqs)

# Setup Parameters (6-param model)
theta_0 = [1.0, 2.0, 1.0, 0.0, 0.5, 0.5]
u_dir = [0.0, 1.0, 0.1, 0.0, 0.0, 0.0]

# Pre-compute waveforms and TDI channels
h1 = scaled_waveform_model(theta_0, freqs)
h2 = scaled_waveform_model(theta_0 .+ 1e-4.*u_dir, freqs)

H1 = project_to_tdi(h1, freqs, theta_0)
H2 = project_to_tdi(h2, freqs, theta_0 .+ 1e-4.*u_dir)

println("=====================================================")
println("  Performance Benchmarks: TwoWaveformDistinguishability")
println("=====================================================")

println("\n[1] Multi-Channel Noise-Weighted Inner Product")
b_ip = @benchmark multi_channel_inner_product($H1, $H2, $Sn_vals, $df)
display(b_ip)
println()

println("\n[2] Massive Orthogonal Tangent Basis Generation (Jacobian)")
b_basis = @benchmark compute_tangent_basis($theta_0, $freqs, $Sn_vals, $df) samples=2 evals=1
display(b_basis)
println()

basis = compute_tangent_basis(theta_0, freqs, Sn_vals, df)

println("\n[3] Lightning-Fast Extrinsic Curvature (Directional Hessian)")
b_geom = @benchmark compute_extrinsic_curvature_from_basis($theta_0, $u_dir, $basis, $freqs, $Sn_vals, $df) samples=10 evals=1
display(b_geom)
println()
