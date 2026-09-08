# # The quartic law on a miniature grid
#
# This executable example reproduces the package's central statement,
# ``D^2 \approx K(u)\,\delta^4/16``, end to end on a grid small enough to
# run in seconds: differential geometry, direct two-source optimization,
# and the comparison figure. The production pipeline automates exactly
# this workflow at scale (see the shipped `configs/`).

using CurvatureDistinguishability
using CairoMakie

# A reduced frequency grid with the Robson et al. (2019) noise model:

physics = (mass_scale = 1.0, time_scale = 100.0, eta = 0.25,
    ecliptic_longitude = 3.1415, ecliptic_latitude = 0.5238,
    inclination = 0.523, polarization = 0.785)
wp = waveform_params(; physics...)
df = 1e-5
freqs = collect(1e-3:df:2e-2)
Sn = analytic_noise_psd.(freqs; noise = robson_confusion_params(1 / df))
length(freqs)

# The reference source and the classic chirp-mass/coalescence-time
# direction:

theta0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
u_raw = [0.0, 1.0, 0.5, 0.0, 0.0, 0.0]

# Directional extrinsic curvature and Fisher norm from one fused
# nested-dual evaluation, then the Fisher-normalized direction and the
# discernibility boundary at ``\rho_{\mathrm{thr}} = 1``:

K_u, g_uu = compute_extrinsic_curvature(theta0, u_raw, freqs, Sn, df, wp)
u_norm = u_raw ./ sqrt(g_uu)
K_norm = K_u / g_uu^2
delta_min = (16 / K_norm)^(1 / 4)
(K_norm, delta_min)

# Direct optimization at five separations against the prediction: inject
# two sources, fit a single source, and record the residual squared
# distance.

deltas = 10.0 .^ range(-3, -1, length = 5)
D2_num = map(deltas) do d
    p2 = theta0 .+ d .* u_norm
    data = map((a, b) -> a .+ b, channel_strain(theta0, freqs, wp),
        channel_strain(p2, freqs, wp))
    d2, _, _ = calculate_numerical_distance(data, copy(theta0), freqs, Sn, df;
        iterations = 60, wp = wp)
    d2
end
D2_theo = (K_norm / 16) .* deltas .^ 4
D2_num ./ D2_theo

# The ratios sit at unity across two decades of separation — the quartic
# law, from nothing but the manifold's curvature. The comparison figure:

fig = with_theme(publication_theme()) do
    fig = Figure(size = (700, 480))
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10,
        xlabel = "Parameter separation δ", ylabel = "D²")
    lines!(ax, deltas, D2_theo; color = :darkred, linewidth = 2,
        label = "Prediction K(u) δ⁴ / 16")
    scatter!(ax, deltas, D2_num; color = :dodgerblue, markersize = 14,
        strokecolor = :black, strokewidth = 1, label = "Direct optimization")
    fig[0, 1] = Legend(fig, ax; orientation = :horizontal, framevisible = false)
    fig
end
