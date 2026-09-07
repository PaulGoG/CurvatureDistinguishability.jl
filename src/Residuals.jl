"""
Residual-spectrum diagnostics of the 1D sweeps: decimated `d(SNR²)/df`
densities of the two-source data, the best-fit single source and the
unabsorbed residual for the A and E channels, with the residual `D²`
integrals per channel.
"""
module Residuals

using DocStringExtensions: TYPEDSIGNATURES
using DataFrames: DataFrame
using Statistics: mean
using ..Physics: WaveformParams, scaled_waveform_model, second_source
using ..Detector: project_to_tdi

export residual_spectrum

"""
$(TYPEDSIGNATURES)

Decimated residual spectrum (density units, `d(SNR²)/df = 4|x|²/Sn`) of the
two-source data — the base source `theta0` plus a second source displaced by
`delta` along the Fisher-normalized direction `u_norm` with amplitude ratio
`amp_ratio` — the best-fit single source `best_fit` and the unabsorbed
residual, for channels A and E on the grid `freqs` with PSD `Sn` and bin
width `df`. Decimation windows are log-uniform in frequency (`n_windows` at
most, capped at the bin count): each holds the window mean of the density
— so the plotted curve integrates to the same `D²` as the full grid — and
its min/max envelope; log-sparse low-frequency windows pass single bins
through unaveraged, and empty windows are skipped. Returns the spectrum
table (`f` and, per channel, `sig_*`, `bf_*`, `res_*` mean/min/max columns)
and the residual `D²` integrals `(int_A, int_E)`.
"""
function residual_spectrum(theta0::AbstractVector, u_norm::AbstractVector,
    amp_ratio::Real, delta::Real, best_fit::AbstractVector,
    freqs::AbstractVector, Sn::AbstractVector, df::Real,
    wp::WaveformParams; n_windows::Integer)
    p2 = second_source(theta0, u_norm, delta, amp_ratio)
    h1 = scaled_waveform_model(theta0, freqs, wp)
    h2 = scaled_waveform_model(p2, freqs, wp)
    ch1 = project_to_tdi(h1, freqs, theta0, wp)
    ch2 = project_to_tdi(h2, freqs, p2, wp)
    data = map((a, b) -> a .+ b, ch1, ch2)
    hb = scaled_waveform_model(best_fit, freqs, wp)
    bf = project_to_tdi(hb, freqs, best_fit, wp)

    dens(x, i) = 4 * abs2(x) / Sn[i]
    n = length(freqs)
    # log-uniform decimation: ~equal plotted points per decade, and the first/
    # last plotted frequencies sit at the band ends. (Linear windows left a
    # half-window gap at the low end of the log axis and compressed the first
    # decade into a handful of points.) Log-sparse low-frequency windows hold
    # single bins and pass them through unaveraged; empty windows are skipped.
    nwin = min(n_windows, n)
    edges = 10.0 .^ range(log10(freqs[1]), log10(freqs[end]), nwin + 1)
    window_bounds = [searchsortedfirst(freqs, e) for e in edges]
    window_bounds[end] = n + 1
    windows = [
        window_bounds[i]:(window_bounds[i+1]-1) for
        i in 1:nwin if window_bounds[i+1] > window_bounds[i]
    ]
    agg(v, stat) = [stat(view(v, r)) for r in windows]

    cols = Dict{Symbol,Vector{Float64}}(:f => agg(freqs, mean))
    for (tag, cA, cE) in (("sig", data[1], data[2]), ("bf", bf[1], bf[2]),
        ("res", data[1] .- bf[1], data[2] .- bf[2]))
        for (ch, arr) in (("A", cA), ("E", cE))
            d = [dens(arr[i], i) for i in 1:n]
            cols[Symbol("$(tag)_mean_$ch")] = agg(d, mean)
            cols[Symbol("$(tag)_min_$ch")] = agg(d, minimum)
            cols[Symbol("$(tag)_max_$ch")] = agg(d, maximum)
        end
    end
    int_A = sum(dens(data[1][i] - bf[1][i], i) for i in 1:n) * df
    int_E = sum(dens(data[2][i] - bf[2][i], i) for i in 1:n) * df
    order = [:f; sort(collect(keys(delete!(copy(cols), :f))))]
    return DataFrame([c => cols[c] for c in order]), (int_A = int_A, int_E = int_E)
end

end # module
