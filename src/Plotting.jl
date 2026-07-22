module Plotting

using CairoMakie
using MathTeXEngine
using LaTeXStrings
using Printf
using Statistics
using ..Provenance: backup_existing!

export twd_theme, save_figure, decade_ticks, pi_ticks, axis_exponent,
       scaled_tickformat, scaling_figure, residual_figure, zone_figure,
       PARAM_LABELS

"""
Short LaTeX axis labels for the six model parameters (deviation form is
composed by the figure builders).
"""
const PARAM_LABELS = (L"\mathcal{A}", L"\mathcal{M}", L"t_c", L"\Phi_0", L"\chi_1", L"\chi_2")

deviation_label(idx) = latexstring("\\Delta ", PARAM_LABELS[idx][2:end-1])

"""
    twd_theme()

Publication theme (Computer Modern via MathTeXEngine, boxed axes, dashed
low-opacity grey grid, no minor ticks, inward ticks, generous padding).
All figure builders apply it via `with_theme`.
"""
function twd_theme()
    return Theme(
        fonts = (; regular = texfont(:text), bold = texfont(:bold), italic = texfont(:italic)),
        fontsize = 25,
        figure_padding = 18,
        Axis = (
            xgridstyle = :dash, ygridstyle = :dash,
            xgridcolor = (:grey, 0.12), ygridcolor = (:grey, 0.12),
            xminorticksvisible = false, yminorticksvisible = false,
            xtickalign = 1, ytickalign = 1,
            xlabelpadding = 12, ylabelpadding = 14, # more distance from tick labels
        ),
        Legend = (framevisible = false, backgroundcolor = :transparent),
    )
end

"""
    save_figure(fig, base_path)

Save `fig` as both vector `.pdf` and raster `.png` (`px_per_unit = 4`),
with `safesave`-style backup of any existing files.
"""
function save_figure(fig, base_path::AbstractString)
    pdf = backup_existing!(base_path * ".pdf")
    save(pdf, fig)
    png = backup_existing!(base_path * ".png")
    save(png, fig; px_per_unit = 4)
    return (pdf, png)
end

# --- tick utilities ---------------------------------------------------------

"""
    decade_ticks(lo, hi; maxticks = 7) -> (values, labels)

Integer power-of-10 ticks covering `[lo, hi]` only (no beyond-range ticks).
The step is chosen so at most `maxticks` ticks appear, and tick exponents are
anchored to multiples of the step — so every figure in a family whose data
share decades shares the same grid.
"""
function decade_ticks(lo::Real, hi::Real; maxticks::Int = 7)
    (lo > 0 && hi > 0 && isfinite(lo) && isfinite(hi)) ||
        throw(ArgumentError("decade_ticks requires finite positive bounds, got ($lo, $hi)"))
    pmin = ceil(Int, log10(lo) - 1e-9)
    pmax = floor(Int, log10(hi) + 1e-9)
    if pmax < pmin # no integer decade inside the range
        p = round(Int, log10(sqrt(lo * hi)))
        return [10.0^p], [L"10^{%$p}"]
    end
    step = max(1, ceil(Int, (pmax - pmin + 1) / maxticks))
    first_p = step * cld(pmin, step)
    ps = collect(first_p:step:pmax)
    isempty(ps) && (ps = [pmin])
    return 10.0 .^ ps, [L"10^{%$p}" for p in ps]
end

"""
    pi_ticks(lo, hi) -> (values, labels) or nothing

Ticks at rational multiples of π with a *single* denominator chosen from
(1, 2, 3, 4, 6, 8, 12) so that 3–7 uniformly spaced ticks fit in `[lo, hi]`.
Returns `nothing` when no denominator fits (caller falls back to linear
ticks).
"""
function pi_ticks(lo::Real, hi::Real)
    for den in (1, 2, 3, 4, 6, 8, 12)
        step = π / den
        kmin = ceil(Int, lo / step - 1e-9)
        kmax = floor(Int, hi / step + 1e-9)
        n = kmax - kmin + 1
        if 3 <= n <= 7
            return [k * step for k in kmin:kmax], [pi_label(k, den) for k in kmin:kmax]
        end
    end
    return nothing
end

function pi_label(k::Integer, den::Integer)
    k == 0 && return L"0"
    g = gcd(abs(k), den)
    num = abs(k) ÷ g
    d = den ÷ g
    sign = k < 0 ? "-" : ""
    core = num == 1 ? "\\pi" : "$(num)\\pi"
    return d == 1 ? latexstring(sign, core) : latexstring(sign, core, "/", d)
end

"""
    axis_exponent(maxabs) -> Int

Common decimal exponent for a linear axis whose data extend to `maxabs`;
0 when plain labels are fine (|values| in [1e-2, 1e4)).
"""
function axis_exponent(maxabs::Real)
    (maxabs <= 0 || !isfinite(maxabs)) && return 0
    e = floor(Int, log10(maxabs))
    return -2 <= e <= 3 ? 0 : e
end

"""
    scaled_tickformat(e) -> Function

Makie tick formatter dividing values by `10^e` (single per-axis exponent —
never mixed exponents on one axis), 2–3 significant digits.
"""
scaled_tickformat(e::Int) = values -> [@sprintf("%.3g", v / 10.0^e) for v in values]

"""
    sci_tick_labels(values) -> Vector{LaTeXString}

Per-tick scientific-notation labels (e.g. `1.5×10⁻⁴`). Used on confusion-map
axes whose deviations are very small. Deliberately puts the exponent on each
tick *individually* rather than factoring a common `(×10ⁿ)` into the axis
label: the parameters are O(1)-rescaled Fisher coordinates, and a common
axis multiplier would read as a physical rescaling and mislead the reader.
"""
function sci_tick_labels(values)
    out = LaTeXString[]
    for v in values
        if abs(v) < 1e-300
            push!(out, L"0")
        else
            p = floor(Int, log10(abs(v)))
            m = round(v / 10.0^p, sigdigits = 2)
            push!(out, latexstring(@sprintf("%g", m), "\\times 10^{", p, "}"))
        end
    end
    return out
end

# --- figure builders --------------------------------------------------------

"""
    scaling_figure(deltas, d2_num, d2_theo; rho_sq, delta_min, slope, slope_err,
                   clean, floor_level) -> Figure

Quartic-scaling validation figure: log–log D²(δ) with the δ⁴ prediction, the
`D² = ρ²` threshold line and δ_min marker, a shaded optimizer-floor band, and
a linked ratio panel `D²_num/D²_theo` that makes prefactor agreement and
higher-order departures visible. `clean` is the Bool mask of points used for
the annotated slope fit.
"""
function scaling_figure(deltas::AbstractVector, d2_num::AbstractVector, d2_theo::AbstractVector;
                        rho_sq::Real, delta_min::Real, slope::Real, slope_err::Real,
                        clean::AbstractVector{Bool}, floor_level::Real,
                        c1::Real = NaN, c2::Real = NaN)
    with_theme(twd_theme()) do
        fig = Figure(size = (920, 900))

        pos = d2_num .> 0
        ylo = min(minimum(d2_num[pos]), minimum(d2_theo)) / 3
        yhi = max(maximum(d2_num[pos]), maximum(d2_theo)) * 3
        xt = decade_ticks(minimum(deltas), maximum(deltas))
        yt = decade_ticks(ylo, yhi)

        ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                   ylabel = L"D^2", xticks = xt, yticks = yt, yticklabelspace = 66.0)

        lines!(ax1, deltas, d2_theo; color = :crimson, linestyle = :dash, linewidth = 3.2,
               label = L"(1/16)\, K(u)\, \delta^4")
        scatter!(ax1, deltas[pos], d2_num[pos]; color = :dodgerblue, strokecolor = :black,
                 strokewidth = 1.2, markersize = 15,
                 label = @sprintf("numerical optimization\nfitted slope %.3f ± %.3f", slope, slope_err))
        ylims!(ax1, ylo, yhi)

        if isfinite(floor_level) && floor_level > ylo
            hspan!(ax1, ylo, floor_level; color = (:grey, 0.13))
            # bottom-right, well clear of the rising δ⁴ line (which is high there)
            text!(ax1, maximum(deltas), floor_level;
                  text = "optimizer floor", align = (:right, :bottom),
                  fontsize = 17, color = :grey35)
        end
        if ylo < rho_sq < yhi
            hlines!(ax1, [rho_sq]; color = :grey35, linewidth = 1.8)
            text!(ax1, maximum(deltas), rho_sq; text = L"\rho^2_{\mathrm{thr}}",
                  align = (:right, :bottom), fontsize = 18, color = :grey35)
        end
        if minimum(deltas) < delta_min < maximum(deltas)
            vlines!(ax1, [delta_min]; color = :grey35, linewidth = 1.8, linestyle = :dash)
            text!(ax1, delta_min, ylo; text = L"\delta_{\mathrm{min}}",
                  align = (:left, :bottom), fontsize = 18, color = :grey35)
        end

        # the fitted slope is part of the numerical-optimization legend entry
        axislegend(ax1; position = :lt)

        ax2 = Axis(fig[2, 1]; xscale = log10,
                   xlabel = L"\mathrm{parameter\ separation}\ \delta",
                   ylabel = L"D^2_{\mathrm{num}}/D^2_{\mathrm{theo}}", xticks = xt,
                   yticklabelspace = 66.0)
        ratio = d2_num ./ d2_theo
        hlines!(ax2, [1.0]; color = :grey45, linewidth = 1.6)
        if isfinite(c1)
            dd = 10.0 .^ range(log10(minimum(deltas)), log10(maximum(deltas)), length = 160)
            model = 1.0 .+ c1 .* dd .+ (isfinite(c2) ? c2 : 0.0) .* dd .^ 2
            lines!(ax2, dd, clamp.(model, 0.0, 2.1); color = (:purple, 0.9), linewidth = 2.6)
            # top-right, clear of the clamped floor points at the top-left
            text!(ax2, maximum(deltas), 2.03;
                  text = @sprintf("1 + c₁δ + c₂δ²,  c₁ = %.3g", c1),
                  align = (:right, :top), fontsize = 16, color = :purple)
        end
        excl = .!clean
        any(excl) && scatter!(ax2, deltas[excl], clamp.(ratio[excl], 0.0, 2.05);
                              color = (:grey, 0.55), markersize = 12)
        scatter!(ax2, deltas[clean], ratio[clean]; color = :dodgerblue,
                 strokecolor = :black, strokewidth = 1.0, markersize = 12)
        ylims!(ax2, 0.0, 2.1)

        linkxaxes!(ax1, ax2)
        hidexdecorations!(ax1; grid = false, ticks = false)
        rowsize!(fig.layout, 1, Relative(0.68))
        return fig
    end
end

"""
    residual_figure(spec, meta) -> Figure

Residual-spectrum figure in true density units: top panel `d(SNR²)/df` of the
two-source data and the best-fit single source (channels A and E, RMS line
with min/max envelope), bottom panel the unabsorbed residual `d(D²)/df` with
the per-bin unit-noise reference `1/df` — the integral of the bottom curves
is the D² of the scaling law. `spec` is the (decimated) spectrum table;
`meta` carries `delta_star`, `df`, and integral annotations.
"""
function residual_figure(spec, meta)
    with_theme(twd_theme()) do
        fig = Figure(size = (950, 850))
        # Tick and limit the frequency axis from the true grid band, not the
        # window-decimated spec.f: the first/last decimation-window means sit
        # inside the band, so ticking off them drops the endpoint decade
        # (e.g. the 1e-4 tick) and stops the frame short of the band edge.
        fmin = get(meta, :f_min, minimum(spec.f))
        fmax = get(meta, :f_max, maximum(spec.f))
        xt = decade_ticks(fmin, fmax)

        ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                   ylabel = L"\mathrm{d}(\mathrm{SNR}^2)/\mathrm{d}f\ \ [\mathrm{Hz}^{-1}]",
                   xticks = xt, yticklabelspace = 70.0)
        band!(ax1, spec.f, spec.sig_min_A, spec.sig_max_A; color = (:black, 0.16))
        lines!(ax1, spec.f, spec.sig_rms_A; color = :black, linewidth = 2.8,
               label = "data, channel A")
        lines!(ax1, spec.f, spec.sig_rms_E; color = :grey55, linewidth = 2.8,
               label = "data, channel E")
        lines!(ax1, spec.f, spec.bf_rms_A; color = :dodgerblue, linestyle = :dash,
               linewidth = 3.0, label = "best fit, channel A")
        lines!(ax1, spec.f, spec.bf_rms_E; color = :steelblue4, linestyle = :dash,
               linewidth = 3.0, label = "best fit, channel E")
        axislegend(ax1; position = :rt)

        ax2 = Axis(fig[2, 1]; xscale = log10, yscale = log10,
                   xlabel = L"f\ \ [\mathrm{Hz}]",
                   ylabel = L"\mathrm{d}(D^2)/\mathrm{d}f\ \ [\mathrm{Hz}^{-1}]",
                   xticks = xt, yticklabelspace = 70.0)
        band!(ax2, spec.f, max.(spec.res_min_A, 1e-300), spec.res_max_A; color = (:crimson, 0.18))
        lines!(ax2, spec.f, spec.res_rms_A; color = :crimson, linewidth = 2.8,
               label = "residual, channel A")
        lines!(ax2, spec.f, spec.res_rms_E; color = :darkorange3, linewidth = 2.8,
               label = "residual, channel E")
        noise_level = 1.0 / meta.df
        hlines!(ax2, [noise_level]; color = :grey35, linewidth = 1.8, linestyle = :dot)
        text!(ax2, fmax, noise_level; text = "per-bin noise level",
              align = (:right, :top), fontsize = 16, color = :grey35)
        axislegend(ax2; position = :rb)
        # δ*/integral annotation in the empty middle band (relative coords):
        # the residual curves sit at the bottom and the per-bin-noise line near
        # the top, so ~60% up the left edge crosses neither.
        text!(ax2, 0.03, 0.62; space = :relative,
              text = @sprintf("δ* = %.3g;  ∫df = D² = %.3g  (theory %.3g)",
                              meta.delta_star, meta.d2_num, meta.d2_theo),
              align = (:left, :center), fontsize = 17)

        linkxaxes!(ax1, ax2)
        xlims!(ax2, fmin, fmax) # frame ends exactly on the band (linked to ax1)
        hidexdecorations!(ax1; grid = false, ticks = false)
        return fig
    end
end

"""
    _boundary_runs(x, y, edge_prior, want) -> (xs, ys)

Coordinates of the closed-boundary edges whose class (`edge_prior[i]`) equals
`want`, grouped into contiguous runs separated by `NaN` so each run renders as
one polyline (letting a dash pattern form over the whole run).
"""
function _boundary_runs(x, y, edge_prior::AbstractVector{Bool}, want::Bool)
    n = length(x)
    xs = Float64[]; ys = Float64[]
    inrun = false
    for i in 1:n
        j = mod1(i + 1, n)
        if edge_prior[i] == want
            if !inrun
                isempty(xs) || (push!(xs, NaN); push!(ys, NaN))
                push!(xs, x[i]); push!(ys, y[i])
                inrun = true
            end
            push!(xs, x[j]); push!(ys, y[j])
        else
            inrun = false
        end
    end
    return xs, ys
end

"""
    zone_figure(angle, x, y, prior_limited; px, py, box, prior_frac,
                degenerate_frac) -> Figure

Zone-of-confusion map from the capped boundary polygon. The filled zone is
the exact intersection of the mathematical zone with the physical prior box;
curvature-limited boundary segments are drawn solid, prior-limited segments
follow the box and are drawn in a distinct style, and the (finite part of
the) prior box itself is dashed — so a bound-limited zone is visually
distinct from a curvature-limited one. `DataAspect` is applied only for
same-unit planes (spin–spin).
"""
function zone_figure(angle::AbstractVector, x::AbstractVector, y::AbstractVector,
                     prior_limited::AbstractVector{Bool};
                     px::Int, py::Int, box::NTuple{4,Float64},
                     prior_frac::Real, degenerate_frac::Real)
    px == py && error("map plane must use two distinct parameters")
    same_units = (px in (5, 6) && py in (5, 6))
    with_theme(twd_theme()) do
        # squarer canvas for same-unit (DataAspect) planes to avoid wide side margins
        fig = Figure(size = same_units ? (820, 830) : (960, 720))

        lox, hix, loy, hiy = box
        xmax = maximum(abs, x)
        ymax = maximum(abs, y)
        ex = axis_exponent(xmax)
        ey = axis_exponent(ymax)

        # No `(×10ⁿ)` multiplier in the axis label — the deviations are
        # O(1)-rescaled Fisher coordinates; a common multiplier would read as a
        # physical rescaling. Small deviations get per-tick scientific notation.
        ax = Axis(fig[1, 1]; xlabel = deviation_label(px), ylabel = deviation_label(py))
        same_units && (ax.aspect = DataAspect())

        # π ticks for phase axes, per-tick scientific notation for small values
        if px == 4
            pt = pi_ticks(-1.06 * xmax, 1.06 * xmax)
            pt !== nothing && (ax.xticks = pt)
        elseif ex != 0
            ax.xtickformat = sci_tick_labels
        end
        if py == 4
            pt = pi_ticks(-1.06 * ymax, 1.06 * ymax)
            pt !== nothing && (ax.yticks = pt)
        elseif ey != 0
            ax.ytickformat = sci_tick_labels
        end

        poly!(ax, Point2f.(x, y); color = (:dodgerblue, 0.30), strokewidth = 0)

        # Split the closed boundary into contiguous curvature-limited /
        # prior-limited RUNS (not per-edge) so a dash pattern forms cleanly
        # over each run. Curvature-limited (non-cutoff) runs are DASHED: the
        # discernibility-threshold edge, soft and, along degenerate directions,
        # unbounded in principle. Prior-limited (cutoff) runs are SOLID: hard
        # physical walls (|χ|≤1, positivity, phase ±π) that terminate the zone.
        n = length(x)
        edge_prior = [prior_limited[i] && prior_limited[mod1(i + 1, n)] for i in 1:n]
        cx, cy = _boundary_runs(x, y, edge_prior, false)
        bx, by = _boundary_runs(x, y, edge_prior, true)
        isempty(cx) || lines!(ax, cx, cy; color = :dodgerblue4, linewidth = 2.6, linestyle = :dash)
        isempty(bx) || lines!(ax, bx, by; color = :firebrick, linewidth = 3.0)

        # the physical prior box (finite edges only)
        isfinite(lox) && vlines!(ax, [lox]; color = :grey35, linestyle = :dash, linewidth = 1.4)
        isfinite(hix) && vlines!(ax, [hix]; color = :grey35, linestyle = :dash, linewidth = 1.4)
        isfinite(loy) && hlines!(ax, [loy]; color = :grey35, linestyle = :dash, linewidth = 1.4)
        isfinite(hiy) && hlines!(ax, [hiy]; color = :grey35, linestyle = :dash, linewidth = 1.4)

        pad = 0.07
        xlims!(ax, -xmax * (1 + pad), xmax * (1 + pad))
        ylims!(ax, -ymax * (1 + pad), ymax * (1 + pad))

        if prior_frac > 0
            msg = @sprintf("%.0f%% of directions prior-limited", 100 * prior_frac)
            degenerate_frac > 0 &&
                (msg *= @sprintf(" (%.0f%% degenerate)", 100 * degenerate_frac))
            # on top of the plot, outside the box (a thin Label row above the axis)
            Label(fig[0, 1], msg; fontsize = 18, color = :grey35, halign = :right,
                  padding = (0, 4, 2, 0))
        end
        return fig
    end
end

end # module
