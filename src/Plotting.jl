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
        fontsize = 22,
        figure_padding = 16,
        Axis = (
            xgridstyle = :dash, ygridstyle = :dash,
            xgridcolor = (:grey, 0.12), ygridcolor = (:grey, 0.12),
            xminorticksvisible = false, yminorticksvisible = false,
            xtickalign = 1, ytickalign = 1,
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

function scaled_axis_label(idx::Int, e::Int)
    base = deviation_label(idx)
    return e == 0 ? base : latexstring(base[2:end-1], "\\;(\\times 10^{", e, "})")
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
        fig = Figure(size = (900, 850))

        pos = d2_num .> 0
        ylo = min(minimum(d2_num[pos]), minimum(d2_theo)) / 3
        yhi = max(maximum(d2_num[pos]), maximum(d2_theo)) * 3
        xt = decade_ticks(minimum(deltas), maximum(deltas))
        yt = decade_ticks(ylo, yhi)

        ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                   ylabel = L"D^2", xticks = xt, yticks = yt)
        ylims!(ax1, ylo, yhi)

        if isfinite(floor_level) && floor_level > ylo
            hspan!(ax1, ylo, floor_level; color = (:grey, 0.12))
            text!(ax1, minimum(deltas), floor_level;
                  text = "optimizer floor", align = (:left, :bottom),
                  fontsize = 14, color = :grey35)
        end
        if ylo < rho_sq < yhi
            hlines!(ax1, [rho_sq]; color = :grey35, linewidth = 1.2)
            text!(ax1, maximum(deltas), rho_sq; text = L"\rho^2_{thr}",
                  align = (:right, :bottom), fontsize = 15, color = :grey35)
        end
        if minimum(deltas) < delta_min < maximum(deltas)
            vlines!(ax1, [delta_min]; color = :grey35, linewidth = 1.2, linestyle = :dash)
            text!(ax1, delta_min, ylo; text = L"\delta_{min}",
                  align = (:left, :bottom), fontsize = 15, color = :grey35)
        end

        lines!(ax1, deltas, d2_theo; color = :crimson, linestyle = :dash, linewidth = 2.5,
               label = L"(1/16)\, K(u)\, \delta^4")
        scatter!(ax1, deltas[pos], d2_num[pos]; color = :dodgerblue, strokecolor = :black,
                 strokewidth = 0.8, markersize = 11, label = "numerical optimization")
        axislegend(ax1; position = :lt)
        text!(ax1, maximum(deltas), ylo;
              text = @sprintf("fitted slope: %.3f ± %.3f", slope, slope_err),
              align = (:right, :bottom), fontsize = 16)

        ax2 = Axis(fig[2, 1]; xscale = log10,
                   xlabel = L"\mathrm{parameter\ separation}\ \delta",
                   ylabel = L"D^2_{num}/D^2_{theo}", xticks = xt)
        ratio = d2_num ./ d2_theo
        hlines!(ax2, [1.0]; color = :grey35, linewidth = 1.2)
        if isfinite(c1)
            dd = 10.0 .^ range(log10(minimum(deltas)), log10(maximum(deltas)), length = 120)
            model = 1.0 .+ c1 .* dd .+ (isfinite(c2) ? c2 : 0.0) .* dd .^ 2
            lines!(ax2, dd, clamp.(model, 0.0, 2.1); color = (:grey35, 0.8), linewidth = 1.4)
            text!(ax2, minimum(deltas), 2.02;
                  text = @sprintf("1 + c₁δ + c₂δ²,  c₁ = %.3g", c1),
                  align = (:left, :top), fontsize = 14, color = :grey35)
        end
        excl = .!clean
        any(excl) && scatter!(ax2, deltas[excl], clamp.(ratio[excl], 0.0, 2.05);
                              color = (:grey, 0.55), markersize = 9)
        scatter!(ax2, deltas[clean], ratio[clean]; color = :dodgerblue,
                 strokecolor = :black, strokewidth = 0.8, markersize = 9)
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
        xt = decade_ticks(minimum(spec.f), maximum(spec.f))

        ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                   ylabel = L"d(\mathrm{SNR}^2)/df\ \ [\mathrm{Hz}^{-1}]", xticks = xt)
        band!(ax1, spec.f, spec.sig_min_A, spec.sig_max_A; color = (:black, 0.18))
        lines!(ax1, spec.f, spec.sig_rms_A; color = :black, linewidth = 2.0,
               label = "data, channel A")
        lines!(ax1, spec.f, spec.sig_rms_E; color = :grey55, linewidth = 2.0,
               label = "data, channel E")
        lines!(ax1, spec.f, spec.bf_rms_A; color = :dodgerblue, linestyle = :dash,
               linewidth = 2.2, label = "best fit, channel A")
        lines!(ax1, spec.f, spec.bf_rms_E; color = :steelblue4, linestyle = :dash,
               linewidth = 2.2, label = "best fit, channel E")
        axislegend(ax1; position = :rt)

        ax2 = Axis(fig[2, 1]; xscale = log10, yscale = log10,
                   xlabel = L"f\ \ [\mathrm{Hz}]",
                   ylabel = L"d(D^2)/df\ \ [\mathrm{Hz}^{-1}]", xticks = xt)
        band!(ax2, spec.f, max.(spec.res_min_A, 1e-300), spec.res_max_A; color = (:crimson, 0.2))
        lines!(ax2, spec.f, spec.res_rms_A; color = :crimson, linewidth = 2.0,
               label = "residual, channel A")
        lines!(ax2, spec.f, spec.res_rms_E; color = :darkorange3, linewidth = 2.0,
               label = "residual, channel E")
        noise_level = 1.0 / meta.df
        hlines!(ax2, [noise_level]; color = :grey35, linewidth = 1.4, linestyle = :dot)
        text!(ax2, maximum(spec.f), noise_level; text = "per-bin noise level",
              align = (:right, :top), fontsize = 14, color = :grey35)
        axislegend(ax2; position = :rb)
        text!(ax2, minimum(spec.f), noise_level;
              text = @sprintf("δ* = %.3g;  ∫df = D² = %.3g  (theory %.3g)",
                              meta.delta_star, meta.d2_num, meta.d2_theo),
              align = (:left, :top), fontsize = 15)

        linkxaxes!(ax1, ax2)
        hidexdecorations!(ax1; grid = false, ticks = false)
        return fig
    end
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
    with_theme(twd_theme()) do
        fig = Figure(size = (900, 700))

        lox, hix, loy, hiy = box
        xmax = maximum(abs, x)
        ymax = maximum(abs, y)
        ex = axis_exponent(xmax)
        ey = axis_exponent(ymax)

        ax = Axis(fig[1, 1]; xlabel = scaled_axis_label(px, ex), ylabel = scaled_axis_label(py, ey))
        px == py && error("map plane must use two distinct parameters")
        same_units = (px in (5, 6) && py in (5, 6))
        same_units && (ax.aspect = DataAspect())

        # π ticks for phase axes, common-exponent ticks otherwise
        if px == 4
            pt = pi_ticks(-1.06 * xmax, 1.06 * xmax)
            pt !== nothing && (ax.xticks = pt)
        elseif ex != 0
            ax.xtickformat = scaled_tickformat(ex)
        end
        if py == 4
            pt = pi_ticks(-1.06 * ymax, 1.06 * ymax)
            pt !== nothing && (ax.yticks = pt)
        elseif ey != 0
            ax.ytickformat = scaled_tickformat(ey)
        end

        poly!(ax, Point2f.(x, y); color = (:dodgerblue, 0.30), strokewidth = 0)

        # split the closed boundary into curvature-limited / prior-limited runs
        n = length(x)
        curv_x = Float64[]; curv_y = Float64[]
        prior_x = Float64[]; prior_y = Float64[]
        for i in 1:n
            j = mod1(i + 1, n)
            seg_prior = prior_limited[i] && prior_limited[j]
            tx, ty = seg_prior ? (prior_x, prior_y) : (curv_x, curv_y)
            push!(tx, x[i]); push!(tx, x[j]); push!(tx, NaN)
            push!(ty, y[i]); push!(ty, y[j]); push!(ty, NaN)
        end
        isempty(curv_x) || lines!(ax, curv_x, curv_y; color = :dodgerblue4, linewidth = 2.2)
        isempty(prior_x) || lines!(ax, prior_x, prior_y; color = :firebrick, linewidth = 2.6)

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
            # relative axis coordinates keep the annotation inside the frame
            text!(ax, 0.985, 0.015; text = msg, space = :relative,
                  align = (:right, :bottom), fontsize = 14, color = :grey35)
        end
        return fig
    end
end

end # module
