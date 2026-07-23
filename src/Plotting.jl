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

Ticks at rational multiples of π with a *single* denominator so that 5–9
(or, failing that, 3–9) uniformly spaced ticks fit in `[lo, hi]`.
Denominators are tried in readability order — halves/quarters before
thirds — so a ±π range gets `π/4` ticks rather than the unusual `π/3`
family. Returns `nothing` when no denominator fits (caller falls back to
linear ticks).
"""
function pi_ticks(lo::Real, hi::Real)
    for wanted in (5:9, 3:9), den in (4, 2, 6, 8, 1, 3, 12)
        step = π / den
        kmin = ceil(Int, lo / step - 1e-9)
        kmax = floor(Int, hi / step + 1e-9)
        n = kmax - kmin + 1
        if n in wanted
            return [k * step for k in kmin:kmax], [pi_label(k, den) for k in kmin:kmax]
        end
    end
    return nothing
end

"""
    sci_latex(v; sig = 3) -> String

LaTeX fragment for a scalar: plain `%.4g` when the exponent is small,
`m×10^e` otherwise — for annotations, never bare `1e-05` e-notation.
"""
function sci_latex(v::Real; sig::Int = 3)
    v == 0 && return "0"
    isfinite(v) || return string(v)
    e = floor(Int, log10(abs(v)))
    -3 <= e <= 3 && return @sprintf("%.4g", round(v, sigdigits = sig + 1))
    m = round(v / 10.0^e, sigdigits = sig)
    return string(@sprintf("%g", m), "\\times 10^{", e, "}")
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
    offset_ticks(lo, hi) -> (values, labels, exponent, lo_snap, hi_snap) or nothing

Tick selection for small-value linear axes: every tick is an **integer
mantissa** of one common power of 10 (the `exponent`, annotated once at the
end of the axis by the caller), with mantissa steps preferring multiples of
5 over 2 over 1, and the axis limits snapped *outward* to the outermost
ticks so the frame ends exactly on labelled ticks. Two acceptance passes
(4–8 ticks with ≤35% stretch, then 3–9 with ≤60%) keep the stretching
modest; returns `nothing` when no clean grid exists (caller falls back to
per-tick scientific notation).
"""
function offset_ticks(lo::Real, hi::Real)
    span = hi - lo
    (isfinite(span) && span > 0) || return nothing
    kmid = floor(Int, log10(span))
    for (nrange, cap) in ((4:8, 0.35), (3:9, 0.60))
        for q in (5, 2, 1), k in (kmid - 2):(kmid + 1)
            step = q * 10.0^k
            lo_s = floor(lo / step + 1e-9) * step
            hi_s = ceil(hi / step - 1e-9) * step
            n = round(Int, (hi_s - lo_s) / step) + 1
            n in nrange || continue
            (hi_s - hi) + (lo - lo_s) <= cap * span || continue
            vals = [lo_s + i * step for i in 0:(n - 1)]
            labels = [latexstring(string(round(Int, v / 10.0^k))) for v in vals]
            return (vals, labels, k, lo_s, hi_s)
        end
    end
    return nothing
end

"""
    sci_tick_labels(values) -> Vector{LaTeXString}

Per-tick scientific-notation labels with a **common exponent** across the
axis (e.g. `-1×10⁻³, -0.5×10⁻³, 0, 0.5×10⁻³, 1×10⁻³`) — the fallback for
small-value axes where `offset_ticks` finds no clean integer-mantissa grid.
The exponent is the *same* for every tick, so an axis never mixes 10⁻³
with 10⁻⁴ labels.
"""
function sci_tick_labels(values)
    maxabs = maximum(abs, values; init = 0.0)
    maxabs <= 0 && return [L"0" for _ in values]
    e = floor(Int, log10(maxabs))
    return map(values) do v
        abs(v) < 1e-300 && return L"0"
        m = round(v / 10.0^e, sigdigits = 3)
        latexstring(@sprintf("%g", m), "\\times 10^{", e, "}")
    end
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
            # top-right, clear of the off-scale floor markers at the top-left
            text!(ax2, maximum(deltas), 1.98;
                  text = latexstring("1 + c_1\\delta + c_2\\delta^2,\\;\\; c_1 = ",
                                     sci_latex(c1)),
                  align = (:right, :top), fontsize = 16, color = :purple)
        end
        # excluded (floor-dominated) points: in-range ones as faint grey dots,
        # divergent ones as open triangles pinned at the top = "off scale above"
        excl = .!clean
        offscale = excl .& (ratio .> 2.05)
        inrange = excl .& .!offscale
        any(inrange) && scatter!(ax2, deltas[inrange], ratio[inrange];
                                 color = (:grey, 0.55), markersize = 12)
        any(offscale) && scatter!(ax2, deltas[offscale], fill(2.02, count(offscale));
                                  marker = :utriangle, color = :transparent,
                                  strokecolor = :grey45, strokewidth = 1.6, markersize = 14)
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
two-source data and the best-fit single source, bottom panel the unabsorbed
residual `d(D²)/df` — the integral of the bottom curves is the D² of the
scaling law. One grouped legend sits on top of the figure (channel A and
channel E blocks, each with data / best fit / residual). Channel encodes hue
(A blue, E warm); the best fit, which lies on top of the data, is a brighter
dash-dotted line over the dark solid data line; min/max decimation envelopes
shade both channels. Frame limits hug the plotted data. `spec` is the
(log-uniformly decimated) spectrum table; `meta` carries `delta_star`, `df`,
and the integral annotations.
"""
function residual_figure(spec, meta)
    with_theme(twd_theme()) do
        fig = Figure(size = (950, 880))
        # Limits and ticks hug the plotted data: with log-uniform decimation
        # the first/last plotted frequencies sit at the band ends, so the
        # frame ends on the data with no gap at either side.
        fmin = minimum(spec.f)
        fmax = maximum(spec.f)
        xt = decade_ticks(fmin, fmax)

        # channel = hue (A blue, E warm), role = shade + line style: data is
        # the dark solid line, the best fit — which sits right on top of it —
        # is a brighter dash-dotted line over it, the residual a medium solid
        col_data_A, col_bf_A, col_res_A = :steelblue4, :deepskyblue, :dodgerblue2
        col_data_E, col_bf_E, col_res_E = :sienna4, :orange, :darkorange3

        ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
                   ylabel = L"\mathrm{d}(\mathrm{SNR}^2)/\mathrm{d}f\ \ [\mathrm{Hz}^{-1}]",
                   xticks = xt, yticklabelspace = 70.0)
        band!(ax1, spec.f, spec.sig_min_A, spec.sig_max_A; color = (col_data_A, 0.14))
        band!(ax1, spec.f, spec.sig_min_E, spec.sig_max_E; color = (col_data_E, 0.14))
        dA = lines!(ax1, spec.f, spec.sig_rms_A; color = col_data_A, linewidth = 3.6)
        dE = lines!(ax1, spec.f, spec.sig_rms_E; color = col_data_E, linewidth = 3.6)
        bA = lines!(ax1, spec.f, spec.bf_rms_A; color = col_bf_A,
                    linestyle = :dashdot, linewidth = 3.0)
        bE = lines!(ax1, spec.f, spec.bf_rms_E; color = col_bf_E,
                    linestyle = :dashdot, linewidth = 3.0)

        # y-range of the residual panel comes from the residual DATA — the
        # per-bin noise reference 1/Δf can sit many decades above the curves,
        # and forcing it into frame would crush the physics into a thin band.
        res_pos = filter(>(0), vcat(spec.res_rms_A, spec.res_rms_E))
        ylo2 = minimum(res_pos) / 6
        yhi2 = max(maximum(spec.res_max_A), maximum(res_pos)) * 6
        noise_level = 1.0 / meta.df
        noise_in_frame = noise_level < 30 * yhi2
        noise_in_frame && (yhi2 = max(yhi2, 3 * noise_level))

        ax2 = Axis(fig[2, 1]; xscale = log10, yscale = log10,
                   xlabel = L"f\ \ [\mathrm{Hz}]",
                   ylabel = L"\mathrm{d}(D^2)/\mathrm{d}f\ \ [\mathrm{Hz}^{-1}]",
                   xticks = xt, yticks = decade_ticks(ylo2, yhi2),
                   yticklabelspace = 70.0)
        band!(ax2, spec.f, max.(spec.res_min_A, 1e-300), spec.res_max_A;
              color = (col_res_A, 0.16))
        band!(ax2, spec.f, max.(spec.res_min_E, 1e-300), spec.res_max_E;
              color = (col_res_E, 0.16))
        rA = lines!(ax2, spec.f, spec.res_rms_A; color = col_res_A, linewidth = 3.2)
        rE = lines!(ax2, spec.f, spec.res_rms_E; color = col_res_E, linewidth = 3.2)

        # one grouped legend on top of the figure (title position), spanning
        # both panels: channel A and channel E blocks with data/best fit/
        # residual entries each
        Legend(fig[0, 1],
               [[dA, bA, rA], [dE, bE, rE]],
               [["data", "best fit", "residual"], ["data", "best fit", "residual"]],
               ["channel A", "channel E"];
               orientation = :horizontal, titleposition = :left,
               framevisible = false, tellwidth = false, tellheight = true,
               labelsize = 18, titlesize = 19, titlefont = :bold,
               patchsize = (26, 4), groupgap = 18, patchlabelgap = 4,
               colgap = 10, titlegap = 8, padding = (0, 0, 4, 0))
        if noise_in_frame
            hlines!(ax2, [noise_level]; color = :grey35, linewidth = 1.8, linestyle = :dot)
            text!(ax2, fmax, noise_level; text = "per-bin noise level",
                  align = (:right, :top), offset = (-6, -4), fontsize = 16, color = :grey35)
        else
            # reference is off scale — state its value instead of distorting the
            # axis; second annotation line under the δ*/integral line (top-left)
            text!(ax2, 0.03, 0.81; space = :relative,
                  text = latexstring("\\mathrm{noise\\ level\\ per\\ bin:}\\ 1/\\Delta f = ",
                                     sci_latex(noise_level),
                                     "\\ \\mathrm{Hz^{-1}}\\ \\mathrm{(off\\ scale)}"),
                  align = (:left, :center), fontsize = 15, color = :grey35)
        end
        # δ*/integral annotation top-left: the residual curves rise towards
        # high f, so the upper-left region is free once the y-range is tight.
        text!(ax2, 0.03, 0.90; space = :relative,
              text = latexstring("\\delta^* = ", sci_latex(meta.delta_star),
                                 ";\\;\\; \\int\\!\\mathrm{d}f = D^2 = ", sci_latex(meta.d2_num),
                                 "\\;\\; (\\mathrm{theory}\\ ", sci_latex(meta.d2_theo), ")"),
              align = (:left, :center), fontsize = 17)
        ylims!(ax2, ylo2, yhi2)

        linkxaxes!(ax1, ax2)
        xlims!(ax2, fmin, fmax) # frame ends on the data — no gap at either side
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
                degenerate_frac, x_math, y_math) -> Figure

Zone-of-confusion map from the capped boundary polygon. The filled zone is
the exact intersection of the mathematical zone with the physical prior box,
and its boundary is drawn SOLID throughout — blue where curvature-limited,
red where it runs along a hard physical wall (|χ|≤1, positivity, phase ±π).
Where the prior cuts the zone off, the *uncapped mathematical* contour
(`x_math`/`y_math`, when provided) continues past the wall as an empty
dashed line with no fill — showing what curvature alone would allow; along
degenerate directions it is unbounded and simply runs off the frame. The
prior box itself is not drawn: the red runs mark where a wall is active, and
the annotation names the active walls with their values (from `box`, e.g.
"… prior-limited by Δχ₁ = 0.2"). `DataAspect` is applied only for same-unit
planes (spin–spin).
"""
function zone_figure(angle::AbstractVector, x::AbstractVector, y::AbstractVector,
                     prior_limited::AbstractVector{Bool};
                     px::Int, py::Int, box::NTuple{4,Float64},
                     prior_frac::Real, degenerate_frac::Real,
                     x_math = nothing, y_math = nothing)
    px == py && error("map plane must use two distinct parameters")
    same_units = (px in (5, 6) && py in (5, 6))
    with_theme(twd_theme()) do
        # squarer canvas for same-unit (DataAspect) planes to avoid wide side margins
        fig = Figure(size = same_units ? (820, 830) : (960, 720))

        # Limits fit the ZONE, not a symmetric ±max box: prior-capped zones are
        # strongly asymmetric (e.g. spins live in the lower-left wedge) and
        # symmetric limits waste most of the canvas on empty quadrants.
        xlo_d, xhi_d = extrema(x)
        ylo_d, yhi_d = extrema(y)
        # extend the view towards the uncapped mathematical contour so its
        # dashed continuation past the prior wall is visible — but never by
        # more than ~40% of the physical zone's span per side (degenerate
        # directions are unbounded; their dashed arcs just exit the frame)
        if x_math !== nothing
            xr = xhi_d - xlo_d
            yr = yhi_d - ylo_d
            fx = filter(isfinite, x_math)
            fy = filter(isfinite, y_math)
            if !isempty(fx) && !isempty(fy)
                xlo_d = max(min(xlo_d, minimum(fx)), xlo_d - 0.4 * xr)
                xhi_d = min(max(xhi_d, maximum(fx)), xhi_d + 0.4 * xr)
                ylo_d = max(min(ylo_d, minimum(fy)), ylo_d - 0.4 * yr)
                yhi_d = min(max(yhi_d, maximum(fy)), yhi_d + 0.4 * yr)
            end
        end
        ex = axis_exponent(max(abs(xlo_d), abs(xhi_d)))
        ey = axis_exponent(max(abs(ylo_d), abs(yhi_d)))

        ax = Axis(fig[1, 1]; xlabel = deviation_label(px), ylabel = deviation_label(py))
        same_units && (ax.aspect = DataAspect())

        # Per-axis ticks and final limits. Phase axes: rational-π ticks over a
        # padded range. Small-value axes: integer-mantissa ticks of ONE common
        # power of 10 — the power annotated once at the end of the axis, never
        # per tick and never inside the axis label — with mantissa steps
        # preferring multiples of 5 and the limits snapped outward so the
        # frame ends exactly on labelled ticks. Fallback when no clean grid
        # exists: per-tick common-exponent scientific notation.
        xlo, xhi = xlo_d - 0.08 * (xhi_d - xlo_d), xhi_d + 0.08 * (xhi_d - xlo_d)
        ylo, yhi = ylo_d - 0.08 * (yhi_d - ylo_d), yhi_d + 0.08 * (yhi_d - ylo_d)
        if px == 4
            pt = pi_ticks(xlo, xhi)
            pt !== nothing && (ax.xticks = pt)
        elseif ex != 0
            ot = offset_ticks(xlo_d, xhi_d)
            if ot === nothing
                ax.xticks = LinearTicks(6)
                ax.xtickformat = sci_tick_labels
            else
                vals, labels, e10, lo_s, hi_s = ot
                ax.xticks = (vals, labels)
                xlo, xhi = lo_s, hi_s
                # common power of 10 at the end of the x axis: just right of
                # the frame's bottom corner, clear of the last tick label
                e10 != 0 && Label(fig[1, 2], latexstring("\\times 10^{", e10, "}");
                                  fontsize = 20, halign = :left, valign = :bottom,
                                  padding = (2, 0, 0, 0), tellheight = false)
            end
        else
            ax.xticks = LinearTicks(6)
        end
        if py == 4
            pt = pi_ticks(ylo, yhi)
            pt !== nothing && (ax.yticks = pt)
        elseif ey != 0
            ot = offset_ticks(ylo_d, yhi_d)
            if ot === nothing
                ax.yticks = LinearTicks(6)
                ax.ytickformat = sci_tick_labels
            else
                vals, labels, e10, lo_s, hi_s = ot
                ax.yticks = (vals, labels)
                ylo, yhi = lo_s, hi_s
                # common power of 10 at the end of the y axis: above the frame
                e10 != 0 && Label(fig[0, 1], latexstring("\\times 10^{", e10, "}");
                                  fontsize = 20, halign = :left, valign = :bottom,
                                  padding = (0, 0, 2, 0), tellwidth = false)
            end
        else
            ax.yticks = LinearTicks(6)
        end

        poly!(ax, Point2f.(x, y); color = (:dodgerblue, 0.30), strokewidth = 0)

        # The boundary of the filled (physical) zone is SOLID throughout:
        # curvature-limited runs in blue, prior-limited runs in red along the
        # hard physical walls. Where the prior cuts the zone off, the uncapped
        # MATHEMATICAL contour continues past the wall as an empty dashed line
        # (no fill) — drawn first so the solid boundary sits on top of the
        # junctions.
        n = length(x)
        edge_prior = [prior_limited[i] && prior_limited[mod1(i + 1, n)] for i in 1:n]
        if x_math !== nothing && any(prior_limited)
            # dilate the mask by one edge per side so each dashed arc joins the
            # solid contour at the capping crossover (where r_math = r_capped);
            # non-finite (degenerate) points break the polyline
            xm = [isfinite(v) ? Float64(v) : NaN for v in x_math]
            ym = [isfinite(v) ? Float64(v) : NaN for v in y_math]
            edge_math = [prior_limited[i] || prior_limited[mod1(i + 1, n)] for i in 1:n]
            mx, my = _boundary_runs(xm, ym, edge_math, true)
            isempty(mx) || lines!(ax, mx, my; color = (:dodgerblue4, 0.75),
                                  linewidth = 2.4, linestyle = :dash)
        end
        cx, cy = _boundary_runs(x, y, edge_prior, false)
        bx, by = _boundary_runs(x, y, edge_prior, true)
        isempty(cx) || lines!(ax, cx, cy; color = :dodgerblue4, linewidth = 3.0)
        isempty(bx) || lines!(ax, bx, by; color = :firebrick, linewidth = 3.0)

        xlims!(ax, xlo, xhi)
        ylims!(ax, ylo, yhi)

        if prior_frac > 0
            # never round a nonzero fraction to "0%": the capped boundary run
            # can be visually dominant yet contain few sampled directions
            # (adaptive refinement leaves flat prior walls sparsely sampled)
            pcttex(f) = 100f < 0.5 ? "{<}1\\%" : @sprintf("%.0f\\%%", 100f)
            # name the ACTIVE walls with their values: a prior-limited boundary
            # point sits exactly on the box edge its ray exited through, so an
            # edge is active iff some capped point lies on it
            devtex(idx) = string("\\Delta ", PARAM_LABELS[idx][2:end-1])
            fmt_edge(v, isphase) = isphase && isapprox(abs(v), π; atol = 1e-9) ?
                                   (v < 0 ? "-\\pi" : "\\pi") : sci_latex(v)
            plx = view(x, prior_limited)
            ply = view(y, prior_limited)
            lox, hix, loy, hiy = box
            walls = String[]
            for (lo_e, hi_e, vals, tol, idx) in ((lox, hix, plx, 1e-6 * (xhi - xlo), px),
                                                 (loy, hiy, ply, 1e-6 * (yhi - ylo), py))
                lo_hit = isfinite(lo_e) && any(v -> abs(v - lo_e) < tol, vals)
                hi_hit = isfinite(hi_e) && any(v -> abs(v - hi_e) < tol, vals)
                isph = idx == 4
                if lo_hit && hi_hit && isapprox(lo_e, -hi_e; rtol = 1e-9)
                    push!(walls, string(devtex(idx), " = \\pm ", fmt_edge(abs(hi_e), isph)))
                else
                    lo_hit && push!(walls, string(devtex(idx), " = ", fmt_edge(lo_e, isph)))
                    hi_hit && push!(walls, string(devtex(idx), " = ", fmt_edge(hi_e, isph)))
                end
            end
            # \!-\! cancels the binary-operator spacing MathTeXEngine would put
            # around the hyphen in "prior-limited"
            msg = string(pcttex(prior_frac),
                         "\\ \\mathrm{of\\ directions\\ prior}\\!-\\!\\mathrm{limited}")
            isempty(walls) || (msg *= string("\\ \\mathrm{by}\\ ", join(walls, ",\\;\\ ")))
            degenerate_frac > 0 &&
                (msg *= string("\\;\\ (", pcttex(degenerate_frac), "\\ \\mathrm{degenerate})"))
            # on top of the plot, outside the box (a thin Label row above the
            # axis); tellwidth = false so the label's own width never dictates
            # the column width — otherwise the axis collapses to a narrow strip
            Label(fig[0, 1], latexstring(msg); fontsize = 18, color = :grey35,
                  halign = :right, padding = (0, 4, 2, 0), tellwidth = false)
        end
        return fig
    end
end

end # module
