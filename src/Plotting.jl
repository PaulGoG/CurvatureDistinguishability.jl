"""
Publication CairoMakie figures under one theme and a family-wide tick
policy (log 1-2-5 series, axis offset multipliers, rational-pi ticks).
"""
module Plotting

using DocStringExtensions: TYPEDSIGNATURES
using CairoMakie: Axis, DataAspect, Figure, Label, Legend,
    LinearTicks, Point2f, Relative, Theme, band!,
    hidexdecorations!, hlines!, hspan!, lines!, linkxaxes!,
    poly!, rowgap!, rowsize!, save, scatter!, text!, vlines!,
    with_theme, xlims!, ylims!
using LaTeXStrings: LaTeXStrings, @L_str, latexstring
using MathTeXEngine: texfont
using Printf: @sprintf
using ..Provenance: backup_existing!

export publication_theme, save_figure, scaling_figure, residual_figure, zone_figure
public decade_ticks, pi_ticks

"""
Short LaTeX axis labels for the six model parameters (deviation form is
composed by the figure builders).
"""
const PARAM_LABELS =
    (L"\mathcal{A}", L"\mathcal{M}", L"t_c", L"\Phi_0", L"\chi_1", L"\chi_2")

deviation_label(idx) = latexstring("\\Delta ", PARAM_LABELS[idx][2:(end-1)])

"""
$(TYPEDSIGNATURES)

Publication theme (Computer Modern via MathTeXEngine, boxed axes, dashed
low-opacity grey grid, no minor ticks, inward ticks, generous padding).
All figure builders apply it via `with_theme`.
"""
function publication_theme()
    return Theme(
        fonts = (;
            regular = texfont(:text),
            bold = texfont(:bold),
            italic = texfont(:italic),
        ),
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
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

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
    decade_label(p) = p == 0 ? L"1" : L"10^{%$p}" # 10⁰ always shows as 1
    if pmax < pmin # no integer decade inside the range
        p = round(Int, log10(sqrt(lo * hi)))
        return [10.0^p], [decade_label(p)]
    end
    step = max(1, ceil(Int, (pmax - pmin + 1) / maxticks))
    first_p = step * cld(pmin, step)
    ps = collect(first_p:step:pmax)
    isempty(ps) && (ps = [pmin])
    return 10.0 .^ ps, [decade_label(p) for p in ps]
end

"""
$(TYPEDSIGNATURES)

Dense log-axis ticks on the 1–2–5 mantissa series inside `[lo, hi]`: whole
decades labelled `10ⁿ`, intermediate ticks `2×10ⁿ` / `5×10ⁿ`. For log axes
spanning a few decades where whole-power ticks alone are too sparse.
"""
function log_ticks_125(lo::Real, hi::Real)
    (lo > 0 && hi > lo && isfinite(hi)) ||
        throw(
            ArgumentError(
                "log_ticks_125 requires finite bounds 0 < lo < hi, got ($lo, $hi)",
            ),
        )
    vals = Float64[]
    labels = LaTeXStrings.LaTeXString[]
    for p in floor(Int, log10(lo)):ceil(Int, log10(hi)), m in (1, 2, 5)
        v = m * 10.0^p
        (lo * (1 - 1e-9) <= v <= hi * (1 + 1e-9)) || continue
        push!(vals, v)
        # 10⁰ never appears as a factor: 1, 2, 5 in the unit decade
        push!(
            labels,
            p == 0 ? latexstring(m) :
            m == 1 ? latexstring("10^{", p, "}") :
            latexstring(m, "\\times 10^{", p, "}"),
        )
    end
    return vals, labels
end

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Common decimal exponent for a linear axis whose data extend to `maxabs`;
0 when plain labels are fine (|values| in [1e-2, 1e4)).
"""
function axis_exponent(maxabs::Real)
    (maxabs <= 0 || !isfinite(maxabs)) && return 0
    e = floor(Int, log10(maxabs))
    return -2 <= e <= 3 ? 0 : e
end

"""
$(TYPEDSIGNATURES)

Tick selection for small-value linear axes: every tick is an integer
mantissa of one common power of 10 (the `exponent`, annotated once at the
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
        for q in (5, 2, 1), k in (kmid-2):(kmid+1)
            step = q * 10.0^k
            lo_s = floor(lo / step + 1e-9) * step
            hi_s = ceil(hi / step - 1e-9) * step
            n = round(Int, (hi_s - lo_s) / step) + 1
            n in nrange || continue
            (hi_s - hi) + (lo - lo_s) <= cap * span || continue
            vals = [lo_s + i * step for i in 0:(n-1)]
            labels = [latexstring(string(round(Int, v / 10.0^k))) for v in vals]
            return (vals, labels, k, lo_s, hi_s)
        end
    end
    return nothing
end

"""
$(TYPEDSIGNATURES)

Fit-coefficient formatting: always mantissa × power-of-10 with two decimal
places (`1.07×10⁻³`); the ×10⁰ factor alone is omitted.
"""
function coef_latex(v::Real)
    v == 0 && return "0"
    isfinite(v) || return string(v)
    e = floor(Int, log10(abs(v)))
    m = @sprintf("%.2f", v / 10.0^e)
    return e == 0 ? m : string(m, "\\times 10^{", e, "}")
end

"""
$(TYPEDSIGNATURES)

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
$(TYPEDSIGNATURES)

Quartic-scaling validation figure: log–log D²(δ) with the δ⁴ prediction
(solid dark red), the `D² = ρ²` threshold line and δ_min marker, a shaded
optimizer-floor band, and a linked ratio panel `D²_num/D²_theo` that makes
prefactor agreement and higher-order departures visible (navy fitted
departure curve, from the first point to the right margin). Both panels use
one vocabulary: blue dots, and floor-band points stricken through by a thin
X (clamped at the ratio-panel top, where their true ratio diverges);
non-convergence carries no special mark. The legend sits on top:
`D²_theoretical`, `D²_numerical`, and the fitted slope as a text-only
entry. `clean` is the Bool mask of points used for the fits — the caller
should pass the above-optimizer-floor mask.
"""
function scaling_figure(deltas::AbstractVector, d2_num::AbstractVector,
    d2_theo::AbstractVector;
    rho_sq::Real, delta_min::Real, slope::Real, slope_err::Real,
    clean::AbstractVector{Bool}, floor_level::Real,
    c1::Real = NaN, c2::Real = NaN)
    with_theme(publication_theme()) do
        fig = Figure(size = (920, 900))

        pos = d2_num .> 0
        ylo = min(minimum(d2_num[pos]), minimum(d2_theo)) / 3
        yhi = max(maximum(d2_num[pos]), maximum(d2_theo)) * 3
        x_ticks = decade_ticks(minimum(deltas), maximum(deltas))
        y_ticks = decade_ticks(ylo, yhi)

        ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
            ylabel = L"D^2", xticks = x_ticks, yticks = y_ticks, yticklabelspace = 66.0)

        theory_line = lines!(ax1, deltas, d2_theo; color = :darkred, linewidth = 4.0)
        numerical_scatter = scatter!(ax1, deltas[pos], d2_num[pos]; color = :dodgerblue,
            strokecolor = :black, strokewidth = 1.4, markersize = 22)
        # points inside the optimizer-floor band are stricken through by a
        # thin X stretching over the symbol (no legend entry): the floor, not
        # the physics, sets them
        floored =
            isfinite(floor_level) ? (pos .& (d2_num .<= floor_level)) :
            falses(length(d2_num))
        any(floored) && scatter!(ax1, deltas[floored], d2_num[floored];
            marker = '×', color = :grey15, markersize = 42)
        ylims!(ax1, ylo, yhi)

        # legend on top of the figure: two entries only — the fitted slope is
        # part of the D²_numerical label, not a separate item
        Legend(fig[0, 1],
            [theory_line, numerical_scatter],
            [L"D^2_{\mathrm{theoretical}}",
                latexstring(
                    @sprintf(
                        "D^2_{\\mathrm{numerical}}\\;\\;\\ \\mathrm{slope:}\\ %.3f \\pm %.3f",
                        slope, slope_err)
                )];
            orientation = :horizontal, framevisible = false,
            tellwidth = false, tellheight = true, labelsize = 19,
            patchsize = (30, 4), colgap = 16, patchlabelgap = 5,
            padding = (0, 0, 4, 0))

        if isfinite(floor_level) && floor_level > ylo
            hspan!(ax1, ylo, floor_level; color = (:grey, 0.30))
            # bottom-right, well clear of the rising δ⁴ line (which is high there)
            text!(ax1, maximum(deltas), floor_level;
                text = "Optimizer floor", align = (:right, :bottom),
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

        ax2 = Axis(fig[2, 1]; xscale = log10,
            xlabel = L"\mathrm{parameter\ separation}\ \delta",
            ylabel = L"D^2_{\mathrm{num}}/D^2_{\mathrm{theo}}", xticks = x_ticks,
            yticklabelspace = 66.0)
        # The ratio panel repeats the top panel's vocabulary exactly: dark-red
        # solid reference at 1 (the theory), blue dots, and floor-band points
        # stricken through by the same thin X — their true ratio diverges, so
        # they sit clamped at the panel top. Non-converged points carry no
        # special mark: the iteration/tolerance caps are strict enough that
        # flagged points at ratio ≈ 1 are genuine optima.
        ratio = d2_num ./ d2_theo
        hlines!(ax2, [1.0]; color = :darkred, linewidth = 3.0)
        # points below the optimizer floor are discarded here — their ratio
        # is floor-set and meaningless; the top panel shows them X-stricken
        keep = pos .& .!floored
        # symmetric, dynamically chosen y-limits around ratio 1: the smallest
        # symmetry-appealing half-width that holds every kept point with
        # ~15% headroom
        spread = any(keep) ? maximum(abs, ratio[keep] .- 1.0) : 0.5
        half_width_index = something(findfirst(w -> w >= 1.15 * spread,
            (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)), 6)
        ratio_half_width = (0.02, 0.05, 0.1, 0.25, 0.5, 1.0)[half_width_index]
        if isfinite(c1) && any(keep)
            # higher-order-terms fit in dashed dark blue (reads cleanly over
            # the solid reference): starting slightly left of the first kept
            # point, reaching past the right margin (clipped by the frame)
            delta_fit_grid =
                10.0 .^ range(log10(minimum(deltas[keep])) - 0.15,
                    log10(maximum(deltas)) + 0.4, length = 200)
            model =
                1.0 .+ c1 .* delta_fit_grid .+
                (isfinite(c2) ? c2 : 0.0) .* delta_fit_grid .^ 2
            lines!(ax2, delta_fit_grid,
                clamp.(model, 1.0 - ratio_half_width, 1.0 + ratio_half_width);
                color = :navy,
                linestyle = :dash, linewidth = 3.0)
            # coefficients always written out as mantissa × 10ⁿ
            ctext = "1 + c_1\\delta + c_2\\delta^2,\\;\\; c_1 = " * coef_latex(c1)
            isfinite(c2) && (ctext *= ",\\;\\ c_2 = " * coef_latex(c2))
            text!(ax2, 0.985, 0.92; space = :relative,
                text = latexstring(ctext),
                align = (:right, :top), fontsize = 16, color = :navy)
        end
        scatter!(ax2, deltas[keep], ratio[keep]; color = :dodgerblue,
            strokecolor = :black, strokewidth = 1.2, markersize = 17)
        ylims!(ax2, 1.0 - ratio_half_width, 1.0 + ratio_half_width)

        # explicit x-limits from the data with a small log margin: the fit
        # curve and reference line intentionally overshoot the last point, and
        # autolimits would otherwise stretch the frame after them
        logspan = log10(maximum(deltas)) - log10(minimum(deltas))
        xpad = 10.0^(0.04 * logspan)
        xlims!(ax2, minimum(deltas) / xpad, maximum(deltas) * xpad)

        linkxaxes!(ax1, ax2)
        hidexdecorations!(ax1; grid = false, ticks = false)
        rowsize!(fig.layout, 1, Relative(0.68))
        return fig
    end
end

"""
$(TYPEDSIGNATURES)

Residual-spectrum figure in true density units: top panel `d(SNR²)/df` of the
two-source data and the best-fit single source, bottom panel the unabsorbed
residual `d(D²)/df` — the integral of the bottom curves is the D² of the
scaling law. One grouped legend sits on top of the figure (channel A and
channel E blocks, each with data / best fit / residual). Channel encodes hue
(A blue, E warm); the best fit, which lies on top of the data, is a brighter
dash-dotted line over the dark solid data line; min/max decimation envelopes
shade both channels and are covered by the bottom panel's y-range (depth-
capped so a cancellation spike cannot compress the curves). Frame limits
follow the plotted data with dense 1–2–5 log ticks, and the δ*/integral and
off-scale-noise annotations sit between the panels, outside the frames.
`spec` is the (log-uniformly decimated) spectrum table; `meta` must carry
the fields `delta_star`, `df`, `d2_num` and `d2_theo` (evaluation
separation, frequency resolution, and the two integral annotations).
"""
function residual_figure(spec, meta)
    with_theme(publication_theme()) do
        fig = Figure(size = (950, 950))
        # Limits and ticks follow the plotted data: with log-uniform decimation
        # the first/last plotted frequencies sit at the band ends, so the
        # frame ends on the data with no gap at either side. Dense 1–2–5
        # log ticks — whole decades alone are too sparse over ~3 decades.
        fmin = minimum(spec.f)
        fmax = maximum(spec.f)
        x_ticks = log_ticks_125(fmin, fmax)

        # channel = hue (A blue, E warm), role = shade + line style: data is
        # the dark solid line, the best fit — which sits right on top of it —
        # is a brighter dash-dotted line over it, the residual a medium solid
        col_data_A, col_bf_A, col_res_A = :steelblue4, :deepskyblue, :dodgerblue2
        col_data_E, col_bf_E, col_res_E = :sienna4, :orange, :darkorange3

        ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
            ylabel = L"\mathrm{d}\rho^2/\mathrm{d}f\ \ [\mathrm{Hz}^{-1}]",
            xticks = x_ticks, yticklabelspace = 70.0)
        band!(ax1, spec.f, spec.sig_min_A, spec.sig_max_A; color = (col_data_A, 0.14))
        band!(ax1, spec.f, spec.sig_min_E, spec.sig_max_E; color = (col_data_E, 0.14))
        dA = lines!(ax1, spec.f, spec.sig_rms_A; color = col_data_A, linewidth = 3.6)
        dE = lines!(ax1, spec.f, spec.sig_rms_E; color = col_data_E, linewidth = 3.6)
        bA = lines!(ax1, spec.f, spec.bf_rms_A; color = col_bf_A,
            linestyle = :dashdot, linewidth = 3.0)
        bE = lines!(ax1, spec.f, spec.bf_rms_E; color = col_bf_E,
            linestyle = :dashdot, linewidth = 3.0)

        # y-range of the residual panel: cover the lines AND the min/max
        # shadings — but cap the extra depth at ~1.6 decades below the rms
        # floor, so a near-cancellation spike in a single decimation window
        # cannot compress the curves into a negligible band (the band then clips only
        # inside the dip). The per-bin noise reference 1/Δf can sit many
        # decades above the curves and is never allowed to distort the range.
        res_pos = filter(>(0), vcat(spec.res_rms_A, spec.res_rms_E))
        band_pos = filter(>(0), vcat(spec.res_min_A, spec.res_min_E))
        band_lo = isempty(band_pos) ? minimum(res_pos) : minimum(band_pos)
        ylo2 = max(band_lo / 2, minimum(res_pos) / 40)
        yhi2 = max(maximum(spec.res_max_A), maximum(spec.res_max_E),
            maximum(res_pos)) * 4
        noise_level = 1.0 / meta.df
        noise_in_frame = noise_level < 30 * yhi2
        noise_in_frame && (yhi2 = max(yhi2, 3 * noise_level))

        # annotations sit between the panels — above the bottom plot, outside
        # its frame, on one line: the δ*/integral text left-aligned, and the
        # off-scale noise note right-aligned in the same row (only when the
        # 1/Δf reference cannot be drawn inside the frame)
        Label(fig[2, 1],
            latexstring("\\delta^* = ", sci_latex(meta.delta_star),
                ";\\;\\; \\int\\!\\mathrm{d}f = D^2 = ", sci_latex(meta.d2_num),
                "\\;\\; (\\mathrm{th.}\\ ", sci_latex(meta.d2_theo), ")");
            fontsize = 16, halign = :left, tellwidth = false, tellheight = true,
            padding = (4, 0, 2, 8))
        if !noise_in_frame
            Label(fig[2, 1],
                latexstring("\\mathrm{Noise\\ level\\ per\\ bin:}\\ 1/\\Delta f = ",
                    sci_latex(noise_level),
                    "\\ \\mathrm{Hz^{-1}}\\ \\mathrm{(off\\ scale)}");
                fontsize = 14, color = :grey35, halign = :right,
                tellwidth = false, tellheight = true, padding = (0, 4, 2, 8))
        end

        ax2 = Axis(fig[3, 1]; xscale = log10, yscale = log10,
            xlabel = L"f\ \ [\mathrm{Hz}]",
            ylabel = L"\mathrm{d}(D^2)/\mathrm{d}f\ \ [\mathrm{Hz}^{-1}]",
            xticks = x_ticks, yticks = decade_ticks(ylo2, yhi2),
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
            ["channel A:", "channel E:"];
            orientation = :horizontal, titleposition = :left,
            framevisible = false, tellwidth = false, tellheight = true,
            labelsize = 18, titlesize = 19, titlefont = :bold,
            patchsize = (26, 4), groupgap = 48, patchlabelgap = 4,
            colgap = 10, titlegap = 8, padding = (0, 0, 4, 0))
        if noise_in_frame
            hlines!(ax2, [noise_level]; color = :grey35, linewidth = 1.8, linestyle = :dot)
            text!(ax2, fmax, noise_level; text = "per-bin noise level",
                align = (:right, :top), offset = (-6, -4), fontsize = 16, color = :grey35)
        end
        ylims!(ax2, ylo2, yhi2)

        linkxaxes!(ax1, ax2)
        xlims!(ax2, fmin, fmax) # frame ends on the data — no gap at either side
        hidexdecorations!(ax1; grid = false, ticks = false)
        rowgap!(fig.layout, 8) # keep the between-panel annotation block compact
        return fig
    end
end

"""
$(TYPEDSIGNATURES)

Coordinates of the closed-boundary edges whose class (`edge_prior[i]`) equals
`want`, grouped into contiguous runs separated by `NaN` so each run renders as
one polyline (letting a dash pattern form over the whole run).
"""
function _boundary_runs(x, y, edge_prior::AbstractVector{Bool}, want::Bool)
    n = length(x)
    xs = Float64[]
    ys = Float64[]
    inrun = false
    for i in 1:n
        j = mod1(i + 1, n)
        if edge_prior[i] == want
            if !inrun
                isempty(xs) || (push!(xs, NaN); push!(ys, NaN))
                push!(xs, x[i])
                push!(ys, y[i])
                inrun = true
            end
            push!(xs, x[j])
            push!(ys, y[j])
        else
            inrun = false
        end
    end
    return xs, ys
end

"""
$(TYPEDSIGNATURES)

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
function zone_figure(x::AbstractVector, y::AbstractVector,
    prior_limited::AbstractVector{Bool};
    px::Int, py::Int, box::NTuple{4,Float64},
    prior_frac::Real, degenerate_frac::Real,
    x_math = nothing, y_math = nothing)
    px == py && error("map plane must use two distinct parameters")
    same_units = (px in (5, 6) && py in (5, 6))
    with_theme(publication_theme()) do
        # squarer canvas for same-unit (DataAspect) planes to avoid wide side margins
        fig = Figure(size = same_units ? (820, 830) : (960, 720))

        # Limits fit the zone, not a symmetric ±max box: prior-capped zones are
        # strongly asymmetric (e.g. spins live in the lower-left wedge) and
        # symmetric limits waste most of the canvas on empty quadrants.
        xlo_d, xhi_d = extrema(x)
        ylo_d, yhi_d = extrema(y)
        # extend the view towards the uncapped mathematical contour with a
        # soft clamp: when the full contour lies only modestly beyond the
        # zone (≤ 80% of the zone span per side) include it entirely —
        # showing the complete contour justifies a slightly larger frame; only
        # beyond that (unbounded degenerate directions) clamp at 40% of the
        # span and let the dashed curve run off the frame
        if x_math !== nothing
            xr = xhi_d - xlo_d
            yr = yhi_d - ylo_d
            fx = filter(isfinite, x_math)
            fy = filter(isfinite, y_math)
            if !isempty(fx) && !isempty(fy)
                soft_extension(need, span) = need <= 0.8 * span ? need : 0.4 * span
                xlo_d -= soft_extension(max(0.0, xlo_d - minimum(fx)), xr)
                xhi_d += soft_extension(max(0.0, maximum(fx) - xhi_d), xr)
                ylo_d -= soft_extension(max(0.0, ylo_d - minimum(fy)), yr)
                yhi_d += soft_extension(max(0.0, maximum(fy) - yhi_d), yr)
            end
        end
        x_exponent = axis_exponent(max(abs(xlo_d), abs(xhi_d)))
        y_exponent = axis_exponent(max(abs(ylo_d), abs(yhi_d)))

        ax = Axis(fig[1, 1]; xlabel = deviation_label(px), ylabel = deviation_label(py))
        same_units && (ax.aspect = DataAspect())

        # Per-axis ticks and final limits. Phase axes: rational-π ticks over a
        # padded range. Small-value axes: integer-mantissa ticks of one common
        # power of 10 — the power annotated once at the end of the axis, never
        # per tick and never inside the axis label — with mantissa steps
        # preferring multiples of 5 and the limits snapped outward so the
        # frame ends exactly on labelled ticks. Fallback when no clean grid
        # exists: per-tick common-exponent scientific notation.
        xlo, xhi = xlo_d - 0.08 * (xhi_d - xlo_d), xhi_d + 0.08 * (xhi_d - xlo_d)
        ylo, yhi = ylo_d - 0.08 * (yhi_d - ylo_d), yhi_d + 0.08 * (yhi_d - ylo_d)
        if px == 4
            pi_tick_values = pi_ticks(xlo, xhi)
            pi_tick_values !== nothing && (ax.xticks = pi_tick_values)
        elseif x_exponent != 0
            offset_result = offset_ticks(xlo_d, xhi_d)
            if offset_result === nothing
                ax.xticks = LinearTicks(6)
                ax.xtickformat = sci_tick_labels
            else
                vals, labels, axis_power, lo_s, hi_s = offset_result
                ax.xticks = (vals, labels)
                xlo, xhi = lo_s, hi_s
                # common power of 10 at the end of the x axis: just right of
                # the frame's bottom corner, clear of the last tick label
                axis_power != 0 &&
                    Label(fig[1, 2], latexstring("\\times 10^{", axis_power, "}");
                        fontsize = 20, halign = :left, valign = :bottom,
                        padding = (2, 0, 0, 0), tellheight = false)
            end
        else
            ax.xticks = LinearTicks(6)
        end
        if py == 4
            pi_tick_values = pi_ticks(ylo, yhi)
            pi_tick_values !== nothing && (ax.yticks = pi_tick_values)
        elseif y_exponent != 0
            offset_result = offset_ticks(ylo_d, yhi_d)
            if offset_result === nothing
                ax.yticks = LinearTicks(6)
                ax.ytickformat = sci_tick_labels
            else
                vals, labels, axis_power, lo_s, hi_s = offset_result
                ax.yticks = (vals, labels)
                ylo, yhi = lo_s, hi_s
                # common power of 10 at the end of the y axis: above the frame
                axis_power != 0 &&
                    Label(fig[0, 1], latexstring("\\times 10^{", axis_power, "}");
                        fontsize = 20, halign = :left, valign = :bottom,
                        padding = (0, 0, 2, 0), tellwidth = false)
            end
        else
            ax.yticks = LinearTicks(6)
        end

        poly!(ax, Point2f.(x, y); color = (:dodgerblue, 0.30), strokewidth = 0)

        # The boundary of the filled (physical) zone is solid throughout:
        # curvature-limited runs in blue, prior-limited runs in red along the
        # hard physical walls. Where the prior cuts the zone off, the uncapped
        # mathematical contour continues past the wall as an empty dashed line
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
            # name the active walls with their values: a prior-limited boundary
            # point sits exactly on the box edge its ray exited through, so an
            # edge is active iff some capped point lies on it
            devtex(idx) = string("\\Delta ", PARAM_LABELS[idx][2:(end-1)])
            fmt_edge(v, isphase) =
                isphase && isapprox(abs(v), π; atol = 1e-9) ?
                (v < 0 ? "-\\pi" : "\\pi") : sci_latex(v)
            prior_x = view(x, prior_limited)
            prior_y = view(y, prior_limited)
            lox, hix, loy, hiy = box
            walls = String[]
            for (lo_e, hi_e, vals, tol, idx) in
                ((lox, hix, prior_x, 1e-6 * (xhi - xlo), px),
                (loy, hiy, prior_y, 1e-6 * (yhi - ylo), py))
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
                (
                    msg *= string(
                        "\\;\\ (",
                        pcttex(degenerate_frac),
                        "\\ \\mathrm{degenerate})",
                    )
                )
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
