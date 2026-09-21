"""
Publication CairoMakie figures under one theme and a family-wide tick
policy (log 1-2-5 series, axis offset multipliers, rational-pi ticks).
"""
module Plotting

using DocStringExtensions: TYPEDSIGNATURES
using CairoMakie: Axis, DataAspect, Figure, GridLayout, Label, Legend,
    LinearTicks, Point2f, PolyElement, Relative, Theme, band!,
    hidexdecorations!, hlines!, hspan!, lines!, linkxaxes!,
    poly!, rowgap!, rowsize!, save, scatter!, text!, vlines!,
    with_theme, xlims!, ylims!
using LaTeXStrings: LaTeXStrings, @L_str, latexstring
using DataFrames: AbstractDataFrame
using ..Bounds: PHASE_INDEX, SPIN_INDICES
using MathTeXEngine: texfont
using Printf: Printf, @sprintf
using UnicodePlots: UnicodePlots
using ..Provenance: backup_existing!

export publication_theme,
    save_figure, scaling_figure, residual_figure,
    ResidualFigureMeta, zone_figure
public decade_ticks, pi_ticks, log_ticks_125, offset_ticks, sci_tick_labels,
    sci_latex, coef_latex, slope_latex, axis_exponent, sweep_diagnostic_panel,
    map_diagnostic_panel

"""
    ResidualFigureMeta

Annotation record of a [`residual_figure`](@ref): the evaluation
separation `delta_star`, the frequency resolution `df` [Hz] (setting the
per-bin noise reference `1/Δf`), and the numerical and theoretical `D²`
at that separation.
"""
struct ResidualFigureMeta
    delta_star::Float64
    df::Float64
    d2_num::Float64
    d2_theo::Float64
end

"""
Short LaTeX axis labels for the six model parameters (deviation form is
composed by the figure builders).
"""
const PARAM_LABELS =
    (L"D_L", L"\mathcal{M}", L"t_c", L"\Phi_0", L"\chi_1", L"\chi_2")

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
function save_figure(fig::Figure, base_path::AbstractString)
    pdf = backup_existing!(base_path * ".pdf")
    save(pdf, fig)
    png = backup_existing!(base_path * ".png")
    save(png, fig; px_per_unit = 4)
    return (pdf, png)
end

"""
$(TYPEDSIGNATURES)

In-terminal diagnostic of a completed sweep: log-log `D²` against the
theoretical prediction (UnicodePlots), followed by the clean-point count and
fitted slope. Opt-in via `[monitoring].enabled`; printed to stdout on TTY
sessions only, never into the file logs.
"""
function sweep_diagnostic_panel(deltas::AbstractVector, D2_num::AbstractVector,
    D2_theo::AbstractVector, clean::AbstractVector{Bool},
    slope::Real, slope_err::Real)
    pos = D2_num .> 0
    any(pos) || return "sweep diagnostic: no positive D² values"
    plt = UnicodePlots.lineplot(log10.(collect(deltas)), log10.(collect(D2_theo));
        name = "theory", xlabel = "log₁₀ δ",
        ylabel = "log₁₀ D²", width = 64, height = 14)
    UnicodePlots.scatterplot!(plt, log10.(collect(deltas[pos])), log10.(D2_num[pos]);
        name = "numerical")
    footer = @sprintf("clean points %d/%d; fitted slope %.4f ± %.4f",
        count(clean), length(clean), slope, slope_err)
    return sprint(io -> show(io, plt)) * "\n" * footer
end

"""
$(TYPEDSIGNATURES)

In-terminal diagnostic of a completed confusion map: capped boundary radius
against direction angle (UnicodePlots), followed by the prior-limited
fraction. Opt-in via `[monitoring].enabled`; stdout on TTY sessions only.
"""
function map_diagnostic_panel(angle::AbstractVector, r_cap::AbstractVector,
    prior_frac::Real)
    plt = UnicodePlots.lineplot(collect(angle), collect(r_cap);
        xlabel = "φ [rad]", ylabel = "capped radius",
        width = 64, height = 14)
    footer = @sprintf("prior-limited directions: %.1f%%", 100 * prior_frac)
    return sprint(io -> show(io, plt)) * "\n" * footer
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
    # 10⁰ always shows as 1 and 10¹ as 10 (never a redundant power form)
    decade_label(p) = p == 0 ? L"1" : p == 1 ? L"10" : L"10^{%$p}"
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
        # 10⁰ and 10¹ never appear as factors: 1, 2, 5 in the unit decade
        # and 10, 20, 50 in the tens decade
        push!(
            labels,
            p == 0 ? latexstring(m) :
            p == 1 ? latexstring(10m) :
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

LaTeX fragment for a scalar with `sig` significant digits: plain decimal
only while the exponent is in −1..1 (the range where the power-of-ten form
would collapse anyway: 5×10⁰ → 5, 2×10¹ → 20, 2×10⁻¹ → 0.2), `m×10^e`
everywhere else — for annotations, never bare `1e-05` e-notation.
"""
function sci_latex(v::Real; sig::Int = 3)
    v == 0 && return "0"
    isfinite(v) || return string(v)
    e = floor(Int, log10(abs(v)))
    -1 <= e <= 1 && return @sprintf("%.4g", round(v, sigdigits = sig))
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

Tick values symmetric about zero inside `[-h, h]`, taken from the
1–2–2.5–5 × 10ᵏ step series with the step chosen so that 5 or 7 labelled
ticks fall strictly inside the limits — zero always among them, and the frame
free to end between ticks. Returns `(values, k)`, where `k` is the power of
ten of the step, so the mantissas `values / 10^k` are integers (half-integers
for the 2.5 step); `nothing` when no usable step exists.
"""
function symmetric_ticks(h::Real)
    (isfinite(h) && h > 0) || return nothing
    kmid = floor(Int, log10(h))
    best = nothing
    # denser grids first, then the plainer step mantissas
    for (rank, q) in enumerate((1.0, 2.0, 5.0, 2.5)), k in (kmid-2):(kmid+1)
        step = q * 10.0^k
        n = floor(Int, h / step + 1e-9)
        n in 2:3 || continue
        score = (n, -rank)
        (best === nothing || score > best[1]) && (best = (score, step, n, k))
    end
    best === nothing && return nothing
    _, step, n, k = best
    return ([i * step for i in (-n):n], k)
end

"""
$(TYPEDSIGNATURES)

Fitted-slope annotation `value ± error` with the decimal places set by the
error — one significant digit of the uncertainty, never fewer than three
decimals nor more than six (`3.9998 ± 0.0001`, `3.824 ± 0.043`); the bare
three-decimal value when the error is not finite.
"""
function slope_latex(slope::Real, slope_err::Real)
    isfinite(slope) || return string(slope)
    decimals = 3
    if isfinite(slope_err) && slope_err > 0
        decimals = clamp(-floor(Int, log10(slope_err)), 3, 6)
    end
    value = Printf.format(Printf.Format("%.$(decimals)f"), slope)
    (isfinite(slope_err) && slope_err > 0) || return value
    return value * " \\pm " * Printf.format(Printf.Format("%.$(decimals)f"), slope_err)
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
    # exponents −1..1 render as plain decimals keeping the two mantissa decimals
    -1 <= e <= 1 && return Printf.format(Printf.Format("%.$(2 - e)f"), v)
    return string(@sprintf("%.2f", v / 10.0^e), "\\times 10^{", e, "}")
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
        # a common power in −1..1 is never displayed: plain decimals instead
        -1 <= e <= 1 && return latexstring(@sprintf("%g", round(v, sigdigits = 3)))
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
`D²_theoretical`, `D²_numerical` with the fitted slope folded into its
label, and — only when the band is drawn — a patch in the band's own grey
naming the optimizer floor. `clean` is the Bool mask of points used for the
fits — the caller should pass the above-optimizer-floor mask.
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
        # the threshold can sit close to the frame top (near-degenerate
        # sweeps end just past D² = ρ²): guarantee headroom for its label
        # before the ticks are chosen, so every decade of the frame is labelled
        threshold_visible = ylo < rho_sq < yhi
        threshold_visible && (yhi = max(yhi, rho_sq * 30))
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

        # legend on top of the figure: the fitted slope is part of the
        # D²_numerical label, not a separate item, and the sub-floor band is
        # named by a patch in its own colour rather than by an in-axis text
        # (which the δ_min line crosses whenever δ_min sits in the last decade)
        floor_color = (:grey, 0.30)
        floor_band = isfinite(floor_level) && floor_level > ylo
        legend_entries = Any[theory_line, numerical_scatter]
        legend_labels = AbstractString[L"D^2_{\mathrm{theoretical}}",
            latexstring(
                "D^2_{\\mathrm{numerical}}\\;\\;\\ \\mathrm{slope:}\\ " *
                slope_latex(slope, slope_err),
            )]
        if floor_band
            push!(legend_entries, PolyElement(color = floor_color))
            push!(legend_labels, "Optimizer floor") # upright text: no math in it
        end
        Legend(fig[0, 1], legend_entries, legend_labels;
            orientation = :horizontal, framevisible = false,
            tellwidth = false, tellheight = true, labelsize = 19,
            # patch box tall enough for the band swatch to read; line and
            # marker entries are drawn centred in it, so they are unaffected
            patchsize = (30, 12), colgap = 16, patchlabelgap = 5,
            padding = (0, 0, 4, 0))

        floor_band && hspan!(ax1, ylo, floor_level; color = floor_color)
        if threshold_visible
            hlines!(ax1, [rho_sq]; color = :grey35, linewidth = 1.8)
            # left side: the δ⁴ line is many decades below the threshold
            # there, so the label cannot collide with data or fit
            text!(ax1, minimum(deltas), rho_sq; text = L"\rho^2_{\mathrm{thr}}",
                align = (:left, :bottom), offset = (2, 3),
                fontsize = 18, color = :grey35)
        end
        if minimum(deltas) < delta_min < maximum(deltas)
            vlines!(ax1, [delta_min]; color = :grey35, linewidth = 1.8, linestyle = :dash)
            text!(ax1, delta_min, ylo; text = L"\delta_{\mathrm{min}}",
                align = (:right, :bottom), offset = (-5, 4),
                fontsize = 18, color = :grey35)
        end

        ax2 = Axis(fig[2, 1]; xscale = log10,
            xlabel = L"\delta",
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
        # clearance between the linked panels so the top panel's lowest and
        # the ratio panel's highest y-tick labels can never meet: each label
        # extends ~half its height past its frame edge, so the junction gap
        # must exceed one full label height with margin (the legend gap above
        # the top panel stays compact). Gap indices follow the layout's row
        # numbering (legend row 0, panels 1 and 2): gap 1 lies below row 1.
        rowgap!(fig.layout, 12)
        rowgap!(fig.layout, 1, 36)
        return fig
    end
end

"""
Number of passes of the five-point binomial kernel applied to the window means
of the residual-spectrum figure before plotting. Oscillations of the density
whose period is within a few decimation windows would otherwise alias into a
sawtooth; the min–max envelope is drawn unsmoothed and keeps the full range.
Display only — persisted tables are never smoothed.
"""
const RESIDUAL_SMOOTHING_PASSES = 2

"""
$(TYPEDSIGNATURES)

`passes` applications of the symmetric five-point binomial kernel
`[1, 4, 6, 4, 1]/16` to `v`, returning a new `Vector{Float64}` (`passes = 0`
copies). The kernel acts on the values themselves, not their logarithms, so
the area under the curve is preserved. At the ends the truncated kernel is
renormalised by the weights actually used — no padding, no reflection —
and non-finite entries are skipped in the local average, a non-finite entry
staying non-finite.
"""
function binomial_smooth(v::AbstractVector{<:Real}, passes::Integer)
    passes >= 0 ||
        throw(ArgumentError("binomial_smooth needs passes >= 0, got $passes"))
    out = Float64.(v)
    passes == 0 && return out
    w = (1.0, 4.0, 6.0, 4.0, 1.0)
    n = length(out)
    buf = similar(out)
    for _ in 1:passes
        for i in 1:n
            if !isfinite(out[i])
                buf[i] = out[i]
                continue
            end
            acc = 0.0
            wsum = 0.0
            for (k, wk) in enumerate(w)
                j = i + k - 3
                (1 <= j <= n && isfinite(out[j])) || continue
                acc += wk * out[j]
                wsum += wk
            end
            buf[i] = wsum > 0 ? acc / wsum : out[i]
        end
        copyto!(out, buf)
    end
    return out
end

"""
$(TYPEDSIGNATURES)

Residual-spectrum figure in true density units: top panel `d(SNR²)/df` of the
two-source data and the best-fit single source, bottom panel the unabsorbed
residual `d(D²)/df` — the integral of the bottom curves is the D² of the
scaling law. The legend sits on top of the figure as two stacked rows, one per
channel (data / best fit / residual), left-aligned so the entries line up in
columns. Channel encodes hue
(A blue, E warm); the best fit, which lies on top of the data, is a brighter
dash-dotted line over the dark solid data line; the plotted means are
binomially smoothed for display (`RESIDUAL_SMOOTHING_PASSES`) while
the min/max decimation envelopes are drawn from the raw columns, shading both
channels and covered by the bottom panel's y-range (depth-
capped so a cancellation spike cannot compress the curves). Frame limits
follow the plotted data with dense 1–2–5 log ticks, and the δ*/integral and
off-scale-noise annotations sit between the panels, outside the frames.
`spec` is the (log-uniformly decimated) spectrum table; `meta` is the
[`ResidualFigureMeta`](@ref) annotation record. `delta_symbol` names the
evaluation separation in the annotation — the default `\\delta^*` for the
validity-window panel, `\\delta_{\\mathrm{thr}}` for the threshold
companion.
"""
function residual_figure(spec::AbstractDataFrame, meta::ResidualFigureMeta;
    delta_symbol::String = "\\delta^*")
    with_theme(publication_theme()) do
        # height carries the second legend row (~32) on top of the 950 the two
        # panels and the annotation row need, so the axes keep their size
        fig = Figure(size = (950, 982))
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

        # display-only smoothing of the window means: oscillations of the
        # density whose period spans a few decimation windows alias into a
        # sawtooth in the means (the envelopes below stay raw)
        smooth(v) = binomial_smooth(v, RESIDUAL_SMOOTHING_PASSES)
        sig_mean_A = smooth(spec.sig_mean_A)
        sig_mean_E = smooth(spec.sig_mean_E)
        bf_mean_A = smooth(spec.bf_mean_A)
        bf_mean_E = smooth(spec.bf_mean_E)
        res_mean_A = smooth(spec.res_mean_A)
        res_mean_E = smooth(spec.res_mean_E)

        # explicit decade ticks (Makie's default log labels would render the
        # unit and tens decades as 10⁰/10¹ instead of 1/10)
        sig_pos = filter(
            >(0),
            vcat(spec.sig_min_A, spec.sig_min_E,
                spec.sig_max_A, spec.sig_max_E),
        )
        ax1 = Axis(fig[1, 1]; xscale = log10, yscale = log10,
            ylabel = L"\mathrm{d}\rho^2/\mathrm{d}f\ \ [\mathrm{Hz}^{-1}]",
            xticks = x_ticks,
            yticks = decade_ticks(minimum(sig_pos), maximum(sig_pos)),
            yticklabelspace = 70.0)
        band!(ax1, spec.f, spec.sig_min_A, spec.sig_max_A; color = (col_data_A, 0.14))
        band!(ax1, spec.f, spec.sig_min_E, spec.sig_max_E; color = (col_data_E, 0.14))
        dA = lines!(ax1, spec.f, sig_mean_A; color = col_data_A, linewidth = 3.6)
        dE = lines!(ax1, spec.f, sig_mean_E; color = col_data_E, linewidth = 3.6)
        bA = lines!(ax1, spec.f, bf_mean_A; color = col_bf_A,
            linestyle = :dashdot, linewidth = 3.0)
        bE = lines!(ax1, spec.f, bf_mean_E; color = col_bf_E,
            linestyle = :dashdot, linewidth = 3.0)

        # y-range of the residual panel: cover the lines AND the min/max
        # shadings — but cap the extra depth at ~1.6 decades below the mean
        # floor, so a near-cancellation spike in a single decimation window
        # cannot compress the curves into a negligible band (the band then clips only
        # inside the dip). The per-bin noise reference 1/Δf can sit many
        # decades above the curves and is never allowed to distort the range.
        res_pos = filter(>(0), vcat(res_mean_A, res_mean_E))
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
            latexstring(delta_symbol, " = ", sci_latex(meta.delta_star),
                ":\\;\\; D^2_{", delta_symbol, "} = ", sci_latex(meta.d2_num),
                "\\;\\; (D^2_{\\mathrm{th},\\,", delta_symbol, "} = ",
                sci_latex(meta.d2_theo), ")");
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
        rA = lines!(ax2, spec.f, res_mean_A; color = col_res_A, linewidth = 3.2)
        rE = lines!(ax2, spec.f, res_mean_E; color = col_res_E, linewidth = 3.2)

        # Legend on top of the figure, spanning both panels. Mechanism: two
        # single-group `Legend`s stacked in a nested `GridLayout` — one row per
        # channel, each left-aligned in the block so the three entries line up
        # in columns. A single grouped legend laid the two blocks side by side
        # and ran wider than the canvas; `nbanks` cannot reproduce the
        # per-channel rows with their own bold header.
        legend_kw = (; orientation = :horizontal, titleposition = :left,
            framevisible = false, tellwidth = true, tellheight = true,
            halign = :left, labelsize = 18, titlesize = 19, titlefont = :bold,
            patchsize = (26, 4), patchlabelgap = 4,
            colgap = 10, titlegap = 8, padding = (0, 0, 1, 1))
        legend_block = GridLayout(fig[0, 1];
            halign = :center, tellwidth = false, tellheight = true)
        Legend(legend_block[1, 1], [dA, bA, rA],
            ["Data", "Best fit", "Residual"], "Channel A:"; legend_kw...)
        Legend(legend_block[2, 1], [dE, bE, rE],
            ["Data", "Best fit", "Residual"], "Channel E:"; legend_kw...)
        rowgap!(legend_block, 3) # the block costs as little height as possible
        if noise_in_frame
            hlines!(ax2, [noise_level]; color = :grey35, linewidth = 1.8, linestyle = :dot)
            text!(ax2, fmax, noise_level; text = "Per-bin noise level",
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
function boundary_runs(x, y, edge_prior::AbstractVector{Bool}, want::Bool)
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

Per-axis tick policy of the zone map for `axis` (`:x` or `:y`) of `ax`,
showing the deviation of parameter `param` over the symmetric range
`[-h, h]` whose common power of ten is `exponent`. Phase axes: rational-π
ticks. Every other axis: 5 or 7 ticks symmetric about zero from the
1–2–2.5–5 step series, labelled by their mantissa with respect to one
common power of 10 — annotated once at the end of the axis (right of the
frame's bottom corner for x, above the frame for y), never per tick and
never inside the axis label — while plain decimals are used where they read
fine (`exponent == 0`). Fallback when no clean grid exists: per-tick
common-exponent scientific notation. Returns the `(lo, hi)` limits.
"""
function zone_axis_ticks!(fig, ax, axis::Symbol, param::Int, h::Real, exponent::Int)
    ticks_property = axis === :x ? :xticks : :yticks
    format_property = axis === :x ? :xtickformat : :ytickformat
    lo, hi = -h, h
    if param == PHASE_INDEX
        pi_tick_values = pi_ticks(lo, hi)
        pi_tick_values !== nothing && setproperty!(ax, ticks_property, pi_tick_values)
        return lo, hi
    end
    tick_result = symmetric_ticks(h)
    if tick_result === nothing
        setproperty!(ax, ticks_property, LinearTicks(6))
        exponent != 0 && setproperty!(ax, format_property, sci_tick_labels)
        return lo, hi
    end
    vals, step_power = tick_result
    # the multiplier is factored out only where plain labels would be poor
    axis_power = exponent == 0 ? 0 : step_power
    labels = [
        latexstring(@sprintf("%g", round(v / 10.0^axis_power, sigdigits = 3)))
        for v in vals
    ]
    setproperty!(ax, ticks_property, (vals, labels))
    if axis_power != 0
        power_label = latexstring("\\times 10^{", axis_power, "}")
        if axis === :x
            # just right of the frame's bottom corner, clear of the last tick label
            Label(fig[1, 2], power_label;
                fontsize = 20, halign = :left, valign = :bottom,
                padding = (2, 0, 0, 0), tellheight = false)
        else
            Label(fig[0, 1], power_label;
                fontsize = 20, halign = :left, valign = :bottom,
                padding = (0, 0, 2, 0), tellwidth = false)
        end
    end
    return lo, hi
end

# Okabe–Ito hues of the zone maps: curvature-limited boundary and fill, and
# the prior-wall segments (colorblind-safe, distinct in grayscale by weight)
const ZONE_BLUE = "#0072B2"
const WALL_VERMILLION = "#D55E00"

"""
Fractional margin of the zone-map frame: each axis is limited symmetrically
about zero at `(1 + ZONE_LIMIT_MARGIN)` times the largest coordinate that must
be visible on it. The uncapped mathematical contour is not included in that
maximum — along degenerate directions it is unbounded — and may leave the
frame.
"""
const ZONE_LIMIT_MARGIN = 0.08

# A same-unit zone whose principal-axis aspect ratio reaches this value is a
# needle along a null direction and is drawn in its principal frame (see
# `zone_figure`). Slated for removal with the 2PN spin–spin term, which closes
# the spin zone (roadmap §1).
const NEEDLE_ASPECT = 20.0

"""
$(TYPEDSIGNATURES)

Principal axis of a boundary polygon: the angle `ϑ ∈ (-π/2, π/2]` of the
major axis of the polygon's second area moments about its centroid (shoelace
form; vertex moments when the polygon is degenerate) and the aspect ratio
`extent_∥ / extent_⊥` of the vertices in that frame (`Inf` for a line). Area
moments are insensitive to how the edges are sampled, but the axis of a band
cut obliquely by prior walls is off by `O((w/L)²)` — invisible in parameter
units, visible once the transverse axis is stretched — so when the aspect
reaches `NEEDLE_ASPECT` the angle is snapped to the longest polygon edge,
which runs exactly along the null direction of the band.
"""
function principal_axis(x::AbstractVector, y::AbstractVector)
    n = length(x)
    n == length(y) || throw(DimensionMismatch("x and y must have equal length"))
    n >= 3 || throw(ArgumentError("a boundary polygon needs at least three vertices"))
    A = 0.0
    Cx = 0.0
    Cy = 0.0
    Ixx = 0.0
    Iyy = 0.0
    Ixy = 0.0
    for i in 1:n
        j = mod1(i + 1, n)
        a = x[i] * y[j] - x[j] * y[i]
        A += a
        Cx += a * (x[i] + x[j])
        Cy += a * (y[i] + y[j])
        Ixx += a * (x[i]^2 + x[i] * x[j] + x[j]^2)
        Iyy += a * (y[i]^2 + y[i] * y[j] + y[j]^2)
        Ixy += a * (x[i] * y[j] + 2x[i] * y[i] + 2x[j] * y[j] + x[j] * y[i])
    end
    if A != 0
        # normalize by the signed area: the covariances of the region are then
        # independent of the polygon's orientation (a clockwise polygon flips
        # the sign of every moment, which would rotate the atan2 axis by 90°)
        A /= 2
        Cx /= 6A
        Cy /= 6A
        Ixx = Ixx / (12A) - Cx^2
        Iyy = Iyy / (12A) - Cy^2
        Ixy = Ixy / (24A) - Cx * Cy
    else
        Ixx = sum(abs2, x)
        Iyy = sum(abs2, y)
        Ixy = sum(x .* y)
    end
    ϑ = 0.5 * atan(2Ixy, Ixx - Iyy)
    aspect = frame_aspect(x, y, ϑ)
    if aspect >= NEEDLE_ASPECT
        _, k = findmax(i -> hypot(x[mod1(i + 1, n)] - x[i], y[mod1(i + 1, n)] - y[i]), 1:n)
        ϑ = atan(y[mod1(k + 1, n)] - y[k], x[mod1(k + 1, n)] - x[k])
        ϑ > π / 2 && (ϑ -= π)
        ϑ <= -π / 2 && (ϑ += π)
        aspect = frame_aspect(x, y, ϑ)
    end
    return ϑ, aspect
end

# extent ratio of the vertices in the frame rotated by ϑ (Inf for a line)
function frame_aspect(x, y, ϑ)
    s = x .* cos(ϑ) .+ y .* sin(ϑ)
    t = y .* cos(ϑ) .- x .* sin(ϑ)
    span_perp = maximum(t) - minimum(t)
    return span_perp > 0 ? (maximum(s) - minimum(s)) / span_perp : Inf
end

"""
$(TYPEDSIGNATURES)

Zone-of-confusion map from the capped boundary polygon. The filled zone is
the exact intersection of the mathematical zone with the physical prior box,
and its boundary is drawn SOLID throughout — Okabe–Ito blue where
curvature-limited, heavier vermillion where it runs along a hard physical
wall (|χ|≤1, positivity, phase ±π), the line weight keeping the two classes
apart in grayscale.
Where the prior cuts the zone off, the *uncapped mathematical* contour
(`x_math`/`y_math`, when provided) continues past the wall as an empty
dashed line with no fill — showing what curvature alone would allow; along
degenerate directions it is unbounded and simply runs off the frame. The
prior box itself is not drawn: the red runs mark where a wall is active, and
the annotation names the active walls with their values (from `box`, e.g.
"… prior-limited by Δχ₁ = 0.2"). Same-unit planes (spin–spin) are drawn
with `DataAspect`, except a needle: a zone whose principal-axis aspect
(`principal_axis`) reaches `NEEDLE_ASPECT` — the exact null spin direction of
the 1.5PN model — is rotated into the frame of that direction (abscissa along
it, ordinate transverse, independent scales) so that its transverse width is
visible; the x label quotes the null slope dχ₂/dχ₁, while wall detection and
the annotation stay in parameter units.
"""
function zone_figure(x::AbstractVector, y::AbstractVector,
    prior_limited::AbstractVector{Bool};
    px::Int, py::Int, box::NTuple{4,Float64},
    prior_frac::Real, degenerate_frac::Real,
    x_math = nothing, y_math = nothing)
    px == py && throw(ArgumentError("map plane must use two distinct parameters"))
    (x_math === nothing) == (y_math === nothing) ||
        throw(ArgumentError("x_math and y_math must be supplied together"))
    x_math === nothing || length(x_math) == length(y_math) == length(x) ||
        throw(DimensionMismatch("x_math/y_math must match the boundary polygon length"))
    same_units = (px in SPIN_INDICES && py in SPIN_INDICES)
    # Needle frame (same-unit planes only, where a rotation is meaningful): a
    # zone drawn as a needle along a null direction is rotated into its
    # principal frame — abscissa along the needle, ordinate transverse — on
    # independent scales, so the transverse width shows instead of an edge-on
    # line. Wall detection below keeps the parameter-unit coordinates. Remove
    # with the 2PN spin–spin term (roadmap §1), which closes the spin zone and
    # returns the parameter frame.
    ϑ, aspect = principal_axis(x, y)
    needle = same_units && aspect >= NEEDLE_ASPECT
    rotate(u, v) = (u .* cos(ϑ) .+ v .* sin(ϑ), v .* cos(ϑ) .- u .* sin(ϑ))
    xd, yd = needle ? rotate(x, y) : (x, y)
    xm_d, ym_d = (x_math === nothing || !needle) ? (x_math, y_math) : rotate(x_math, y_math)
    with_theme(publication_theme()) do
        # squarer canvas for equal-aspect same-unit planes to avoid wide side margins
        fig = Figure(size = (same_units && !needle) ? (820, 830) : (960, 720))

        # Limits symmetric about zero on each axis, sized by everything that
        # must be visible on it — the zone polygon with its wall runs — plus
        # ZONE_LIMIT_MARGIN. The uncapped mathematical contour does not enter:
        # along degenerate directions it is unbounded, and it is free to run
        # off the frame.
        function half_width(v)
            m = maximum(abs, filter(isfinite, v); init = 0.0)
            return m > 0 ? (1 + ZONE_LIMIT_MARGIN) * m : 1.0
        end
        hx = half_width(xd)
        hy = half_width(yd)
        # commensurate axes (DataAspect) share the larger half-width, so the
        # frame stays square in data units
        same_units && !needle && (hx = hy = max(hx, hy))
        x_exponent = axis_exponent(hx)
        y_exponent = axis_exponent(hy)

        if needle
            xlabel = latexstring(
                "\\Delta\\chi_{\\parallel}\\ \\ (\\mathrm{null\\ direction},\\ ",
                "\\mathrm{d}\\chi_2/\\mathrm{d}\\chi_1 = ", @sprintf("%.2f", tan(ϑ)),
                ")")
            ylabel = L"\Delta\chi_{\perp}"
        else
            xlabel = deviation_label(px)
            ylabel = deviation_label(py)
        end
        ax = Axis(fig[1, 1]; xlabel = xlabel, ylabel = ylabel)
        same_units && !needle && (ax.aspect = DataAspect())

        xlo, xhi = zone_axis_ticks!(fig, ax, :x, px, hx, x_exponent)
        ylo, yhi = zone_axis_ticks!(fig, ax, :y, py, hy, y_exponent)
        # a tall same-unit zone (needle along a null spin direction) leaves the
        # x axis narrow: rotate its tick labels so they cannot collide
        if same_units && !needle && (xhi - xlo) < 0.6 * (yhi - ylo)
            ax.xticklabelrotation = π / 4
        end

        poly!(ax, Point2f.(xd, yd); color = (ZONE_BLUE, 0.25), strokewidth = 0)

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
            xm = [isfinite(v) ? Float64(v) : NaN for v in xm_d]
            ym = [isfinite(v) ? Float64(v) : NaN for v in ym_d]
            edge_math = [prior_limited[i] || prior_limited[mod1(i + 1, n)] for i in 1:n]
            mx, my = boundary_runs(xm, ym, edge_math, true)
            isempty(mx) || lines!(ax, mx, my; color = (ZONE_BLUE, 0.75),
                linewidth = 2.4, linestyle = :dash)
        end
        cx, cy = boundary_runs(xd, yd, edge_prior, false)
        bx, by = boundary_runs(xd, yd, edge_prior, true)
        isempty(cx) || lines!(ax, cx, cy; color = ZONE_BLUE, linewidth = 3.0)
        isempty(bx) || lines!(ax, bx, by; color = WALL_VERMILLION, linewidth = 4.5)

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
                ((lox, hix, prior_x, 1e-6 * (maximum(x) - minimum(x)), px),
                (loy, hiy, prior_y, 1e-6 * (maximum(y) - minimum(y)), py))
                lo_hit = isfinite(lo_e) && any(v -> abs(v - lo_e) < tol, vals)
                hi_hit = isfinite(hi_e) && any(v -> abs(v - hi_e) < tol, vals)
                isph = idx == PHASE_INDEX
                if lo_hit && hi_hit && isapprox(lo_e, -hi_e; rtol = 1e-9)
                    push!(walls, string(devtex(idx), " = \\pm ", fmt_edge(abs(hi_e), isph)))
                else
                    lo_hit && push!(walls, string(devtex(idx), " = ", fmt_edge(lo_e, isph)))
                    hi_hit && push!(walls, string(devtex(idx), " = ", fmt_edge(hi_e, isph)))
                end
            end
            # worded without a hyphen (the math engine sets "-" as a minus sign)
            # and short enough to clear the axis multiplier at the left end of
            # the same row
            msg =
                isempty(walls) ? "\\text{Prior bounds}" :
                string("\\text{Prior walls}\\ ", join(walls, ",\\;\\ "), "\\ \\text{bound}")
            msg *= string("\\ ", pcttex(prior_frac), "\\ \\text{of directions}")
            # in the needle frame the long axis is the prior-capped null
            # direction, not a curvature scale
            needle && (msg *= "\\ \\text{and the long axis}")
            degenerate_frac > 0 && (
                msg *= string("\\;\\ (", pcttex(degenerate_frac), "\\ \\text{degenerate})")
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
