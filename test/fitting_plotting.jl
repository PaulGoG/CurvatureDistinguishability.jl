# Fit statistics, monitoring panels and figures. Included by runtests.jl.

@testset "Monitoring diagnostic panels" begin
    d = 10 .^ range(-3, -1, length = 8)
    th = d .^ 4
    num = th .* 1.01
    panel = CD.Orchestrator.sweep_diagnostic_panel(d, num, th, trues(8), 4.0, 0.01)
    @test occursin("fitted slope", panel) && occursin("8/8", panel)
    @test occursin("theory", panel)
    empty_panel =
        CD.Orchestrator.sweep_diagnostic_panel(d, zeros(8), th, falses(8), NaN, NaN)
    @test occursin("no positive", empty_panel)
    mp = CD.Orchestrator.map_diagnostic_panel(collect(range(0, 2π, length = 16)),
        fill(1.0, 16), 0.25)
    @test occursin("prior-limited directions: 25.0%", mp)
end

@testset "Fitting: slope and clean-point rule" begin
    slope, slope_err = CD.Fitting.loglog_slope([1.0, 10.0, 100.0], [1.0, 1e4, 1e8])
    @test slope ≈ 4.0 atol = 1e-12
    @test slope_err ≈ 0.0 atol = 1e-9
    @test isnan(CD.Fitting.loglog_slope([1.0, 10.0], [1.0, 1e4])[2]) # no error with 2 points
    @test CD.Fitting.above_floor_mask([1.0, 2.0, 3.0], 2.0) == [false, false, true]
    @test CD.Fitting.above_floor_mask([0.0, 1.0, -1.0], NaN) == [false, true, false]
    # the O(δ⁵) window keeps clean points within the departure bound only
    @test CD.Fitting.perturbative_mask(
        [1.05, 0.8, 1.4, 1.0],
        [true, true, true, false],
        0.3,
    ) ==
          [true, true, false, false]
    # perturbative-window slope: the window excludes the points where the
    # O(δ) departure has grown beyond 30 %, which bias the all-point slope
    deltas = 10.0 .^ range(-2, 2; length = 30)
    D2_theo = deltas .^ 4
    D2_num = D2_theo .* (1 .+ 0.02 .* deltas)
    ratio = D2_num ./ D2_theo
    window = CD.Fitting.perturbative_mask(ratio, trues(30), 0.3)
    @test count(window) == 24
    s_all, e_all = CD.Fitting.loglog_slope(deltas, D2_num)
    s_win, e_win = CD.Fitting.loglog_slope(deltas[window], D2_num[window])
    @test s_all ≈ 4.080383377803117 rtol = 1e-9
    @test s_win ≈ 4.0241212293066635 rtol = 1e-9
    @test e_win ≈ 0.004121123271661602 rtol = 1e-6
    @test s_win < s_all

    # slope, c₁ and their standard errors against the normal equations of
    # ordinary least squares, β = X \ y and cov = s² (XᵀX)⁻¹ with
    # s² = SSR/(n − 2), on data that no model of the package generated
    xs = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    ys = [3.1, 44.0, 790.0, 1.21e4, 2.1e5, 3.0e6]
    X = hcat(ones(6), log10.(xs))
    b = X \ log10.(ys)
    s2 = sum(abs2, log10.(ys) .- X * b) / 4
    slope_ols, slope_se = CD.Fitting.loglog_slope(xs, ys)
    @test slope_ols ≈ b[2] rtol = 1e-12
    @test slope_se ≈ sqrt(s2 * inv(X' * X)[2, 2]) rtol = 1e-10
    dd = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
    rr = [1.012, 1.018, 1.047, 1.055, 1.11, 1.13]
    Xc = hcat(dd, dd .^ 2)
    bc = Xc \ (rr .- 1)
    s2c = sum(abs2, (rr .- 1) .- Xc * bc) / 4
    c1_fit, c1_se, c2_fit = CD.Fitting.ratio_correction_fit(dd, rr)
    @test c1_fit ≈ bc[1] rtol = 1e-10
    @test c2_fit ≈ bc[2] rtol = 1e-10
    @test c1_se ≈ sqrt(s2c * inv(Xc' * Xc)[1, 1]) rtol = 1e-8

    # angular fraction of flagged directions: half-gap weights on the
    # cyclic grid. Three flagged directions clustered at π between coarse
    # neighbors at π/2 and 3π/2 subtend (π/4 + 0.01 + π/4) of 2π, while
    # they are half of the six directions
    ang = [0.0, π / 2, π - 0.01, π, π + 0.01, 3π / 2]
    flagged = [false, false, true, true, true, false]
    @test CD.Fitting.angular_fraction(ang, flagged) ≈ (π / 2 + 0.01) / (2π) rtol = 1e-12
    perm = [4, 1, 6, 3, 2, 5] # the order of the grid is immaterial
    @test CD.Fitting.angular_fraction(ang[perm], flagged[perm]) ≈
          (π / 2 + 0.01) / (2π) rtol = 1e-12
    uniform = collect(range(0, 2π; length = 9))[1:8]
    @test CD.Fitting.angular_fraction(uniform, isodd.(1:8)) ≈ 0.5 rtol = 1e-12
    @test CD.Fitting.angular_fraction(uniform, trues(8)) ≈ 1.0 rtol = 1e-12
    @test CD.Fitting.angular_fraction(uniform, falses(8)) == 0.0
    @test CD.Fitting.angular_fraction(Float64[], Bool[]) == 0.0
    @test_throws DimensionMismatch CD.Fitting.angular_fraction([0.0, 1.0], [true])
    # a grid that lists both 0 and 2π shares one direction between the copies
    @test CD.Fitting.angular_fraction([0.0, π / 2, π, 3π / 2, 2π],
        [true, false, false, false, true]) ≈ 0.25 rtol = 1e-12
    @test CD.Fitting.angular_fraction([0.0, π / 2, π, 3π / 2],
        [true, false, false, false]) ≈ 0.25 rtol = 1e-12
    # more than one turn has no angular measure
    @test_throws ArgumentError CD.Fitting.angular_fraction([0.0, 3π, 6π],
        [true, false, false])
end

@testset "Ratio-correction fit (O(δ⁵) quantification)" begin
    d = [0.01, 0.05, 0.1, 0.2, 0.3]
    r = 1.0 .+ 0.3 .* d .- 0.1 .* d .^ 2
    c1, c1_err, c2 = CD.Fitting.ratio_correction_fit(d, r)
    @test c1 ≈ 0.3 atol = 1e-6
    @test c2 ≈ -0.1 atol = 1e-6
    @test c1_err < 1e-10 # exact model → zero residual
    c1n, _, _ = CD.Fitting.ratio_correction_fit(d[1:2], r[1:2])
    @test isnan(c1n) # too few points

    # floor detection: the contiguous small-δ run above the ratio
    # threshold; a super-threshold ratio at large δ (breakdown of the
    # leading-order law) is not a floor
    @test isnan(CD.Fitting.optimizer_floor([1.0, 2.0], [1.1, 1.9], 2.0))
    @test isnan(CD.Fitting.optimizer_floor([1.0, 2.0], [1.1, 2.5], 2.0))
    @test CD.Fitting.optimizer_floor([2.0, 1.0, 3.0], [2.5, 2.2, 1.0], 2.0) == 2.0
    @test CD.Fitting.optimizer_floor([1.0, 2.0], [1.9, 1.1], 1.5) == 1.0
    # a fit that returned exactly zero is on the floor, and so are the
    # floor-dominated separations after it
    @test CD.Fitting.optimizer_floor([0.0, 1e-13, 1e-12, 1.0], [0.0, 1e5, 1e3, 1.01],
        2.0) == 1e-12
    # validity radius: smallest positive root of |c₁δ + c₂δ²| = fraction, worked
    # by hand. c₁ = 0.3, c₂ = −0.1: 0.3δ − 0.1δ² = 0.1 ⇒ δ = (3 − √5)/2
    @test CD.Fitting.validity_radius(0.3, 0.01, -0.1, 0.1) ≈ (3 - sqrt(5)) / 2 rtol = 1e-14
    # an insignificant c₁ (below twice its error) is dropped: δ = √(0.1/|c₂|)
    @test CD.Fitting.validity_radius(1e-6, 3e-4, -3.1e-7, 0.1) ≈ sqrt(0.1 / 3.1e-7) rtol =
        1e-14
    @test CD.Fitting.validity_radius(0.2, 0.01, 0.0, 0.1) ≈ 0.5 rtol = 1e-14
    @test CD.Fitting.validity_radius(0.2, 0.01, NaN, 0.1) ≈ 0.5 rtol = 1e-14
    @test CD.Fitting.validity_radius(NaN, NaN, -0.4, 0.1) ≈ 0.5 rtol = 1e-14
    @test CD.Fitting.validity_radius(1e-6, 1.0, 0.0, 0.1) == Inf
    # c₁ and c₂ of opposite sign that never reach the fraction on one branch:
    # c₁ = 0.1, c₂ = −0.1, fraction 0.5: 0.1δ − 0.1δ² = −0.5 ⇒ δ = (1 + √21)/2
    @test CD.Fitting.validity_radius(0.1, 0.001, -0.1, 0.5) ≈ (1 + sqrt(21)) / 2 rtol =
        1e-14
    @test_throws ArgumentError CD.Fitting.validity_radius(0.1, 0.01, 0.1, 0.0)
    # a c₂ at round-off level beside a significant c₁: the root tends to
    # fraction/|c₁| (the textbook quadratic formula loses it to cancellation)
    @test CD.Fitting.validity_radius(0.5, 0.01, 1e-17, 0.1) ≈ 0.2 rtol = 1e-12
    @test CD.Fitting.validity_radius(0.5, 0.01, -1e-17, 0.1) ≈ 0.2 rtol = 1e-12
    @test CD.Fitting.validity_radius(1.0, 0.01, 1e-15, 0.1) ≈ 0.1 rtol = 1e-12
    # and with a negative c₁, where the sign of the discriminant root matters
    @test CD.Fitting.validity_radius(-0.5, 0.01, 1e-17, 0.1) ≈ 0.2 rtol = 1e-12
    @test CD.Fitting.validity_radius(-0.5, 0.01, -1e-17, 0.1) ≈ 0.2 rtol = 1e-12
    @test CD.Fitting.validity_radius(-1.0, 0.01, -1e-15, 0.1) ≈ 0.1 rtol = 1e-12
end

@testset "Plotting utilities" begin
    vals, labels = CD.Plotting.decade_ticks(2e-22, 3e-4)
    @test all(2e-22 .<= vals .<= 3e-4)
    ps = round.(Int, log10.(vals))
    @test all(diff(ps) .== ps[2] - ps[1]) # uniform decade step
    @test all(p -> mod(p, ps[2] - ps[1]) == 0, ps) # family-consistent anchor

    pt = CD.Plotting.pi_ticks(-π / 2 - 0.1, π / 6 + 0.1)
    @test pt !== nothing
    vals_π, _ = pt
    steps = diff(vals_π)
    @test all(isapprox.(steps, steps[1]; rtol = 1e-12)) # single denominator

    # coverage: a ±0.78π range must be ticked out to ±3π/4, not stop at ±π/2
    vc, _ = CD.Plotting.pi_ticks(-0.78π, 0.78π)
    @test length(vc) >= 5
    @test maximum(vc) ≈ 3π / 4 atol = 1e-12
    @test minimum(vc) ≈ -3π / 4 atol = 1e-12

    @test CD.Plotting.axis_exponent(6e-4) == -4
    @test CD.Plotting.axis_exponent(2.0) == 0

    # residual-band ticks: from the true band [1e-4, 0.05] every whole-power
    # decade is present, including the low endpoint
    vb, _ = CD.Plotting.decade_ticks(1e-4, 0.05)
    @test round.(Int, log10.(vb)) == [-4, -3, -2]

    # 10⁰ always renders as plain "1", on decade ticks and the 1-2-5 series
    _, l0 = CD.Plotting.decade_ticks(0.5, 50.0)
    @test l0[1].s == "\$1\$"
    v125, l125 = CD.Plotting.log_ticks_125(5e-2, 6.0)
    @test any(≈(1.0), v125) && any(≈(2e-1), v125)
    @test l125[findfirst(≈(1.0), v125)].s == "\$1\$"
    @test !any(l -> occursin("10^{0}", l.s), l125)
    # …and 10¹ as plain 10 (products fold: 2×10¹ → 20)
    @test l0[2].s == "\$10\$"
    v125b, l125b = CD.Plotting.log_ticks_125(5.0, 60.0)
    @test l125b[findfirst(≈(10.0), v125b)].s == "\$10\$"
    @test l125b[findfirst(≈(20.0), v125b)].s == "\$20\$"
    # annotations go scientific outside exponents −1..1, 3 significant digits
    @test CD.Plotting.sci_latex(0.00173) == "1.73\\times 10^{-3}"
    @test CD.Plotting.sci_latex(23.4) == "23.4"

    # offset ticks: one factored power of 10 with
    # integer mantissas preferring multiples of 5, limits snapped outward
    # so the frame ends exactly on the outermost labeled ticks
    ot = CD.Plotting.offset_ticks(-1.32e-3, 1.32e-3)
    @test ot !== nothing
    ovals, olabels, e10, lo_s, hi_s = ot
    @test e10 == -4
    @test ovals[1] == lo_s && ovals[end] == hi_s
    @test lo_s <= -1.32e-3 && hi_s >= 1.32e-3
    @test all(m -> m % 5 == 0, round.(Int, ovals ./ 10.0^e10))
    @test occursin("15", olabels[end].s) && !occursin("10^", olabels[end].s)
    ot2 = CD.Plotting.offset_ticks(-1.15e-4, 1.8e-4) # asymmetric range
    @test ot2 !== nothing
    @test ot2[4] <= -1.15e-4 && ot2[5] >= 1.8e-4

    # symmetric ticks of the zone-map axes: 5 or 7 ticks, zero among
    # them, inside ±h, integer or half-integer mantissas of 10ᵏ
    for h in (1.32e-3, 0.21, 0.66, 8.6, 330.0)
        st = CD.Plotting.symmetric_ticks(h)
        @test st !== nothing
        svals, k10 = st
        @test length(svals) in (5, 7)
        @test 0.0 in svals && svals ≈ -reverse(svals)
        @test maximum(svals) <= h
        mant = svals ./ 10.0^k10
        @test all(m -> isapprox(2m, round(2m); atol = 1e-9), mant)
    end
    @test CD.Plotting.symmetric_ticks(0.0) === nothing
    @test CD.Plotting.symmetric_ticks(Inf) === nothing

    # binomial smoothing: a constant is invariant, an interior impulse
    # keeps its sum, and zero passes copy
    @test CD.Plotting.binomial_smooth(fill(2.5, 9), 3) ≈ fill(2.5, 9) rtol = 1e-14
    impulse = zeros(15)
    impulse[8] = 1.0
    @test sum(CD.Plotting.binomial_smooth(impulse, 2)) ≈ 1.0 rtol = 1e-14
    @test CD.Plotting.binomial_smooth(impulse, 0) == impulse

    # per-tick common-exponent scientific notation (fallback when no
    # clean integer-mantissa grid exists): 5e-5 renders as 0.5×10⁻⁴,
    # never mixing 10⁻⁴ with 10⁻⁵ labels on one axis
    sl = CD.Plotting.sci_tick_labels([-1.5e-4, 0.0, 5e-5])
    @test occursin("-1.5", sl[1].s) && occursin("10^{-4}", sl[1].s)
    @test !occursin("times", sl[2].s) # zero renders as plain "0"
    @test occursin("0.5", sl[3].s) && occursin("10^{-4}", sl[3].s)
    @test !occursin("10^{-5}", sl[3].s)

    # annotation number formatting: LaTeX ×10ⁿ, never bare e-notation
    @test CD.Plotting.sci_latex(9.34e-5) == "9.34\\times 10^{-5}"
    @test CD.Plotting.sci_latex(0.316) == "0.316"
    @test CD.Plotting.sci_latex(0) == "0"

    # fit coefficients: mantissa ×10ⁿ with two decimals outside exponents −1..1
    # slope annotation: decimals follow the uncertainty (≥ 3, ≤ 6)
    @test CD.Plotting.slope_latex(3.99984, 0.00012) == "3.9998 \\pm 0.0001"
    @test CD.Plotting.slope_latex(3.8241, 0.0432) == "3.824 \\pm 0.043"
    @test CD.Plotting.slope_latex(3.9812, 0.0081) == "3.981 \\pm 0.008"
    @test CD.Plotting.slope_latex(4.0, NaN) == "4.000"
    @test CD.Plotting.coef_latex(0.001278) == "1.28\\times 10^{-3}"
    @test CD.Plotting.coef_latex(-0.0235) == "{-}2.35\\times 10^{-2}" # tight unary minus
    @test CD.Plotting.coef_latex(1.5) == "1.50" # ×10⁰ factor omitted
    @test CD.Plotting.coef_latex(14.46) == "14.5" # 2×10¹-style products fold
    @test CD.Plotting.coef_latex(0.2346) == "0.235" # …and 10⁻¹ likewise
    @test CD.Plotting.coef_latex(-14.46) == "{-}14.5"
    @test CD.Plotting.tex_minus("-3.4") == "{-}3.4" &&
          CD.Plotting.tex_minus("3.4") == "3.4"
    # annotation values: three significant digits, trailing zeros kept, 1 ≤ |v| < 1000
    @test CD.Plotting.fixed_sig_latex(30.03) == "30.0"
    @test CD.Plotting.fixed_sig_latex(31.44) == "31.4"
    @test CD.Plotting.fixed_sig_latex(316.2) == "316"
    @test CD.Plotting.fixed_sig_latex(1.318) == "1.32"
    @test CD.Plotting.fixed_sig_latex(-2.5) == "{-}2.50"
    @test CD.Plotting.fixed_sig_latex(0.05) == CD.Plotting.sci_latex(0.05)
    # per-tick common-exponent labels never show a power in −1..1
    plain = CD.Plotting.sci_tick_labels([-1.5, 0.0, 0.5])
    @test plain[1].s == "\$-1.5\$" && plain[3].s == "\$0.5\$"
    @test !any(l -> occursin("times", l.s), plain)

    # needle frame of same-unit zone maps: a zone drawn edge-on along a
    # null direction is rotated into its principal frame (abscissa along,
    # ordinate transverse), a round zone keeps the parameter frame with
    # equal data aspect
    null_slope = -2.29
    ϑ_null = atan(null_slope)
    s_needle = range(-0.4, 0.4; length = 40)
    w_needle = 2e-3
    x_needle = vcat(s_needle .* cos(ϑ_null) .- w_needle * sin(ϑ_null),
        reverse(s_needle) .* cos(ϑ_null) .+ w_needle * sin(ϑ_null))
    y_needle = vcat(s_needle .* sin(ϑ_null) .+ w_needle * cos(ϑ_null),
        reverse(s_needle) .* sin(ϑ_null) .- w_needle * cos(ϑ_null))
    ϑ_fit, aspect = CD.Plotting.principal_axis(x_needle, y_needle)
    @test tan(ϑ_fit) ≈ null_slope rtol = 1e-6
    @test aspect ≈ 0.8 / (2w_needle) rtol = 1e-6
    @test aspect > CD.Plotting.NEEDLE_ASPECT
    fig_needle = CD.Plotting.zone_figure(x_needle, y_needle, falses(length(x_needle));
        px = 5, py = 6, box = (-1.8, 0.2, -1.8, 0.2),
        prior_frac = 0.0, degenerate_frac = 0.0)
    ax_needle = first(filter(c -> c isa CD.Plotting.Axis, fig_needle.content))
    @test occursin("parallel", ax_needle.xlabel[])
    @test occursin("{-}2.29", ax_needle.xlabel[])
    @test !(ax_needle.aspect[] isa CD.Plotting.DataAspect)
    φ_round = range(0, 2π; length = 64)[1:(end-1)]
    _, aspect_round =
        CD.Plotting.principal_axis(0.1 .* cos.(φ_round), 0.1 .* sin.(φ_round))
    @test aspect_round ≈ 1 rtol = 1e-2
    fig_round = CD.Plotting.zone_figure(0.1 .* cos.(φ_round), 0.1 .* sin.(φ_round),
        falses(length(φ_round));
        px = 5, py = 6, box = (-1.8, 0.2, -1.8, 0.2),
        prior_frac = 0.0, degenerate_frac = 0.0)
    ax_round = first(filter(c -> c isa CD.Plotting.Axis, fig_round.content))
    @test ax_round.xlabel[] == CD.Plotting.deviation_label(5)
    @test ax_round.aspect[] isa CD.Plotting.DataAspect
    # display units: coordinates and frame follow the display_scale
    # factors and the labels carry the units; the computation's rescaled
    # coordinates stay unlabeled (above)
    scale = CD.Plotting.display_scale(CD.Physics.WaveformParams())
    @test scale[2] ≈ 1 / 4.925490947e-6 rtol = 1e-9
    @test scale[3] == 1.0e6
    fig_units = CD.Plotting.Figure()
    panel_units = CD.Plotting.zone_panel!(fig_units[1, 1],
        1e-4 .* cos.(φ_round), 2e-6 .* sin.(φ_round), falses(length(φ_round));
        px = 2, py = 3, box = (-1.0, 1.0, -1.0, 1.0),
        prior_frac = 0.0, degenerate_frac = 0.0, scale)
    @test occursin("\\mathcal{M}\\ [M_\\odot]", panel_units.axis.xlabel[])
    @test occursin("t_c\\ [\\mathrm{s}]", panel_units.axis.ylabel[])
    (_, xhi), (_, yhi) = panel_units.axis.limits[]
    margin = 1 + CD.Plotting.ZONE_LIMIT_MARGIN
    @test xhi ≈ margin * maximum(abs, 1e-4 .* cos.(φ_round)) * scale[2] rtol = 1e-9
    @test yhi ≈ margin * maximum(abs, 2e-6 .* sin.(φ_round)) * scale[3] rtol = 1e-9
end

@testset "Plotting: residual panel range" begin
    # a spectrum tapering off at the high-frequency end: the top frame
    # follows the signal means (at most eight decades deep), the bottom
    # frame reaches at most eight decades below the residual maximum
    f = 10 .^ range(-4, -1.3; length = 300)
    cut = 0.5 .* (1 .- tanh.((f .- 0.02) ./ 0.002))
    sig = 1e10 .* (f ./ 1e-3) .^ (-7 / 3) .* cut .+ 1e-40
    spec = DataFrame(f = f, sig_mean_A = sig, sig_mean_E = 0.8 .* sig,
        bf_mean_A = sig, bf_mean_E = 0.8 .* sig,
        res_mean_A = 1e-3 .* sig, res_mean_E = 8e-4 .* sig,
        sig_min_A = 0.5 .* sig, sig_max_A = 2 .* sig,
        sig_min_E = 0.4 .* sig, sig_max_E = 1.6 .* sig,
        res_min_A = 5e-4 .* sig, res_max_A = 2e-3 .* sig,
        res_min_E = 4e-4 .* sig, res_max_E = 1.6e-3 .* sig)
    spec.res_min_A[1:20] .= 0.0
    meta = ResidualFigureMeta(10.0, 3e-8, 1.0, 1.0)
    p = residual_panel!(CD.Plotting.Figure()[1, 1], spec, meta)
    lims_top = p.top.limits[]
    @test lims_top[2][1] ≈ max(minimum(sig), maximum(sig) * 1e-8) / 3 rtol = 1e-6
    @test lims_top[2][2] ≈ maximum(sig) * 3 rtol = 1e-6
    lims_bot = p.bottom.limits[]
    @test lims_bot[2][1] ≈ maximum(spec.res_max_A) * CD.Plotting.FRAME_DEPTH rtol = 1e-6
    fig_r = residual_figure(spec, meta)
    @test fig_r isa CD.Plotting.Figure
end
