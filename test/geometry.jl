# Tangent basis, curvature, residuals, bounds and the observation window. Included by runtests.jl.

@testset "Geometry: basis, fused derivatives" begin
    basis = @inferred compute_tangent_basis(THETA0, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
    @test basis isa Vector{NTuple{2,Vector{ComplexF64}}}
    @test 1 <= length(basis) <= 6
    # the χ_a direction is exactly degenerate in this model → rank 5
    @test length(basis) == 5
    for i in eachindex(basis), j in eachindex(basis)
        ip = multi_channel_inner_product(basis[i], basis[j], FIX_SN, FIX_DF)
        @test isapprox(ip, i == j ? 1.0 : 0.0; atol = 1e-9)
    end

    # single-channel inner product through the public path: the closed
    # form 4 df Σ Re(h1* h2)/Sn, and the multi-channel sum reducing to it
    A2, E2 = channel_strain(THETA0, FIX_FREQS, FIX_WP)
    ip_closed =
        4 * FIX_DF * sum(real(conj(a) * e) / s for (a, e, s) in zip(A2, E2, FIX_SN))
    @test inner_product(A2, E2, FIX_SN, FIX_DF) ≈ ip_closed rtol = 1e-12
    @test inner_product(A2, A2, FIX_SN, FIX_DF) > 0
    @test multi_channel_inner_product((A2,), (E2,), FIX_SN, FIX_DF) ≈
          inner_product(A2, E2, FIX_SN, FIX_DF) rtol = 1e-14
    @test multi_channel_inner_product((A2, E2), (A2, E2), FIX_SN, FIX_DF) ≈
          inner_product(A2, A2, FIX_SN, FIX_DF) + inner_product(E2, E2, FIX_SN, FIX_DF) rtol =
        1e-14

    # fused nested-dual derivatives against independent ForwardDiff passes
    gvec(s) = [sin(2s) + s^3, exp(s) * cos(s)]
    h0, dh, d2h = CD.Geometry.value_and_directional_derivs(gvec, 0.3)
    @test h0 ≈ gvec(0.3) rtol = 1e-14
    @test dh ≈ ForwardDiff.derivative(gvec, 0.3) rtol = 1e-12
    @test d2h ≈ ForwardDiff.derivative(s -> ForwardDiff.derivative(gvec, s), 0.3) rtol =
        1e-12
end

@testset "Whitened Jacobian factor" begin
    # RᵀR is the Fisher matrix built from inner products of the Jacobian
    # columns; the blocked factorization does not depend on the block size
    J = ForwardDiff.jacobian(p -> CD.Geometry.flat_response(p, FIX_FREQS, FIX_WP), THETA0)
    cols = [
        CD.Geometry.unflatten_channels(view(J, :, i), length(FIX_FREQS), Val(2))
        for i in 1:6
    ]
    fisher = [
        multi_channel_inner_product(cols[i], cols[j], FIX_SN, FIX_DF)
        for i in 1:6, j in 1:6
    ]
    for chunk in (37, 4096)
        R = CD.Geometry.whitened_jacobian_factor(THETA0, FIX_FREQS, FIX_SN, FIX_DF,
            FIX_WP; chunk)
        @test size(R) == (6, 6)
        @test istriu(R)
        @test maximum(abs.(R' * R .- fisher)) <= 1e-10 * maximum(abs, fisher)
    end
    @test_throws ArgumentError CD.Geometry.whitened_jacobian_factor(THETA0,
        FIX_FREQS[1:1], FIX_SN[1:1], FIX_DF, FIX_WP)
    # the transform makes the Gauss–Newton Hessian the identity on every
    # constrained direction and keeps a flat direction in parameter units:
    # at 1.5PN order one spin combination is flat, at 2PN order off the
    # residual line none is
    wp2 = waveform_params(; merge(FIX_PHYS, (; eta = 2 / 9, phase_pn_order = 2))...)
    θ_off = [1.0, 1.5, 2.0, 0.0, 0.5, -0.387]
    R2 = CD.Geometry.whitened_jacobian_factor(θ_off, FIX_FREQS, FIX_SN, FIX_DF, wp2)
    T2 = CD.Inference.whitening_transform(R2)
    # (R T)ᵀ(R T) = I: formed from R T, not from the Fisher matrix RᵀR, whose
    # condition number would swamp the check
    RT2 = R2 * T2
    @test RT2' * RT2 ≈ Matrix(1.0I, 6, 6) atol = 1e-10
    # the decisive check is against the whitened Jacobian itself, with R from
    # a blocked factorization: (J_w T)ᵀ(J_w T) = I to ~ε·cond(J_w). A factor
    # from the Fisher matrix (normal equations) carries errors ~ε·cond(J_w)²
    # on the weak direction (cond(J_w) ~ 1e6 here), which this tolerance rejects
    J2 = ForwardDiff.jacobian(p -> CD.Geometry.flat_response(p, FIX_FREQS, wp2), θ_off)
    Jw2 = J2 .* repeat(sqrt.(4 * FIX_DF ./ FIX_SN), 4)
    T2b = CD.Inference.whitening_transform(
        CD.Geometry.whitened_jacobian_factor(θ_off, FIX_FREQS, FIX_SN, FIX_DF, wp2;
            chunk = 37))
    @test (Jw2 * T2b)' * (Jw2 * T2b) ≈ Matrix(1.0I, 6, 6) atol = 1e-8
    σ_off = svdvals(Jw2)
    @test σ_off[1] / σ_off[end] > 1e4 # the test is meaningful only for a weak direction
    @test_throws ArgumentError CD.Inference.whitening_transform(zeros(6, 6))
    R15 = CD.Geometry.whitened_jacobian_factor(THETA0, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
    T15 = CD.Inference.whitening_transform(R15)
    RT15 = R15 * T15
    G15 = RT15' * RT15
    σ = svdvals(R15)
    @test σ[end] < CD.Inference.WHITENING_RTOL * σ[1] # the flat spin combination
    @test count(i -> isapprox(G15[i, i], 1.0; atol = 1e-6), 1:6) == 5
    @test count(j -> isapprox(norm(T15[:, j]), 1.0; rtol = 1e-12), 1:6) >= 1
    # re-orthogonalization: near the 2PN residual line the sixth tangent column
    # is weak (independent part ~1e-9 of its norm); the basis stays orthonormal
    # to round-off and keeps rank 6
    let θ_near = [1.0, 1.5, 2.0, 0.0, 0.5, 0.5 * (-1238 / 901) + 1e-3]
        basis_near = compute_tangent_basis(θ_near, FIX_FREQS, FIX_SN, FIX_DF, wp2)
        @test length(basis_near) == 6
        gram = [
            multi_channel_inner_product(a, b, FIX_SN, FIX_DF)
            for a in basis_near, b in basis_near
        ]
        @test maximum(abs.(gram .- Matrix(1.0I, 6, 6))) < 1e-12
    end
end

@testset "Geometry A/B against reference fixtures" begin
    # 1D sweep geometry
    sweep_fix = CSV.read(joinpath(FIXDIR, "sweep.csv"), DataFrame)
    u_raw = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
    K_u, g_uu =
        compute_extrinsic_curvature(THETA0, u_raw, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
    K_norm = K_u / g_uu^2
    @test K_norm ≈ sweep_fix.K_u_Norm[1] rtol = 1e-8
    @test all(
        isapprox.(sweep_fix.D2_Theoretical,
            (1 / 16) .* K_norm .* sweep_fix.Delta .^ 4; rtol = 1e-8),
    )

    # 2D map geometry: per-angle K, g and boundary radius
    basis = compute_tangent_basis(THETA0, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
    for (px, py) in ((2, 3), (5, 6), (2, 4))
        fix = CSV.read(joinpath(FIXDIR, "map_$(px)_$(py).csv"), DataFrame)
        Kmax = maximum(fix.K_raw)
        for row in eachrow(fix)
            dir = zeros(6)
            dir[px] = cos(row.Angle)
            dir[py] = sin(row.Angle)
            K, g = compute_extrinsic_curvature_from_basis(THETA0, dir, basis,
                FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
            # directions whose curvature is round-off (∂²h in the tangent
            # space, e.g. the pure coalescence-phase direction of a
            # single-harmonic signal) carry no reproducible K or radius
            if row.K_raw > 1e-10 * Kmax
                @test K ≈ row.K_raw rtol = 1e-8
                # capped-mapping radius formula ≡ reference x/y bounds (s(φ) cancels)
                r_math = (16.0 * 1.0 / K)^(1 / 4)
                @test r_math ≈ hypot(row.X_Bound, row.Y_Bound) rtol = 1e-8
            end
            if row.G_uu > 1e-6
                @test g ≈ row.G_uu rtol = 1e-6
            end
        end
    end
end

@testset "Residual spectrum (Residuals)" begin
    u_raw = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
    _, g_uu =
        compute_extrinsic_curvature(THETA0, u_raw, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
    u_norm = u_raw ./ sqrt(g_uu)
    delta = 0.05
    spec, integrals = residual_spectrum(THETA0, u_norm, 1.0, delta, THETA0,
        FIX_FREQS, FIX_SN, FIX_DF, FIX_WP; n_windows = 16)
    @test spec isa DataFrame
    # f plus mean/min/max of signal, best fit and residual for A and E
    @test names(spec)[1] == "f" && ncol(spec) == 19
    @test issorted(spec.f) && nrow(spec) <= 16
    @test FIX_FREQS[1] <= spec.f[1] && spec.f[end] <= FIX_FREQS[end]
    @test all(spec.res_min_A .<= spec.res_mean_A .<= spec.res_max_A)
    @test all(spec.sig_min_E .<= spec.sig_mean_E .<= spec.sig_max_E)
    # with the base source as the "best fit" the residual is exactly the
    # second source, so the residual integrals sum to its SNR²
    p2 = THETA0 .+ delta .* u_norm
    ch2 = channel_strain(p2, FIX_FREQS, FIX_WP)
    @test integrals.int_A + integrals.int_E ≈
          multi_channel_inner_product(ch2, ch2, FIX_SN, FIX_DF) rtol = 1e-12
    @test integrals.int_A > 0 && integrals.int_E > 0
    # a window budget beyond the bin count is capped at one window per bin
    spec_fine, _ = residual_spectrum(THETA0, u_norm, 1.0, delta, THETA0,
        FIX_FREQS, FIX_SN, FIX_DF, FIX_WP; n_windows = 10^6)
    @test nrow(spec_fine) <= length(FIX_FREQS)
end

@testset "Second source, boundary radius, unbounded cap" begin
    # half the amplitude ⇔ twice the luminosity distance
    p2 = second_source([1.0, 2.0, 3.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        0.1, 0.5)
    @test p2 == [2.0, 2.1, 3.0, 0.0, 0.5, 0.5]
    @test boundary_radius(16.0, 1.0) == 1.0
    # prior-wall capping: a crossover vertex reaching the wall to round-off
    # is snapped onto it and flagged; interior and unbounded directions are not
    @test cap_at_prior(0.3 * (1 - 1e-12), 0.3) == (0.3, true)
    @test cap_at_prior(0.5, 0.3) == (0.3, true)
    @test cap_at_prior(0.3 * (1 - 1e-6), 0.3) == (0.3 * (1 - 1e-6), false)
    @test cap_at_prior(0.5, Inf) == (0.5, false)
    @test boundary_radius(16.0, 4.0) ≈ sqrt(2.0)
    @test isinf(boundary_radius(0.0, 1.0)) && isinf(boundary_radius(1e-320, 1.0))
    r = [1.0, Inf, 2.0, Inf]
    @test cap_unbounded_radii!(r, 5.0) == (2, 10.0)
    @test r == [1.0, 10.0, 2.0, 10.0]
    @test isequal(cap_unbounded_radii!(r, 5.0), (0, NaN))
end

@testset "Bounds and polar capping" begin
    b = default_bounds()
    box = deviation_box(b, THETA0, 5, 6) # spins at 0.8
    @test all(isapprox.(collect(box), [-1.8, 0.2, -1.8, 0.2]; rtol = 1e-12))
    box_p = deviation_box(b, THETA0, 2, 4) # mass (lower −1.5) and periodic phase
    @test box_p[1] ≈ -1.5 && box_p[2] == Inf
    @test box_p[3] ≈ -π && box_p[4] ≈ π
    @test b.period[4] ≈ 4π && b.deviation_cap[4] ≈ π
    # the four-argument form caps at the antipode; a wider cap is refused
    b_antipode = ParameterBounds(b.lower, b.upper, b.periodic, b.period)
    @test deviation_box(b_antipode, THETA0, 2, 4)[4] ≈ 2π
    @test_throws ArgumentError ParameterBounds(b.lower, b.upper, b.periodic,
        b.period, (0.0, 0.0, 0.0, 3π, 0.0, 0.0))

    @test ray_box_crossing(1.0, 0.0, -1.8, 0.2, -1.8, 0.2) ≈ 0.2
    @test ray_box_crossing(-1.0, 0.0, -1.8, 0.2, -1.8, 0.2) ≈ 1.8
    @test ray_box_crossing(cos(π / 4), sin(π / 4), -1.0, 1.0, -1.0, 0.5) ≈
          0.5 / sin(π / 4)
    @test ray_box_crossing(0.0, 1.0, -Inf, Inf, -Inf, Inf) == Inf

    # base point on a bound must be rejected
    θ_edge = copy(THETA0)
    θ_edge[5] = 1.0
    @test_throws ArgumentError deviation_box(b, θ_edge, 5, 6)

    θ_out = copy(THETA0)
    θ_out[5] = 1.5
    clamped = clamp_interior(θ_out, b)
    @test clamped[5] < 1.0

    cfgb = bounds_from_config(Dict("spin1" => [-0.5, 0.5]))
    @test cfgb.lower[5] == -0.5 && cfgb.upper[5] == 0.5
    @test_throws ArgumentError bounds_from_config(Dict("spin1" => [1.0, -1.0]))
    # the phase is periodic: a finite interval for it is refused, the
    # unbounded pair is the only value the key accepts
    @test_throws "coalescence_phase" bounds_from_config(
        Dict("coalescence_phase" => [-1.0, 1.0]))
    @test_throws ArgumentError bounds_from_config(
        Dict("coalescence_phase" => [-Inf, 1.0]))
    @test bounds_from_config(Dict("coalescence_phase"=>[-Inf, Inf])).periodic[4]
end

@testset "Finite-observation window" begin
    W = CD.Physics.observation_window
    T, Δ = 3.0e7, 1.0e6
    @test W(0.0, T, Δ) ≈ 0.5 atol = 1e-12
    @test W(T, T, Δ) ≈ 0.5 atol = 1e-12
    @test W(T / 2, T, Δ) ≈ 1.0 atol = 1e-12
    @test W(-10Δ, T, Δ) < 1e-8 && W(T + 10Δ, T, Δ) < 1e-8
    @test W(T / 2 - 1.2e7, T, Δ) ≈ W(T / 2 + 1.2e7, T, Δ) rtol = 1e-12 # symmetric
    @test all(diff(W.(range(-5Δ, T / 2, 200), T, Δ)) .>= 0)             # monotone edge

    f = collect(1.0e-4:1.0e-5:5.0e-2)
    wp_off = WaveformParams(eta = 0.2222)
    wp_on = WaveformParams(eta = 0.2222, observation_time = 3.15576e7)
    @test wp_off.observation_time == 0 # absent by default
    # light system: τ(f_min) = 63 yr against t_c = 0.63 yr. The window removes
    # the band emitted before t = 0 and leaves the rest untouched
    seed = [2.0, 0.1, 20.0, 0.0, 0.7, 0.5]
    a_off = abs.(channel_strain(seed, f, wp_off)[1])
    a_on = abs.(channel_strain(seed, f, wp_on)[1])
    @test a_on[1] < 1e-12 * a_off[1]
    𝓜, t_c = 0.1, 2.0e7
    F0 = (5𝓜 / (256t_c))^(3 / 8) / (π * 𝓜) # τ(F0) = t_c for the (2,2) harmonic
    @test 5.5e-4 < F0 < 5.7e-4
    @test 0.35 < a_on[argmin(abs.(f .- F0))] / a_off[argmin(abs.(f .- F0))] < 0.65
    above = f .> 2.0e-3
    @test maximum(abs.(a_on[above] ./ a_off[above] .- 1)) < 1e-9
    # heavy system: the whole band is emitted inside the observation
    heavy = [5.0, 4.0, 20.0, 0.0, 0.8, 0.8]
    h_off = channel_strain(heavy, f, wp_off)[1]
    h_on = channel_strain(heavy, f, wp_on)[1]
    @test maximum(abs.(h_on .- h_off)) <= 1e-6 * maximum(abs.(h_off))
    # differentiable through the moving edge, and the kernel path agrees
    # with the scalar loop with the window on
    Sn = analytic_noise_psd.(f)
    data = channel_strain(seed, f, wp_on)
    loop = loss_function(data, f, Sn, 1.0e-5, wp_on, CPU())
    p = seed .+ [0.0, 1.0e-7, 1.0e-4, 1.0e-3, 1.0e-4, 0.0]
    g = ForwardDiff.gradient(loop, p)
    @test all(isfinite, g) && any(!iszero, g)
    kern = CD.Inference.device_loss(p, f, Sn, data[1], data[2], 1.0e-5, wp_on, CPU())
    @test kern ≈ loop(p) rtol = 1e-12

    mktempdir() do dir
        base = "[grid]\nT_obs = 1.0e7\nf_min = 1.0e-3\nf_max = 2.0e-3\n"
        plain = joinpath(dir, "plain.toml")
        write(plain, base)
        @test load_and_validate_config(plain).wp.observation_time == 0
        windowed = joinpath(dir, "windowed.toml")
        write(windowed, base * "[physics]\nobservation_window = true\n")
        cfg = load_and_validate_config(windowed)
        @test cfg.wp.observation_time == 1.0e7 && cfg.wp.window_edge_time == 1.0e6
        # the window changes what is computed: it belongs to the run identity
        @test run_id_from_config(windowed) != run_id_from_config(plain)
        write(windowed,
            base * "[physics]\nobservation_window = true\nwindow_edge_time = 2.0e6\n")
        @test_throws "window_edge_time" load_and_validate_config(windowed)
    end
end
