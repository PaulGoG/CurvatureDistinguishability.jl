# Loss kernel, optimizers and bounds. Included by runtests.jl.

@testset "Inference: kernel ≡ loop, optimizers, bounds" begin
    θ2 = THETA0 .+ [0.0, 0.01, 0.02, 0.01, 0.0, 0.0]
    c1 = channel_strain(THETA0, FIX_FREQS, FIX_WP)
    c2 = channel_strain(θ2, FIX_FREQS, FIX_WP)
    data = map((a, b) -> a .+ b, c1, c2)

    loop = loss_function(data, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP, CPU())
    p = [2.0, 1.505, 2.01, 0.005, 0.8, 0.8]
    # the KA kernel on the CPU backend must reproduce the scalar loop,
    # for both the value and the ForwardDiff gradient
    kern = CD.Inference.device_loss(p, FIX_FREQS, FIX_SN, data[1], data[2],
        FIX_DF, FIX_WP, CPU())
    @test kern ≈ loop(p) rtol = 1e-12
    g_loop = ForwardDiff.gradient(loop, p)
    dl =
        q -> CD.Inference.device_loss(q, FIX_FREQS, FIX_SN, data[1], data[2],
            FIX_DF, FIX_WP, CPU())
    @test ForwardDiff.gradient(dl, p) ≈ g_loop rtol = 1e-12
    # the same at 2PN order: the kernel reads the order from its waveform
    # parameters
    wp2 = waveform_params(; FIX_PHYS..., phase_pn_order = 2)
    data2 = map((a, b) -> a .+ b, channel_strain(THETA0, FIX_FREQS, wp2),
        channel_strain(θ2, FIX_FREQS, wp2))
    loop2 = loss_function(data2, FIX_FREQS, FIX_SN, FIX_DF, wp2, CPU())
    dl2 =
        q -> CD.Inference.device_loss(q, FIX_FREQS, FIX_SN, data2[1], data2[2],
            FIX_DF, wp2, CPU())
    @test dl2(p) ≈ loop2(p) rtol = 1e-12
    @test ForwardDiff.gradient(dl2, p) ≈ ForwardDiff.gradient(loop2, p) rtol = 1e-12
    @test loop2(p) != loop(p)
    # and at the highest phase order, whose coefficients carry logarithms
    wp45 = waveform_params(; merge(FIX_PHYS, (; eta = 2 / 9, phase_pn_order = 4.5))...)
    data45 = map((a, b) -> a .+ b, channel_strain(THETA0, FIX_FREQS, wp45),
        channel_strain(θ2, FIX_FREQS, wp45))
    loop45 = loss_function(data45, FIX_FREQS, FIX_SN, FIX_DF, wp45, CPU())
    dl45 =
        q -> CD.Inference.device_loss(q, FIX_FREQS, FIX_SN, data45[1], data45[2],
            FIX_DF, wp45, CPU())
    @test dl45(p) ≈ loop45(p) rtol = 1e-12
    @test ForwardDiff.gradient(dl45, p) ≈ ForwardDiff.gradient(loop45, p) rtol = 1e-10
    # nested duals at 4.5PN with χ_s χ_a ≠ 0, where the mixed spin terms enter
    p_mixed = [2.0, 1.505, 2.01, 0.005, 0.7, -0.2]
    @test ForwardDiff.hessian(dl45, p_mixed) ≈ ForwardDiff.hessian(loop45, p_mixed) rtol =
        1e-10
    # lanes path: nested (Hessian) duals through the same kernel layout
    @test ForwardDiff.hessian(dl, p) ≈ ForwardDiff.hessian(loop, p) rtol = 1e-10
    # chunked Hessians through the lanes layout equal the full 49-lane one on
    # a non-zero residual (two-source data, off the optimum): every chunk
    # size, including those that leave a remainder chunk (4, 5)
    H_full = ForwardDiff.hessian(dl, p)
    @test maximum(abs, H_full) > 1.0
    for c in 1:6
        chunked = ForwardDiff.HessianConfig(dl, p, ForwardDiff.Chunk(c))
        @test ForwardDiff.hessian(dl, p, chunked) ≈ H_full rtol = 1e-10
    end
    # flatten/rebuild round-trip on a nested dual (lane-order contract);
    # the local reference flattening states the contract the kernel's
    # per-scalar stores and `rebuild_dual` must both follow: value first,
    # then partials 1..N, recursively
    flatten_dual(x::Float64) = (x,)
    flatten_dual(d::ForwardDiff.Dual) =
        (flatten_dual(ForwardDiff.value(d))...,
            (Tuple(
                Iterators.flatten(flatten_dual(p) for p in ForwardDiff.partials(d)),
            ))...)
    nd = ForwardDiff.Dual{:o}(ForwardDiff.Dual{:i}(1.0, 2.0, 3.0),
        ForwardDiff.Dual{:i}(4.0, 5.0, 6.0))
    fl = flatten_dual(nd)
    @test fl == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    rb, _ = CD.Inference.rebuild_dual(typeof(nd), collect(fl), 1)
    @test rb === nd

    # perfect match → (near-)zero distance
    d0, bf0, _ = calculate_numerical_distance((c1[1], c1[2]), copy(THETA0),
        FIX_FREQS, FIX_SN, FIX_DF;
        iterations = 50, wp = FIX_WP)
    @test d0 < 1e-5
    # chunked-Hessian option must reproduce the full-chunk optimization
    d0c, _, _ = calculate_numerical_distance((c1[1], c1[2]), copy(THETA0),
        FIX_FREQS, FIX_SN, FIX_DF;
        iterations = 50, hessian_chunk = 2,
        wp = FIX_WP)
    @test d0c < 1e-5

    # the gradient-only fallback optimizer must find the same optimum
    # (much higher floor than IPNewton, hence the loose threshold)
    d0l, bfl, _ = calculate_numerical_distance((c1[1], c1[2]), copy(THETA0),
        FIX_FREQS, FIX_SN, FIX_DF;
        optimizer = :lbfgs_box, iterations = 200,
        wp = FIX_WP)
    @test d0l < 1e-2
    @test all(isfinite, bfl)

    # guardrail: a GPU backend with host Arrays must fail loudly, not crash
    # deep inside a kernel launch (get_best_backend() returns a GPU whenever
    # one is functional, so the mismatch is easy to hit from the REPL).
    # A bare Backend subtype: KernelAbstractions.GPU is deprecated in 0.10.
    struct FakeGPU <: KernelAbstractions.Backend end
    @test_throws ArgumentError loss_function(data, FIX_FREQS, FIX_SN, FIX_DF,
        FIX_WP, FakeGPU())

    # inconsistent arguments are refused at the interface, not read out of
    # bounds under @inbounds
    @test_throws DimensionMismatch loss_function(data, FIX_FREQS, FIX_SN[1:(end-1)],
        FIX_DF, FIX_WP, CPU())
    @test_throws DimensionMismatch loss_function((data[1], data[2][1:(end-1)]),
        FIX_FREQS, FIX_SN, FIX_DF, FIX_WP, CPU())
    @test_throws ArgumentError loss_function(data, FIX_FREQS, FIX_SN, 0.0, FIX_WP,
        CPU())
    fit(; θ = copy(THETA0), kwargs...) = calculate_numerical_distance(
        (c1[1], c1[2]), θ, FIX_FREQS, FIX_SN, FIX_DF; wp = FIX_WP, backend = CPU(),
        kwargs...)
    @test_throws DimensionMismatch fit(; θ = THETA0[1:5])
    @test_throws ArgumentError fit(; hessian_chunk = -1)
    @test_throws ArgumentError fit(; iterations = 0)
    @test_throws ArgumentError fit(; g_tol = 0.0)
    h3 = ComplexF64[1, 2, 3]
    @test inner_product(h3, h3, ones(3), 0.5) ≈ 4 * 0.5 * 14
    @test_throws DimensionMismatch inner_product(h3, h3[1:2], ones(3), 1.0)
    @test_throws DimensionMismatch inner_product(h3, h3, ones(2), 1.0)
    @test_throws DimensionMismatch multi_channel_inner_product((h3, h3), (h3,),
        ones(3), 1.0)
    @test boundary_radius(16.0, 1.0) ≈ 1.0 && boundary_radius(0.0, 1.0) == Inf
    @test_throws DomainError boundary_radius(NaN, 1.0)
    @test_throws DomainError boundary_radius(1.0, 0.0)
    # a named backend that is not available is an error; :auto falls back
    @test get_best_backend(prefer = :none) isa CPU
    @test_throws ArgumentError get_best_backend(prefer = :no_such_backend)
    get_best_backend(prefer = :auto) isa CPU &&
        @test_throws ArgumentError get_best_backend(prefer = :cuda)

    # buffer-cache eviction API empties the cache and is safe to call
    # with no GPU present
    CD.Inference.clear_device_buffers!()
    @test isempty(CD.Inference.DEVICE_BUFFER_CACHE)

    # A/B against the committed reference fixtures on the clean separations
    sweep_fix = CSV.read(joinpath(FIXDIR, "sweep.csv"), DataFrame)
    u_raw = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
    K_u, g_uu =
        compute_extrinsic_curvature(THETA0, u_raw, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
    u_norm = u_raw ./ sqrt(g_uu)
    d2_ipn = Float64[]
    for row in eachrow(sweep_fix)[(end-3):end]
        d = row.Delta
        p2 = THETA0 .+ d .* u_norm
        cc2 = channel_strain(p2, FIX_FREQS, FIX_WP)
        dstream = map((a, b) -> a .+ b, c1, cc2)
        guess = THETA0 .+ (0.5 * d) .* u_norm
        guess[1] /= 2.0 # two equal sources: amplitude 2A ⇔ distance D/2
        d_new, bf, res =
            calculate_numerical_distance(dstream, guess, FIX_FREQS, FIX_SN, FIX_DF;
                optimizer = :ipnewton, wp = FIX_WP)
        @test d_new ≈ row.D2_Numerical rtol = 1e-2
        # cross-method check of the law: the optimized distance against
        # the curvature prediction K δ⁴/16 of the geometry stage
        @test d_new ≈ (K_u / g_uu^2) * d^4 / 16 rtol = 2e-3
        @test all(abs.(bf[5:6]) .<= 1.0 + 1e-9) # physical bounds respected
        push!(d2_ipn, d_new)
    end
    # the fit in whitened coordinates reaches the same optimum as the fit in the
    # parameters where the latter is well conditioned (1.5PN, equal mass), and the
    # curvature prediction where the latter stalls (2PN, η = 2/9)
    for row in eachrow(sweep_fix)[(end-3):end]
        d = row.Delta
        p2 = THETA0 .+ d .* u_norm
        dstream = map((a, b) -> a .+ b, c1, channel_strain(p2, FIX_FREQS, FIX_WP))
        guess = THETA0 .+ (0.5 * d) .* u_norm
        guess[1] /= 2.0
        d_w, bf_w, res_w = calculate_numerical_distance(dstream, guess, FIX_FREQS,
            FIX_SN, FIX_DF; optimizer = :ipnewton, coordinates = :whitened, wp = FIX_WP)
        @test d_w ≈ (K_u / g_uu^2) * d^4 / 16 rtol = 2e-3
        @test CD.Inference.Optim.converged(res_w)
        @test all(abs.(bf_w[5:6]) .<= 1.0 + 1e-9)
    end
    let wp2 = waveform_params(; merge(FIX_PHYS, (; eta = 2 / 9, phase_pn_order = 2))...)
        θb = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
        u2 = [0.0, 1.0, 0.5, 0.0, 0.0, 0.0]
        K2, g2 = compute_extrinsic_curvature(θb, u2, FIX_FREQS, FIX_SN, FIX_DF, wp2)
        un2 = u2 ./ sqrt(g2)
        # well inside δ_min, where the O(δ⁵) departure stays below the tolerance
        for d in (0.1 * (16 / (K2 / g2^2))^(1 / 4), 0.3 * (16 / (K2 / g2^2))^(1 / 4))
            dstream = map((a, b) -> a .+ b, channel_strain(θb, FIX_FREQS, wp2),
                channel_strain(θb .+ d .* un2, FIX_FREQS, wp2))
            guess = θb .+ (0.5 * d) .* un2
            guess[1] /= 2.0
            d_w, _, res_w = calculate_numerical_distance(dstream, guess, FIX_FREQS, FIX_SN,
                FIX_DF; optimizer = :ipnewton, coordinates = :whitened, wp = wp2)
            @test d_w ≈ (K2 / g2^2) * d^4 / 16 rtol = 2e-3
            @test CD.Inference.Optim.converged(res_w)
            # the chunked Hessian of the GPU backends gives the same fit in z
            d_c, _, res_c = calculate_numerical_distance(dstream, guess, FIX_FREQS,
                FIX_SN, FIX_DF; optimizer = :ipnewton, coordinates = :whitened,
                wp = wp2, hessian_chunk = 2)
            @test d_c ≈ d_w rtol = 1e-8
            @test CD.Inference.Optim.converged(res_c)
        end
        # and the same Hessian in z, away from the origin of z
        stream = map((a, b) -> a .+ b, channel_strain(θb, FIX_FREQS, wp2),
            channel_strain(θb .+ 0.3 .* un2, FIX_FREQS, wp2))
        loss = loss_function(stream, FIX_FREQS, FIX_SN, FIX_DF, wp2, CPU())
        T = CD.Inference.whitening_transform(
            CD.Geometry.whitened_jacobian_factor(θb, FIX_FREQS, FIX_SN, FIX_DF, wp2))
        loss_z = z -> loss(θb .+ T * z)
        z = [0.3, -0.2, 0.1, 0.05, -0.1, 0.2]
        H_full, H_chunk = zeros(6, 6), zeros(6, 6)
        CD.Inference.derivative_closures(loss_z, zeros(6), 0)[2](H_full, z)
        CD.Inference.derivative_closures(loss_z, zeros(6), 2)[2](H_chunk, z)
        @test maximum(abs, H_full) > 1.0
        @test H_chunk ≈ H_full rtol = 1e-12
    end
    @test_throws ArgumentError calculate_numerical_distance(data, THETA0, FIX_FREQS, FIX_SN,
        FIX_DF; optimizer = :lbfgs_box, coordinates = :whitened, wp = FIX_WP)
    @test_throws ArgumentError calculate_numerical_distance(data, THETA0, FIX_FREQS, FIX_SN,
        FIX_DF; coordinates = :fisher, wp = FIX_WP)
    # δ⁴ scaling of the fresh results on the clean range
    deltas4 = sweep_fix.Delta[(end-3):end]
    slope =
        (log10(d2_ipn[end]) - log10(d2_ipn[1])) /
        (log10(deltas4[end]) - log10(deltas4[1]))
    @test isapprox(slope, 4.0; atol = 0.15)

    # bound activation: force the optimizer against the spin bound
    db = default_bounds()
    diag_guess = [2.0, 1.5, 2.0, 0.0, 0.99, 0.99]
    d_b, bf_b, res_b =
        calculate_numerical_distance(data, diag_guess, FIX_FREQS, FIX_SN,
            FIX_DF; iterations = 100,
            optimizer = :ipnewton, bounds = db,
            wp = FIX_WP)
    @test all(bf_b .>= collect(db.lower) .- 1e-9)
    @test all(bf_b .<= collect(db.upper) .+ 1e-9)
    diag = optimization_diagnostics(res_b, bf_b, db)
    @test diag isa NamedTuple && haskey(pairs(diag), :converged)
    # an optimum on the spin bound, reached from the midpoint start of the
    # sweeps: in whitened coordinates the box is a set of linear constraints,
    # and both fits end on the bound with the same distance
    θa = [1.0, 1.5, 2.0, 0.0, 0.995, 0.99]
    θb = θa .+ [0.0, 1e-4, 2e-3, 0.0, 0.02, 0.02]
    data_b = map((a, b) -> a .+ b, channel_strain(θa, FIX_FREQS, FIX_WP),
        channel_strain(θb, FIX_FREQS, FIX_WP))
    start = clamp.(0.5 .* (θa .+ θb), collect(db.lower), collect(db.upper))
    start[1] /= 2
    fits = map((:parameters, :whitened)) do coordinates
        calculate_numerical_distance(data_b, start, FIX_FREQS, FIX_SN, FIX_DF;
            iterations = 200, optimizer = :ipnewton, bounds = db, wp = FIX_WP,
            coordinates)
    end
    for (d, bf, res) in fits
        @test CD.Inference.Optim.converged(res)
        @test optimization_diagnostics(res, bf, db).at_bound
        @test all(collect(db.lower) .- 1e-9 .<= bf .<= collect(db.upper) .+ 1e-9)
    end
    @test fits[2][1] ≈ fits[1][1] rtol = 1e-9
    @test fits[2][2] ≈ fits[1][2] rtol = 1e-6
end
