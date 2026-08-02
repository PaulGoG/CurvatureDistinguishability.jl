using CurvatureDistinguishability
using Test
using CSV
using DataFrames
using ForwardDiff
using KernelAbstractions
using Statistics
using TOML
using Logging

const CD = CurvatureDistinguishability
const FIXDIR = joinpath(@__DIR__, "fixtures", "reference")

# Fixture context: the tiny 201-bin grid the reference values were generated on.
const FIX_DF = 1e-5
const FIX_FREQS = collect(1e-3:FIX_DF:3e-3)
const FIX_PHYS = (mass_scale = 10.0, time_scale = 100.0, amp_scale = 1e-21, eta = 0.25,
                  amp_33_factor = 0.1, sky_theta = 1.047, sky_phi = 3.1415,
                  inclination = 0.523, polarization = 0.785)
const FIX_WP = waveform_params(; FIX_PHYS...)
const THETA0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
const FIX_NOISE_OFF = NoiseParams(confusion_enabled = false)
const FIX_SN = analytic_noise_psd.(FIX_FREQS; noise = FIX_NOISE_OFF)

@testset "CurvatureDistinguishability.jl" begin

    @testset "Noise PSD (Robson 2019)" begin
        # instrumental part must reproduce the reference values bitwise-tight
        psd_fix = CSV.read(joinpath(FIXDIR, "psd.csv"), DataFrame; header = false)
        @test all(isapprox.(FIX_SN, psd_fix[:, 2]; rtol = 1e-14))

        # Eq. 14 confusion term, checked against a direct transcription
        np = NoiseParams() # 1-yr defaults
        f = 1e-3
        sc = np.confusion_amp * f^(-7 / 3) *
             exp(-(f^np.confusion_alpha) + np.confusion_beta * f * sin(np.confusion_kappa * f)) *
             (1 + tanh(np.confusion_gamma * (np.confusion_fk - f)))
        @test analytic_noise_psd(f) ≈ analytic_noise_psd(f; noise = FIX_NOISE_OFF) + sc rtol = 1e-14
        @test sc > 0
        # the confusion bump must actually contribute in the mHz band now
        @test analytic_noise_psd(1e-3) > 2 * analytic_noise_psd(1e-3; noise = FIX_NOISE_OFF)

        # Table-1 selection by observation time
        @test robson_confusion_params(SECONDS_PER_YEAR).confusion_beta == 292.0
        @test robson_confusion_params(4 * SECONDS_PER_YEAR).confusion_beta == -221.0
        @test robson_confusion_params(0.4 * SECONDS_PER_YEAR).confusion_fk == 0.00258

        @test analytic_noise_psd(0.0) == 1e-30
        @test analytic_noise_psd(-1.0) == 1e-30

        # instrumental parameters (Eq. 10-12) are configurable; defaults are
        # bit-identical to the published model (fixture test above), and the
        # OMS term must respond quadratically where it dominates (high f)
        f_hi = 2e-2
        s_def = analytic_noise_psd(f_hi; noise = FIX_NOISE_OFF)
        s_oms = analytic_noise_psd(f_hi;
            noise = NoiseParams(confusion_enabled = false, oms_amplitude = 3.0e-11))
        @test s_oms > 3 * s_def
        @test analytic_noise_psd(1e-3;
            noise = NoiseParams(confusion_enabled = false, arm_length = 5.0e9)) != s_def
    end

    @testset "Waveform scalar core" begin
        h_bc = scaled_waveform_model(THETA0, FIX_FREQS, FIX_WP)
        A = THETA0[1] * FIX_WP.amp_scale
        Mc = THETA0[2] * FIX_WP.mass_scale
        tc = THETA0[3] * FIX_WP.time_scale
        beta = spin_beta(THETA0[5], THETA0[6], FIX_WP.eta)
        h_sc = [strain_bin(f, A, Mc, tc, THETA0[4], beta, FIX_WP.amp_33_factor) for f in FIX_FREQS]
        @test h_bc == h_sc
        @test eltype(h_bc) <: Complex
        # type stability of the hot scalar core
        @test (@inferred strain_bin(1e-3, A, Mc, tc, 0.0, beta, 0.1)) isa ComplexF64
    end

    @testset "Detector / TDI projection" begin
        h = scaled_waveform_model(THETA0, FIX_FREQS, FIX_WP)
        A2, E2 = project_to_tdi(h, FIX_FREQS, THETA0, FIX_WP)
        @test length(A2) == length(FIX_FREQS) == length(E2)

        # 3-channel variant (include_t_channel): identical A/E plus a zero T
        wp3 = waveform_params(; FIX_PHYS..., include_t_channel = true)
        A3, E3, T3 = project_to_tdi(h, FIX_FREQS, THETA0, wp3)
        @test A3 == A2 && E3 == E2
        @test all(iszero, T3)

        # fused projection ≡ per-bin modulation
        Mc = THETA0[2] * FIX_WP.mass_scale
        tc = THETA0[3] * FIX_WP.time_scale
        mA, mE = tdi_modulation_bin(FIX_FREQS[7], Mc, tc, FIX_WP)
        @test mA * h[7] ≈ A2[7] rtol = 1e-14
        @test mE * h[7] ≈ E2[7] rtol = 1e-14
        @test (@inferred tdi_modulation_bin(1e-3, Mc, tc, FIX_WP)) isa NTuple{2,ComplexF64}

        # channel count is a WaveformParams type parameter: the projection
        # return type (2- vs 3-tuple) and the flat response infer concretely
        @test n_channels(FIX_WP) == 2 && n_channels(wp3) == 3
        @test (@inferred project_to_tdi(h, FIX_FREQS, THETA0, FIX_WP)) isa
              NTuple{2,Vector{ComplexF64}}
        @test (@inferred project_to_tdi(h, FIX_FREQS, THETA0, wp3)) isa
              NTuple{3,Vector{ComplexF64}}
        @test (@inferred CD.Geometry.flat_response(THETA0, FIX_FREQS, FIX_WP)) isa
              Vector{Float64}
    end

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

        # fused nested-dual derivatives against independent ForwardDiff passes
        gvec(s) = [sin(2s) + s^3, exp(s) * cos(s)]
        h0, dh, d2h = CD.Geometry.value_and_directional_derivs(gvec, 0.3)
        @test h0 ≈ gvec(0.3) rtol = 1e-14
        @test dh ≈ ForwardDiff.derivative(gvec, 0.3) rtol = 1e-12
        @test d2h ≈ ForwardDiff.derivative(s -> ForwardDiff.derivative(gvec, s), 0.3) rtol = 1e-12
    end

    @testset "Geometry A/B against reference fixtures" begin
        # 1D sweep geometry
        sweep_fix = CSV.read(joinpath(FIXDIR, "sweep.csv"), DataFrame)
        u_raw = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
        K_u, g_uu = compute_extrinsic_curvature(THETA0, u_raw, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
        K_norm = K_u / g_uu^2
        @test K_norm ≈ sweep_fix.K_u_Norm[1] rtol = 1e-8
        @test all(isapprox.(sweep_fix.D2_Theoretical,
                            (1 / 16) .* K_norm .* sweep_fix.Delta .^ 4; rtol = 1e-8))

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
                if row.K_raw > 1e-10 * Kmax
                    @test K ≈ row.K_raw rtol = 1e-8
                end
                if row.G_uu > 1e-6
                    @test g ≈ row.G_uu rtol = 1e-6
                    # capped-mapping radius formula ≡ reference x/y bounds (s(φ) cancels)
                    r_math = (16.0 * 1.0 / K)^(1 / 4)
                    @test r_math ≈ hypot(row.X_Bound, row.Y_Bound) rtol = 1e-8
                end
            end
        end
    end

    @testset "Bounds and polar capping" begin
        b = default_bounds()
        box = deviation_box(b, THETA0, 5, 6) # spins at 0.8
        @test all(isapprox.(collect(box), [-1.8, 0.2, -1.8, 0.2]; rtol = 1e-12))
        box_p = deviation_box(b, THETA0, 2, 4) # mass (lower −1.5) and periodic phase
        @test box_p[1] ≈ -1.5 && box_p[2] == Inf
        @test box_p[3] ≈ -π && box_p[4] ≈ π

        @test ray_box_crossing(1.0, 0.0, -1.8, 0.2, -1.8, 0.2) ≈ 0.2
        @test ray_box_crossing(-1.0, 0.0, -1.8, 0.2, -1.8, 0.2) ≈ 1.8
        @test ray_box_crossing(cos(π / 4), sin(π / 4), -1.0, 1.0, -1.0, 0.5) ≈ 0.5 / sin(π / 4)
        @test ray_box_crossing(0.0, 1.0, -Inf, Inf, -Inf, Inf) == Inf

        # base point on a bound must be rejected
        θ_edge = copy(THETA0); θ_edge[5] = 1.0
        @test_throws ErrorException deviation_box(b, θ_edge, 5, 6)

        θ_out = copy(THETA0); θ_out[5] = 1.5
        clamped = clamp_interior(θ_out, b)
        @test clamped[5] < 1.0

        cfgb = bounds_from_config(Dict("spin1" => [-0.5, 0.5]))
        @test cfgb.lower[5] == -0.5 && cfgb.upper[5] == 0.5
        @test_throws ErrorException bounds_from_config(Dict("spin1" => [1.0, -1.0]))
    end

    @testset "Inference: kernel ≡ loop, optimizers, bounds" begin
        h1 = scaled_waveform_model(THETA0, FIX_FREQS, FIX_WP)
        θ2 = THETA0 .+ [0.0, 0.01, 0.02, 0.01, 0.0, 0.0]
        h2 = scaled_waveform_model(θ2, FIX_FREQS, FIX_WP)
        c1 = project_to_tdi(h1, FIX_FREQS, THETA0, FIX_WP)
        c2 = project_to_tdi(h2, FIX_FREQS, θ2, FIX_WP)
        data = map((a, b) -> a .+ b, c1, c2)

        loop = loss_function(data, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP, CPU())
        p = [2.0, 1.505, 2.01, 0.005, 0.8, 0.8]
        # the KA kernel on the CPU backend must reproduce the scalar loop,
        # for both the value and the ForwardDiff gradient
        kern = CD.Inference.device_loss(p, FIX_FREQS, FIX_SN, data[1], data[2],
                                         FIX_DF, FIX_WP, CPU())
        @test kern ≈ loop(p) rtol = 1e-12
        g_loop = ForwardDiff.gradient(loop, p)
        dl = q -> CD.Inference.device_loss(q, FIX_FREQS, FIX_SN, data[1], data[2],
                                            FIX_DF, FIX_WP, CPU())
        @test ForwardDiff.gradient(dl, p) ≈ g_loop rtol = 1e-12
        # lanes path: nested (Hessian) duals through the same kernel layout
        @test ForwardDiff.hessian(dl, p) ≈ ForwardDiff.hessian(loop, p) rtol = 1e-10
        # flatten/rebuild round-trip on a nested dual (lane-order contract);
        # the local reference flattening states the contract the kernel's
        # per-scalar stores and `rebuild_dual` must both follow: value first,
        # then partials 1..N, recursively
        flatten_dual(x::Float64) = (x,)
        flatten_dual(d::ForwardDiff.Dual) =
            (flatten_dual(ForwardDiff.value(d))...,
             (Tuple(Iterators.flatten(flatten_dual(p) for p in ForwardDiff.partials(d))))...)
        nd = ForwardDiff.Dual{:o}(ForwardDiff.Dual{:i}(1.0, 2.0, 3.0),
                                  ForwardDiff.Dual{:i}(4.0, 5.0, 6.0))
        fl = flatten_dual(nd)
        @test fl == (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        rb, _ = CD.Inference.rebuild_dual(typeof(nd), collect(fl), 1)
        @test rb === nd

        # perfect match → (near-)zero distance
        d0, bf0, _ = calculate_numerical_distance((c1[1], c1[2]), copy(THETA0),
                                                  FIX_FREQS, FIX_SN, FIX_DF;
                                                  iterations = 50, FIX_PHYS...)
        @test d0 < 1e-5
        # chunked-Hessian option must reproduce the full-chunk optimization
        d0c, _, _ = calculate_numerical_distance((c1[1], c1[2]), copy(THETA0),
                                                 FIX_FREQS, FIX_SN, FIX_DF;
                                                 iterations = 50, hessian_chunk = 2,
                                                 FIX_PHYS...)
        @test d0c < 1e-5

        # guardrail: a GPU backend with host Arrays must fail loudly, not crash
        # deep inside a kernel launch (get_best_backend() returns a GPU whenever
        # one is functional, so the mismatch is easy to hit from the REPL)
        struct FakeGPU <: KernelAbstractions.GPU end
        @test_throws ErrorException loss_function(data, FIX_FREQS, FIX_SN, FIX_DF,
                                                  FIX_WP, FakeGPU())

        # buffer-cache eviction API empties the cache and is safe to call
        # with no GPU present
        CD.Inference.clear_device_buffers!()
        @test isempty(CD.Inference.DEVICE_BUFFER_CACHE)

        # A/B against the committed reference fixtures on the clean separations
        sweep_fix = CSV.read(joinpath(FIXDIR, "sweep.csv"), DataFrame)
        u_raw = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
        K_u, g_uu = compute_extrinsic_curvature(THETA0, u_raw, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
        u_norm = u_raw ./ sqrt(g_uu)
        d2_ipn = Float64[]
        for row in eachrow(sweep_fix)[end-3:end]
            d = row.Delta
            p2 = THETA0 .+ d .* u_norm
            hh2 = scaled_waveform_model(p2, FIX_FREQS, FIX_WP)
            cc2 = project_to_tdi(hh2, FIX_FREQS, p2, FIX_WP)
            dstream = map((a, b) -> a .+ b, c1, cc2)
            guess = THETA0 .+ (0.5 * d) .* u_norm
            guess[1] *= 2.0
            d_new, bf, res = calculate_numerical_distance(dstream, guess, FIX_FREQS, FIX_SN, FIX_DF;
                                                          optimizer = :ipnewton, FIX_PHYS...)
            @test d_new ≈ row.D2_Numerical rtol = 1e-2
            @test all(abs.(bf[5:6]) .<= 1.0 + 1e-9) # physical bounds respected
            push!(d2_ipn, d_new)
        end
        # δ⁴ scaling of the fresh results on the clean range
        deltas4 = sweep_fix.Delta[end-3:end]
        slope = (log10(d2_ipn[end]) - log10(d2_ipn[1])) /
                (log10(deltas4[end]) - log10(deltas4[1]))
        @test isapprox(slope, 4.0; atol = 0.15)

        # bound activation: force the optimizer against the spin bound
        db = default_bounds()
        diag_guess = [2.0, 1.5, 2.0, 0.0, 0.99, 0.99]
        d_b, bf_b, res_b = calculate_numerical_distance(data, diag_guess, FIX_FREQS, FIX_SN,
                                                        FIX_DF; iterations = 100,
                                                        optimizer = :ipnewton, bounds = db,
                                                        FIX_PHYS...)
        @test all(bf_b .>= collect(db.lower) .- 1e-9)
        @test all(bf_b .<= collect(db.upper) .+ 1e-9)
        diag = optimization_diagnostics(res_b, bf_b, db)
        @test diag isa NamedTuple && haskey(pairs(diag), :converged)
    end

    @testset "Config validation guardrails" begin
        function write_cfg(dir, extra)
            path = joinpath(dir, "config.toml")
            base = """
            [grid]
            T_obs = 1.0e5
            f_min = 1.0e-3
            f_max = 2.0e-3
            """
            write(path, base * extra)
            return path
        end
        mktempdir() do dir
            cfg = load_and_validate_config(write_cfg(dir, ""))
            @test cfg isa PipelineSettings
            @test cfg.optimizer === :ipnewton
            # T_obs = 1e5 s → nearest Robson column is 6 months
            @test cfg.noise.confusion_fk == 0.00258

            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[pipeline]\noptimizer = \"sgd\"\n"))
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\nn_deltas = 1\n"))
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[parameter_bounds]\nspin1 = [2.0, 1.0]\n"))
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[[sweeps]]\nname = \"bad/name\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0,1,0,0,0,0]\n"))
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[[maps]]\nname = \"m\"\nparam_x = 2\nparam_y = 2\ntheta_0 = [1,1,1,0,0,0]\n"))
            # base point outside physical bounds
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[[maps]]\nname = \"m\"\nparam_x = 5\nparam_y = 6\ntheta_0 = [1,1,1,0,1.5,0]\n"))
            # unknown keys warn (typo protection), odd n_angles warns and rounds
            @test_logs (:warn, r"Unknown configuration key 'n_anglse'") match_mode = :any load_and_validate_config(
                write_cfg(dir, "[mapping]\nn_anglse = 100\n"))
            cfg_odd = @test_logs (:warn, r"odd; rounding up") match_mode = :any load_and_validate_config(
                write_cfg(dir, "[mapping]\nn_angles = 33\n"))
            @test cfg_odd.map_n_angles == 34
            @test cfg_odd.corner_bisect_iters == 25 # safe default: corners exact
            cfg_nb = load_and_validate_config(
                write_cfg(dir, "[mapping]\ncorner_bisect_iters = 0\n"))
            @test cfg_nb.corner_bisect_iters == 0 # explicit opt-out allowed
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[mapping]\ncorner_bisect_iters = -1\n"))

            # amp_ratio and multi-start guardrails
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[[sweeps]]\nname = \"s\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0,1,0,0,0,0]\namp_ratio = -1.0\n"))
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[[sweeps]]\nname = \"s\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0.5,1,0,0,0,0]\namp_ratio = 0.5\n"))
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\nn_starts = 0\n"))
            cfg_ms = load_and_validate_config(
                write_cfg(dir, "[pipeline]\nrng_seed = 7\n[pipeline.sweep_settings]\nn_starts = 3\ng_tol = 1e-9\nmax_iterations = 300\n[hardware]\nhessian_chunk = 3\n"))
            @test cfg_ms.n_starts == 3 && cfg_ms.rng_seed == 7
            @test cfg_ms.g_tol == 1e-9 && cfg_ms.max_iterations == 300
            @test cfg_ms.hessian_chunk == 3
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[hardware]\nhessian_chunk = 9\n"))
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\ng_tol = 0.0\n"))

            # safe-by-default optimizer tolerances; monitoring opt-in (off)
            cfg_def = load_and_validate_config(write_cfg(dir, ""))
            @test cfg_def.g_tol == 1e-10 && cfg_def.max_iterations == 100
            @test cfg_def.monitoring_enabled == false
            cfg_mon = load_and_validate_config(write_cfg(dir, "[monitoring]\nenabled = true\n"))
            @test cfg_mon.monitoring_enabled == true
            @test_logs (:warn, r"tighter than the numerical precision floor") match_mode = :any load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\ng_tol = 1e-12\n"))
            @test_logs (:warn, r"floor fits exhaust the full cap") match_mode = :any load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\nmax_iterations = 1000\n"))

            cfg_vs = load_and_validate_config(write_cfg(dir, "[safety]\nmax_vram_gb = 6.0\n"))
            @test cfg_vs.max_vram_gb == 6.0

            # instrumental-noise overrides thread through to NoiseParams and
            # must be strictly positive
            cfg_n = load_and_validate_config(
                write_cfg(dir, "[noise]\narm_length = 5.0e9\nacc_amplitude = 2.4e-15\n"))
            @test cfg_n.noise.arm_length == 5.0e9
            @test cfg_n.noise.acc_amplitude == 2.4e-15
            @test cfg_n.noise.oms_amplitude == 1.5e-11
            @test_throws ErrorException load_and_validate_config(
                write_cfg(dir, "[noise]\narm_length = 0.0\n"))
        end
    end

    @testset "Provenance: comment-invariant run identifiers" begin
        mktempdir() do dir
            a = joinpath(dir, "a.toml")
            b = joinpath(dir, "b.toml")
            c = joinpath(dir, "c.toml")
            write(a, "[grid]\nT_obs = 1.0e5\nf_min = 1.0e-3\n")
            # same values: reordered keys, comments, blank lines
            write(b, "# leading comment\n[grid]\n\nf_min = 1.0e-3 # trailing\nT_obs = 1.0e5\n")
            # one value changed
            write(c, "[grid]\nT_obs = 2.0e5\nf_min = 1.0e-3\n")
            @test run_id_from_config(a) == run_id_from_config(b)
            @test run_id_from_config(a) != run_id_from_config(c)
            @test startswith(run_id_from_config(a), "run_")
        end
    end

    @testset "Monitoring diagnostic panels" begin
        d = 10 .^ range(-3, -1, length = 8)
        th = d .^ 4
        num = th .* 1.01
        panel = CD.Orchestrator.sweep_diagnostic_panel(d, num, th, trues(8), 4.0, 0.01)
        @test occursin("fitted slope", panel) && occursin("8/8", panel)
        @test occursin("theory", panel)
        empty_panel = CD.Orchestrator.sweep_diagnostic_panel(d, zeros(8), th, falses(8), NaN, NaN)
        @test occursin("no positive", empty_panel)
        mp = CD.Orchestrator.map_diagnostic_panel(collect(range(0, 2π, length = 16)),
                                                   fill(1.0, 16), 0.25)
        @test occursin("prior-limited directions: 25.0%", mp)
    end

    @testset "Ratio-correction fit (O(δ⁵) quantification)" begin
        d = [0.01, 0.05, 0.1, 0.2, 0.3]
        r = 1.0 .+ 0.3 .* d .- 0.1 .* d .^ 2
        c1, c1_err, c2 = CD.Orchestrator.ratio_correction_fit(d, r)
        @test c1 ≈ 0.3 atol = 1e-6
        @test c2 ≈ -0.1 atol = 1e-6
        @test c1_err < 1e-10 # exact model → zero residual
        c1n, _, _ = CD.Orchestrator.ratio_correction_fit(d[1:2], r[1:2])
        @test isnan(c1n) # too few points
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

        # coverage: a ±0.78π range must be ticked out to ±3π/4, not stop at
        # ±π/2 (regression: sparse π ticks left the axis ends bare)
        vc, _ = CD.Plotting.pi_ticks(-0.78π, 0.78π)
        @test length(vc) >= 5
        @test maximum(vc) ≈ 3π / 4 atol = 1e-12
        @test minimum(vc) ≈ -3π / 4 atol = 1e-12

        @test CD.Plotting.axis_exponent(6e-4) == -4
        @test CD.Plotting.axis_exponent(2.0) == 0

        # residual-band ticks: from the true band [1e-4, 0.05] every whole-power
        # decade is present, including the low endpoint (regression: ticking
        # off the decimated spec.f range dropped the 1e-4 tick, leaving 2).
        vb, _ = CD.Plotting.decade_ticks(1e-4, 0.05)
        @test round.(Int, log10.(vb)) == [-4, -3, -2]

        # 10⁰ always renders as plain "1" — on decade ticks and the 1-2-5 series
        _, l0 = CD.Plotting.decade_ticks(0.5, 50.0)
        @test l0[1].s == "\$1\$"
        v125, l125 = CD.Plotting.log_ticks_125(5e-2, 6.0)
        @test any(≈(1.0), v125) && any(≈(2e-1), v125)
        @test l125[findfirst(≈(1.0), v125)].s == "\$1\$"
        @test !any(l -> occursin("10^{0}", l.s), l125)

        # offset ticks (confusion-map axes): one factored power of 10 with
        # integer mantissas preferring multiples of 5, limits snapped outward
        # so the frame ends exactly on the outermost labelled ticks
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

        # per-tick common-exponent scientific notation (fallback when no
        # clean integer-mantissa grid exists) — 5e-5 renders as 0.5×10⁻⁴,
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

        # fit coefficients: ALWAYS mantissa ×10ⁿ with two decimals
        @test CD.Plotting.coef_latex(0.001278) == "1.28\\times 10^{-3}"
        @test CD.Plotting.coef_latex(-0.0235) == "-2.35\\times 10^{-2}"
        @test CD.Plotting.coef_latex(1.5) == "1.50" # ×10⁰ factor omitted
    end

    @testset "End-to-end minimal pipeline" begin
        mktempdir() do dir
            cfg_path = joinpath(dir, "config.toml")
            write(cfg_path, """
            [pipeline]
            optimizer = "ipnewton"
            rng_seed = 11
            [pipeline.sweep_settings]
            n_deltas = 6
            min_log_delta = -3.0
            max_log_delta = -0.5
            n_starts = 2
            [grid]
            T_obs = 1.0e5
            f_min = 1.0e-3
            f_max = 3.0e-3
            [physics]
            time_scale = 100.0
            sky_phi = 3.1415
            polarization = 0.785
            [mapping]
            n_angles = 32
            max_refine_levels = 3
            [safety]
            max_ram_gb = 8.0
            [[sweeps]]
            name = "mini_sweep"
            theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
            u_dir = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
            [[sweeps]]
            name = "mini_unequal"
            theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
            u_dir = [0.0, 0.707, 0.5, 0.3, 0.3, -0.2]
            amp_ratio = 0.5
            [[maps]]
            name = "mini_spin_map"
            param_x = 5
            param_y = 6
            rho_thresh = 1.0
            theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
            [[maps]]
            name = "mini_mass_time"
            param_x = 2
            param_y = 3
            theta_0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
            """)
            out_base = run_pipeline(cfg_path, dir, "outputs")
            @test isdir(out_base)
            @test isfile(joinpath(out_base, "config.toml"))
            @test isfile(joinpath(out_base, "run.log"))
            meta = TOML.parsefile(joinpath(out_base, "metadata.toml"))
            @test haskey(meta, "git") && haskey(meta, "finished")
            @test meta["failed_stages"] == ""

            sdir = joinpath(out_base, "sweeps", "mini_sweep")
            res = CSV.read(joinpath(sdir, "results.csv"), DataFrame)
            @test nrow(res) == 6
            @test all(hasproperty.(Ref(res), [:BestFit_Spin1, :Converged, :Iterations, :AtBound]))
            @test all(res.D2_Numerical .>= 0)
            # multi-start bookkeeping (n_starts = 2): best-of never worse than canonical
            @test all(res.Starts .== 2)
            @test all(res.MultiStartGain .>= 1.0 - 1e-12)
            for f in ("scaling_plot.pdf", "scaling_plot.png", "residual_plot.pdf",
                      "residual_plot.png", "residual_spectrum.csv", "sweep_meta.toml")
                @test isfile(joinpath(sdir, f))
            end
            meta_s = TOML.parsefile(joinpath(sdir, "sweep_meta.toml"))
            @test haskey(meta_s, "c1") && haskey(meta_s, "delta_valid")

            # unequal-amplitude sweep: in the perturbative (small-δ) regime the
            # A_harm prefactor (2q/(1+q))² must make the ratio ≈ 1 — a missing
            # prefactor would show as ratio ≈ 2.25. (Only the small-δ rows: for
            # q ≠ 1 the midpoint symmetry that suppresses the odd O(δ⁵) term is
            # absent, so the higher-order departure sets in much earlier.)
            res_u = CSV.read(joinpath(out_base, "sweeps", "mini_unequal", "results.csv"), DataFrame)
            ratio_u = res_u.D2_Numerical[1:3] ./ res_u.D2_Theoretical[1:3]
            @test all(0.85 .< ratio_u .< 1.15)
            meta_u = TOML.parsefile(joinpath(out_base, "sweeps", "mini_unequal", "sweep_meta.toml"))
            @test meta_u["amp_ratio"] == 0.5

            for map_name in ("mini_spin_map", "mini_mass_time")
                mdir = joinpath(out_base, "maps", map_name)
                cm = CSV.read(joinpath(mdir, "confusion_contour.csv"), DataFrame)
                n = nrow(cm)
                @test iseven(n)
                half = n ÷ 2
                # the MATHEMATICAL radius is exactly centrally symmetric by
                # construction (mirrored evaluation); the capped radius need
                # not be — the prior box may be asymmetric about the base point
                @test cm.R_Math[1:half] == cm.R_Math[half+1:end]
                @test cm.K_Raw[1:half] == cm.K_Raw[half+1:end]
                @test all(cm.Angle[half+1:end] .≈ cm.Angle[1:half] .+ π)
                @test isfile(joinpath(mdir, "confusion_zone.pdf"))
                @test isfile(joinpath(mdir, "confusion_zone.png"))
                # exact prior-crossover corners: at every capped/uncapped
                # transition along the boundary polygon, one of the two rows
                # must be the bisection-inserted vertex with r_math ≈ r_box —
                # otherwise the polygon chord chamfers the zone corner
                n_trans = 0
                for i in 1:n
                    j = mod1(i + 1, n)
                    cm.Prior_Limited[i] == cm.Prior_Limited[j] && continue
                    n_trans += 1
                    rel = [abs(cm.R_Math[k] - cm.R_Box[k]) / cm.R_Box[k]
                           for k in (i, j) if isfinite(cm.R_Box[k]) && isfinite(cm.R_Math[k])]
                    @test any(<(1e-3), rel)
                end
                count(cm.Prior_Limited) in (0, n) || @test n_trans >= 2
            end

            # exact box-corner vertex: both walls of the mass-time box are
            # active, so the boundary must pass through their corner exactly
            cmt = CSV.read(joinpath(out_base, "maps", "mini_mass_time", "confusion_contour.csv"),
                           DataFrame)
            @test any((abs.(cmt.X_Bound .+ 1.5) .< 1e-9) .& (abs.(cmt.Y_Bound .+ 2.0) .< 1e-9))

            # the spin map must be capped inside the physical box
            cm = CSV.read(joinpath(out_base, "maps", "mini_spin_map", "confusion_contour.csv"),
                          DataFrame)
            @test all(-1.8 - 1e-9 .<= cm.X_Bound .<= 0.2 + 1e-9)
            @test all(-1.8 - 1e-9 .<= cm.Y_Bound .<= 0.2 + 1e-9)
            @test count(cm.Prior_Limited) > 0 # χ_a degeneracy → prior-limited directions
            @test all(cm.R_Capped .<= cm.R_Box .+ 1e-12)
            @test hasproperty(cm, :Dir_Cos) && hasproperty(cm, :Dir_Sin)

            # threshold-rescaled replot from persisted K (no recomputation).
            # Plain binary (not julia_cmd(): --check-bounds would recompile the
            # plotting stack) and a scrubbed environment (Pkg.test exports its
            # sandbox via JULIA_LOAD_PATH/JULIA_PROJECT, which would poison the
            # child's package resolution).
            replot_script = joinpath(dirname(@__DIR__), "scripts", "replot.jl")
            jlbin = Base.julia_cmd().exec[1]
            cmd = addenv(`$jlbin --startup-file=no $replot_script $out_base --rho 2.0`,
                         "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
            run(pipeline(cmd, stdout = devnull, stderr = devnull))
            rcsv = joinpath(out_base, "maps", "mini_spin_map", "confusion_contour_rho2p0.csv")
            @test isfile(rcsv)
            cm2 = CSV.read(rcsv, DataFrame)
            @test all(cm2.R_Capped .<= cm2.R_Box .+ 1e-12)
            # where neither threshold is box-capped, radii scale exactly by √2
            free = .!cm.Prior_Limited .& .!cm2.Prior_Limited
            if any(free)
                @test all(isapprox.(cm2.R_Capped[free], sqrt(2.0) .* cm.R_Capped[free]; rtol = 1e-10))
            end
            @test isfile(joinpath(out_base, "maps", "mini_spin_map", "confusion_zone_rho2p0.png"))

            # unified regeneration API (RunFigures): figures rebuild from the
            # persisted CSVs alone, and the library rho-rescale reproduces the
            # replot child's persisted contour exactly
            cases = run_cases(out_base)
            @test cases.sweeps == ["mini_sweep", "mini_unequal"]
            @test cases.maps == ["mini_mass_time", "mini_spin_map"]
            figs = sweep_figures(out_base, "mini_sweep")
            @test figs.residual !== nothing && figs.suffix == ""
            figs_refit = sweep_figures(out_base, "mini_sweep"; refit = true)
            @test figs_refit.suffix == ""
            rz = zone_map_figure(out_base, "mini_spin_map"; rho = 2.0)
            @test rz.suffix == "_rho2p0" && rz.contour isa DataFrame
            @test all(rz.contour.R_Capped .<= rz.contour.R_Box .+ 1e-12)
            @test all(isapprox.(rz.contour.R_Capped, cm2.R_Capped; rtol = 1e-12))

            # collect_plots subprocess: flat browsing PNGs from a run *path*
            # selector plus a destination argument (same env scrubbing as the
            # replot child)
            collect_script = joinpath(dirname(@__DIR__), "scripts", "collect_plots.jl")
            browse_dir = joinpath(dir, "browse")
            cmd_collect = addenv(`$jlbin --startup-file=no $collect_script $browse_dir $out_base`,
                                 "JULIA_LOAD_PATH" => nothing, "JULIA_PROJECT" => nothing)
            run(pipeline(cmd_collect, stdout = devnull, stderr = devnull))
            run_label = basename(out_base)
            for png in ("$(run_label)_sweep_mini_sweep_scaling.png",
                        "$(run_label)_sweep_mini_sweep_residual.png",
                        "$(run_label)_map_mini_spin_map.png")
                @test isfile(joinpath(browse_dir, png))
            end

            # rerun with the same config must NOT overwrite: suffixed run dir
            out2 = run_pipeline(cfg_path, dir, "outputs")
            @test out2 != out_base && isdir(out2)
        end
    end

end
