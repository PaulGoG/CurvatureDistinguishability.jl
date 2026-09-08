using CurvatureDistinguishability
using Test
using Aqua
using ExplicitImports
using JET
using CSV
using DataFrames
using ForwardDiff
using KernelAbstractions
using TOML

const CD = CurvatureDistinguishability
const FIXDIR = joinpath(@__DIR__, "fixtures", "reference")

# Fixture context: the tiny 201-bin grid the reference values were generated on.
const FIX_DF = 1e-5
const FIX_FREQS = collect(1e-3:FIX_DF:3e-3)
const FIX_PHYS = (mass_scale = 1.0, time_scale = 100.0, eta = 0.25,
    ecliptic_longitude = 3.1415, ecliptic_latitude = 0.5238,
    inclination = 0.523, polarization = 0.785)
const FIX_WP = waveform_params(; FIX_PHYS...)
const THETA0 = [1.0, 1.5, 2.0, 0.0, 0.8, 0.8]
const FIX_NOISE_OFF = NoiseParams(confusion_enabled = false)
const FIX_SN = analytic_noise_psd.(FIX_FREQS; noise = FIX_NOISE_OFF)

@testset "CurvatureDistinguishability.jl" begin

    @testset "Static QA" begin
        # the persistent-tasks probe spawns a child precompilation whose
        # completion marker is flaky on cold macOS/Windows runners and under
        # coverage instrumentation (the instrumented child precompile of the
        # full dependency stack outruns the probe); the verdict is
        # platform-independent, so the probe is enforced on uninstrumented
        # Linux — that is, every local Pkg.test()
        run_probe = Sys.islinux() && Base.JLOptions().code_coverage == 0
        Aqua.test_all(CurvatureDistinguishability; persistent_tasks = run_probe)
        # the qualified-access publicity check is deliberately not enforced:
        # ForwardDiff and Optim expose their documented API (Dual, value,
        # partials, minimizer, …) without `public` annotations
        @test ExplicitImports.check_no_implicit_imports(CurvatureDistinguishability) ===
              nothing
        @test ExplicitImports.check_no_stale_explicit_imports(
            CurvatureDistinguishability,
        ) === nothing
        # the top-module check does not recurse: stale aliases in submodules
        # were invisible to CI until checked one by one
        for submodule in (CD.Physics, CD.Detector, CD.Residuals, CD.Bounds,
            CD.Geometry, CD.Fitting, CD.Inference, CD.Backends, CD.Config,
            CD.Provenance, CD.Plotting, CD.Orchestrator, CD.RunFigures)
            @test ExplicitImports.check_no_stale_explicit_imports(submodule) ===
                  nothing
        end
        @test ExplicitImports.check_all_explicit_imports_via_owners(
            CurvatureDistinguishability,
        ) === nothing

        # JET package analysis with error points restricted to this package's
        # modules (upstream abstract-interpretation artifacts in
        # Base/Optim/Makie internals are out of scope). Exactly three known
        # artifacts remain, all anchored at the deliberately untyped
        # device-buffer barrier (launch_loss!/launch_loss_lanes!/
        # column_sums_via!, one per kernel: loss_bins!, loss_bins_lanes!,
        # partial_column_sums!): the GPU kernel-call method exists only once
        # a GPU backend package is loaded, so the GPU branch of the backend
        # union split reports as missing here while being unreachable by
        # construction (get_best_backend returns only registered backends).
        # A change in this count — either direction — must be triaged.
        # JET tracks compiler internals and routinely breaks on pre-release
        # Julia (observed: internal UndefRefError on the CI `pre` leg), so
        # the analysis gates stable releases only
        if isempty(VERSION.prerelease)
            jet_modules = (CD, CD.Physics, CD.Detector, CD.Residuals, CD.Bounds,
                CD.Geometry, CD.Fitting, CD.Inference, CD.Backends, CD.Config,
                CD.Provenance, CD.Plotting, CD.Orchestrator, CD.RunFigures)
            jet = JET.report_package(CD; target_modules = jet_modules,
                toplevel_logger = nothing)
            @test length(JET.get_reports(jet)) == 3
        end
    end

    @testset "Noise PSD (Robson 2019)" begin
        # instrumental part must reproduce the reference values bitwise-tight
        psd_fix = CSV.read(joinpath(FIXDIR, "psd.csv"), DataFrame; header = false)
        @test all(isapprox.(FIX_SN, psd_fix[:, 2]; rtol = 1e-14))

        # Eq. 14 confusion term, checked against a direct transcription
        np = NoiseParams() # 1-yr defaults
        f = 1e-3
        sc =
            np.confusion_amp * f^(-7 / 3) *
            exp(
                -(f^np.confusion_alpha) +
                np.confusion_beta * f * sin(np.confusion_kappa * f),
            ) *
            (1 + tanh(np.confusion_gamma * (np.confusion_knee_freq - f)))
        @test analytic_noise_psd(f) ≈ analytic_noise_psd(f; noise = FIX_NOISE_OFF) + sc rtol =
            1e-14
        @test sc > 0
        # the confusion bump must actually contribute in the mHz band now
        @test analytic_noise_psd(1e-3) > 2 * analytic_noise_psd(1e-3; noise = FIX_NOISE_OFF)

        # Table-1 selection by observation time
        @test robson_confusion_params(SECONDS_PER_YEAR).confusion_beta == 292.0
        @test robson_confusion_params(4 * SECONDS_PER_YEAR).confusion_beta == -221.0
        @test robson_confusion_params(0.4 * SECONDS_PER_YEAR).confusion_knee_freq == 0.00258

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

    @testset "Waveform physics: phasing, amplitudes, taper" begin
        Mc, D = 1.5, CD.Physics.GIGAPARSEC_SEC
        # the leading amplitude is the standard stationary-phase form
        # √(5/24) π^{-2/3} 𝓜^{5/6} f^{-7/6} (1 + c²)/(2D) at every inclination
        for ι in (0.0, 0.523, π / 2)
            a_plus, a_cross = harmonic_amplitudes(2e-3, 2, Mc, 0.25, ι, D)
            closed =
                sqrt(5 / 24) * π^(-2 / 3) * Mc^(5 / 6) * (2e-3)^(-7 / 6) *
                (1 + cos(ι)^2) / (2 * D)
            @test abs(a_plus) ≈ closed rtol = 1e-12
            @test a_cross ≈ a_plus * 2 * cos(ι) / (1 + cos(ι)^2) rtol = 1e-12
        end
        # the 0.5PN harmonics vanish at equal mass and carry the Blanchet
        # coefficients times the mass asymmetry otherwise
        @test all(iszero, harmonic_amplitudes(2e-3, 1, Mc, 0.25, 0.5, D))
        @test all(iszero, harmonic_amplitudes(2e-3, 3, Mc, 0.25, 0.5, D))
        @test CD.Physics.mass_asymmetry(0.16) ≈ 0.6 rtol = 1e-14
        @test CD.Physics.mass_asymmetry(0.25) == 0.0
        c, s_i = cos(0.5), sin(0.5)
        for (k, plus_coef, cross_coef) in ((1, -(s_i / 8) * (5 + c^2), -(3 / 4) * s_i * c),
            (3, (9 * s_i / 8) * (1 + c^2), (9 / 4) * s_i * c))
            F = 2 * 2e-3 / k
            M = total_mass(Mc, 0.16)
            x = (π * M * F)^(2 / 3)
            chirp_rate = (96 / 5) * π^(8 / 3) * Mc^(5 / 3) * F^(11 / 3)
            pref = (M * 0.16 * x / D) * sqrt(2 / (k * chirp_rate)) * sqrt(x) * 0.6
            a_plus, a_cross = harmonic_amplitudes(2e-3, k, Mc, 0.16, 0.5, D)
            @test a_plus ≈ plus_coef * pref rtol = 1e-12
            @test a_cross ≈ cross_coef * pref rtol = 1e-12
        end
        # TaylorF2 phasing to 1.5PN in the total-mass velocity
        for eta in (0.25, 0.16)
            beta = spin_beta(0.7, -0.2, eta)
            v = (π * total_mass(Mc, eta) * 3e-3)^(1 / 3)
            expected =
                3 / (128 * eta * v^5) *
                (1 + (3715 / 756 + 55 * eta / 9) * v^2 + (4 * beta - 16 * π) * v^3)
            @test pn_phase(3e-3, Mc, eta, beta) ≈ expected rtol = 1e-14
        end
        # Poisson–Will spin–orbit coefficient: symmetric part at equal mass,
        # antisymmetric part proportional to the mass asymmetry
        @test spin_beta(0.5, 0.3, 0.25) ≈ 3.1333333333333333 rtol = 1e-14
        @test spin_beta(1.0, 0.0, 0.16) ≈ ((113 - 76 * 0.16) * 0.5 + 113 * 0.6 * 0.5) / 12 rtol =
            1e-14
        # the spins enter the 1.5PN phase only through β: the combination
        # (∂β/∂χ₂, −∂β/∂χ₁) is an exact null direction at every mass ratio
        for eta in (0.25, 2 / 9, 0.16)
            dm = CD.Physics.mass_asymmetry(eta)
            c1 = (113 - 76 * eta + 113 * dm) / 24
            c2 = (113 - 76 * eta - 113 * dm) / 24
            dbeta = ForwardDiff.derivative(
                s -> spin_beta(0.3 + s * c2, -0.3 - s * c1, eta), 0.0)
            @test abs(dbeta) < 1e-12
        end
        # the Newtonian stationary-phase map is the frequency derivative of
        # the (2,2) phase at low velocity: the 1PN term shifts the derivative
        # by (3/5)(3715/756 + 55η/9) v² ≈ 7e-5 at v = 4e-3
        f0, Mc_light = 1e-5, 0.001
        dpsi_df = ForwardDiff.derivative(
            f -> harmonic_phase(f, 2, Mc_light, 0.25, 200.0, 0.3, 0.0), f0)
        @test dpsi_df / (2π) ≈ CD.Physics.spa_time(f0, Mc_light, 200.0) rtol = 5e-4
        # every harmonic shares the arrival-time term and the −π/4; the PN part
        # is (k/2) times the (2,2) phase at the (2,2) frequency 2f/k
        beta = spin_beta(0.5, 0.3, 0.16)
        for k in (1, 3)
            lhs =
                harmonic_phase(3e-3, k, Mc, 0.16, 200.0, 0.3, beta) - 2π * 3e-3 * 200.0 +
                π / 4
            rhs =
                (k / 2) * (
                    harmonic_phase(2 * 3e-3 / k, 2, Mc, 0.16, 200.0, 0.3, beta) -
                    2π * (2 * 3e-3 / k) * 200.0 + π / 4
                )
            @test lhs ≈ rhs rtol = 1e-12
        end
        # innermost stable orbit and the taper around it
        @test isco_frequency(4.925) ≈ 4.398e-3 rtol = 1e-3 # 10⁶ M⊙
        fi = isco_frequency(total_mass(Mc, 0.25))
        @test CD.Physics.inspiral_taper(0.5 * fi, fi, 0.1) ≈ 1.0 atol = 1e-4
        @test CD.Physics.inspiral_taper(fi, fi, 0.1) == 0.5
        @test CD.Physics.inspiral_taper(1.5 * fi, fi, 0.1) < 1e-4
        # the physical-model boundary rejects unknown keywords (typo protection)
        @test_throws ArgumentError waveform_params(sky_thata = 1.0)
        # type stability of the hot scalar cores
        @test (@inferred harmonic_amplitudes(1e-3, 3, 1.5, 0.16, 0.5, 1e17)) isa
              NTuple{2,Float64}
        @test (@inferred pn_phase(1e-3, 1.5, 0.16, 2.0)) isa Float64
    end

    @testset "Detector: orbits, patterns, channels" begin
        arm_sec = FIX_WP.arm_length / CD.Physics.C_LIGHT
        # analytic orbits: equal arms to O(e²) at every orbital phase, the
        # constellation centre on the 1 AU circle, the plane tilted by 60°
        for alpha in (0.0, 1.0, 2.5, 4.0)
            r1, r2, r3 = CD.Detector.spacecraft_positions(cos(alpha), sin(alpha),
                FIX_WP.geometry)
            arm(a, b) = sqrt(sum((a .- b) .^ 2))
            @test all(
                isapprox.((arm(r1, r2), arm(r2, r3), arm(r3, r1)), arm_sec; rtol = 3e-3),
            )
            center = (r1 .+ r2 .+ r3) ./ 3
            @test hypot(center[1], center[2]) ≈ CD.Physics.R_ORBIT_SEC rtol = 1e-4
            e12 = r2 .- r1
            e13 = r3 .- r1
            normal = (e12[2] * e13[3] - e12[3] * e13[2], e12[3] * e13[1] - e12[1] * e13[3],
                e12[1] * e13[2] - e12[2] * e13[1])
            @test abs(normal[3]) / sqrt(sum(normal .^ 2)) ≈ 0.5 atol = 1e-2
        end
        # sky- and polarization-averaged response power of A and E is the
        # long-wavelength 3/10 of a 60° Michelson (Robson et al. 2019, Eq. 13)
        # and the two channels are uncorrelated
        acc = zeros(3)
        n_dir = 0
        for lambda in range(0, 2π, length = 17)[1:(end-1)],
            sin_beta in range(-1, 1, length = 17), psi in
                                                   range(0, π, length = 5)[1:(end-1)]

            w = waveform_params(; ecliptic_longitude = lambda,
                ecliptic_latitude = asin(sin_beta), polarization = psi)
            FA_plus, FA_cross, FE_plus, FE_cross, _, _ =
                CD.Detector.channel_patterns(0.3 * SECONDS_PER_YEAR, w.geometry,
                    w.orbit_phase)
            acc .+= (FA_plus^2 + FA_cross^2, FE_plus^2 + FE_cross^2,
                FA_plus * FE_plus + FA_cross * FE_cross)
            n_dir += 1
        end
        @test acc[1] / n_dir ≈ 0.3 rtol = 0.03
        @test acc[2] / n_dir ≈ 0.3 rtol = 0.03
        @test abs(acc[3] / n_dir) < 0.01
        # Doppler: a wave from +x reaches the constellation at +x one AU early
        w0 = waveform_params(; ecliptic_longitude = 0.0, ecliptic_latitude = 0.0)
        @test CD.Detector.doppler_phase(1e-3, 1.0, 0.0, w0.geometry) ≈
              2π * 1e-3 * CD.Physics.R_ORBIT_SEC rtol = 1e-14
        @test CD.Physics.transfer_frequency(FIX_WP.arm_length) ≈ 2.99792458e8 / (2π * 2.5e9) rtol =
            1e-14
        # the strain dies above the innermost stable orbit
        physical(θ) = (θ[1] * FIX_WP.distance_scale, θ[2] * FIX_WP.mass_scale,
            θ[3] * FIX_WP.time_scale, θ[4], θ[5], θ[6])
        fi = isco_frequency(total_mass(THETA0[2] * FIX_WP.mass_scale, FIX_WP.eta))
        strain(f) = abs(channel_strain_bin(f, physical(THETA0)..., FIX_WP)[1])
        @test strain(2 * fi) < 1e-6 * strain(0.5 * fi)
        # fused channel strains ≡ per-bin core; T identically zero when requested
        A2, E2 = channel_strain(THETA0, FIX_FREQS, FIX_WP)
        @test length(A2) == length(FIX_FREQS) == length(E2)
        wp3 = waveform_params(; FIX_PHYS..., include_t_channel = true)
        A3, E3, T3 = channel_strain(THETA0, FIX_FREQS, wp3)
        @test A3 == A2 && E3 == E2 && all(iszero, T3)
        hA, hE = channel_strain_bin(FIX_FREQS[7], physical(THETA0)..., FIX_WP)
        @test hA == A2[7] && hE == E2[7]
        # both channels carry comparable power (no channel under-weighted)
        @test 0.2 < sum(abs2, E2) / sum(abs2, A2) < 5.0
        @test n_channels(FIX_WP) == 2 && n_channels(wp3) == 3
        @test (@inferred channel_strain(THETA0, FIX_FREQS, FIX_WP)) isa
              NTuple{2,Vector{ComplexF64}}
        @test (@inferred channel_strain(THETA0, FIX_FREQS, wp3)) isa
              NTuple{3,Vector{ComplexF64}}
        @test (@inferred channel_strain_bin(1e-3, physical(THETA0)..., FIX_WP)) isa
              NTuple{2,ComplexF64}
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
        # f plus rms/min/max of signal, best fit and residual for A and E
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
        @test cap_unbounded_radii!(r, 5.0) == (0, NaN) ||
              isnan(cap_unbounded_radii!(r, 5.0)[2])
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
    end

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
        # lanes path: nested (Hessian) duals through the same kernel layout
        @test ForwardDiff.hessian(dl, p) ≈ ForwardDiff.hessian(loop, p) rtol = 1e-10
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
        # one is functional, so the mismatch is easy to hit from the REPL)
        struct FakeGPU <: KernelAbstractions.GPU end
        @test_throws ArgumentError loss_function(data, FIX_FREQS, FIX_SN, FIX_DF,
            FIX_WP, FakeGPU())

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
            @test all(abs.(bf[5:6]) .<= 1.0 + 1e-9) # physical bounds respected
            push!(d2_ipn, d_new)
        end
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
            @test cfg.noise.confusion_knee_freq == 0.00258

            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[pipeline]\noptimizer = \"sgd\"\n"))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\nn_deltas = 1\n"))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[parameter_bounds]\nspin1 = [2.0, 1.0]\n"))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(
                    dir,
                    "[[sweeps]]\nname = \"bad/name\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0,1,0,0,0,0]\n",
                ))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(
                    dir,
                    "[[maps]]\nname = \"m\"\nparam_x = 2\nparam_y = 2\ntheta_0 = [1,1,1,0,0,0]\n",
                ))
            # case names that would escape the case directory are rejected
            for bad in (".", "..", "...")
                @test_throws ArgumentError load_and_validate_config(
                    write_cfg(
                        dir,
                        "[[maps]]\nname = \"$bad\"\nparam_x = 2\nparam_y = 3\ntheta_0 = [1,1,1,0,0,0]\n",
                    ))
            end
            @test CD.Config.fs_safe("a.b-c_1") && !CD.Config.fs_safe("a/b")
            # remaining validation branches: symmetric mass ratio, grid size,
            # progress-log cadence
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[physics]\neta = 0.3\n"))
            @test load_and_validate_config(write_cfg(dir, "[physics]\neta = 0.2\n")).wp.eta ==
                  0.2
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[physics]\necliptic_latitude = 2.0\n"))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[physics]\ncutoff_width = 0.0\n"))
            cfg_arm =
                load_and_validate_config(write_cfg(dir, "[noise]\narm_length = 5.0e9\n"))
            @test cfg_arm.wp.arm_length == 5.0e9 # one arm length for noise and response
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[monitoring]\nprogress_log_fraction = 2.0\n"))
            tiny = joinpath(dir, "tiny.toml")
            write(tiny, "[grid]\nT_obs = 100.0\nf_min = 1.0e-3\nf_max = 2.0e-3\n")
            @test_throws ArgumentError load_and_validate_config(tiny)
            # base point outside physical bounds
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(
                    dir,
                    "[[maps]]\nname = \"m\"\nparam_x = 5\nparam_y = 6\ntheta_0 = [1,1,1,0,1.5,0]\n",
                ))
            # unknown keys warn (typo protection), odd n_angles warns and rounds
            @test_logs (:warn, r"Unknown configuration key 'n_anglse'") match_mode = :any load_and_validate_config(
                write_cfg(dir, "[mapping]\nn_anglse = 100\n"))
            cfg_odd =
                @test_logs (:warn, r"odd; rounding up") match_mode = :any load_and_validate_config(
                    write_cfg(dir, "[mapping]\nn_angles = 33\n"))
            @test cfg_odd.map_n_angles == 34
            @test cfg_odd.corner_bisect_iters == 25 # safe default: corners exact
            cfg_nb = load_and_validate_config(
                write_cfg(dir, "[mapping]\ncorner_bisect_iters = 0\n"))
            @test cfg_nb.corner_bisect_iters == 0 # explicit opt-out allowed
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[mapping]\ncorner_bisect_iters = -1\n"))

            # amp_ratio and multi-start guardrails; an amplitude component in
            # the sweep direction is rejected (amp_ratio carries it)
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(
                    dir,
                    "[[sweeps]]\nname = \"s\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0.2,1,0,0,0,0]\n",
                ))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(
                    dir,
                    "[[sweeps]]\nname = \"s\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0,1,0,0,0,0]\namp_ratio = -1.0\n",
                ))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(
                    dir,
                    "[[sweeps]]\nname = \"s\"\ntheta_0 = [1,1,1,0,0,0]\nu_dir = [0.5,1,0,0,0,0]\namp_ratio = 0.5\n",
                ))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\nn_starts = 0\n"))
            cfg_ms = load_and_validate_config(
                write_cfg(
                    dir,
                    "[pipeline]\nrng_seed = 7\n[pipeline.sweep_settings]\nn_starts = 3\ng_tol = 1e-9\nmax_iterations = 300\n[hardware]\nhessian_chunk = 3\n",
                ))
            @test cfg_ms.n_starts == 3 && cfg_ms.rng_seed == 7
            @test cfg_ms.g_tol == 1e-9 && cfg_ms.max_iterations == 300
            @test cfg_ms.hessian_chunk == 3
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[hardware]\nhessian_chunk = 9\n"))
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\ng_tol = 0.0\n"))

            # safe-by-default optimizer tolerances; monitoring opt-in (off)
            cfg_def = load_and_validate_config(write_cfg(dir, ""))
            @test cfg_def.g_tol == 1e-10 && cfg_def.max_iterations == 100
            @test cfg_def.monitoring_enabled == false
            cfg_mon =
                load_and_validate_config(write_cfg(dir, "[monitoring]\nenabled = true\n"))
            @test cfg_mon.monitoring_enabled == true
            @test_logs (:warn, r"tighter than the numerical precision floor") match_mode =
                :any load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\ng_tol = 1e-12\n"))
            @test_logs (:warn, r"floor fits exhaust the full cap") match_mode = :any load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\nmax_iterations = 1000\n"))

            cfg_vs =
                load_and_validate_config(write_cfg(dir, "[safety]\nmax_vram_gb = 6.0\n"))
            @test cfg_vs.max_vram_gb == 6.0

            # all shipped configurations must validate as-is (production
            # variants carry the full 7-sweep campaign; quickstart carries 2)
            for shipped in filter(f -> endswith(f, ".toml"),
                readdir(joinpath(dirname(@__DIR__), "configs")))
                cfg_ship = load_and_validate_config(
                    joinpath(dirname(@__DIR__), "configs", shipped))
                if startswith(shipped, "production")
                    @test cfg_ship.n_deltas == 30 && length(cfg_ship.sweeps) == 7
                else
                    @test length(cfg_ship.sweeps) == 2
                end
            end
            # analysis tunables: defaults, override roundtrip, validation
            @test cfg_def.floor_detection_ratio == 2.0
            @test cfg_def.min_log_delta_ratio == -2.0 && cfg_def.max_log_delta_ratio == 0.3
            @test cfg_def.correction_fit_max_departure == 0.3
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir,
                    "[pipeline.sweep_settings]\ncorrection_fit_max_departure = 0.0\n"),
            )
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir,
                    "[pipeline.sweep_settings]\nmin_log_delta_ratio = 0.5\nmax_log_delta_ratio = 0.3\n",
                ),
            )
            @test cfg_def.secondary_minimum_gain == 1.5
            @test cfg_def.unbounded_cap_factor == 5.0
            @test cfg_def.residual_spectrum_windows == 600
            cfg_tun = load_and_validate_config(
                write_cfg(dir,
                    "[pipeline.sweep_settings]\nfloor_detection_ratio = 3.0\n" *
                    "[mapping]\nunbounded_cap_factor = 8.0\n"),
            )
            @test cfg_tun.floor_detection_ratio == 3.0
            @test cfg_tun.unbounded_cap_factor == 8.0
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[pipeline.sweep_settings]\nfloor_detection_ratio = 1.0\n"))

            cfg_quick = load_and_validate_config(
                joinpath(dirname(@__DIR__), "configs", "quickstart.toml"))
            @test length(cfg_quick.sweeps) == 2 && length(cfg_quick.maps) == 2
            @test cfg_quick.monitoring_enabled
            # work items parse once into typed specs with resolved defaults
            @test cfg_quick.sweeps isa Vector{SweepSpec}
            @test cfg_quick.maps isa Vector{MapSpec}
            @test cfg_quick.sweeps[1].rho_thresh == 1.0 # pipeline-wide default
            @test cfg_quick.sweeps[1].amp_ratio == 1.0
            @test cfg_quick.maps[1].n_angles == 256 # [mapping].n_angles fallback
            cfg_na =
                @test_logs (:warn, r"odd; rounding up") match_mode = :any load_and_validate_config(
                    write_cfg(
                        dir,
                        "[[maps]]\nname = \"m\"\nparam_x = 5\nparam_y = 6\n" *
                        "theta_0 = [1,1,1,0,0,0]\nn_angles = 33\n",
                    ))
            @test cfg_na.maps[1].n_angles == 34 # per-map override resolved even

            # instrumental-noise overrides thread through to NoiseParams and
            # must be strictly positive
            cfg_n = load_and_validate_config(
                write_cfg(dir, "[noise]\narm_length = 5.0e9\nacc_amplitude = 2.4e-15\n"))
            @test cfg_n.noise.arm_length == 5.0e9
            @test cfg_n.noise.acc_amplitude == 2.4e-15
            @test cfg_n.noise.oms_amplitude == 1.5e-11
            @test_throws ArgumentError load_and_validate_config(
                write_cfg(dir, "[noise]\narm_length = 0.0\n"))
        end
    end

    @testset "Provenance: backup-before-overwrite" begin
        mktempdir() do dir
            target = joinpath(dir, "results.csv")
            write(target, "first")
            # each call moves the existing file to the next free #k backup
            @test backup_existing!(target) == target
            write(target, "second")
            @test backup_existing!(target) == target
            write(target, "third")
            @test read(joinpath(dir, "results#1.csv"), String) == "first"
            @test read(joinpath(dir, "results#2.csv"), String) == "second"
            @test read(target, String) == "third"
            # a fresh path is returned untouched, creating nothing
            @test backup_existing!(joinpath(dir, "new.csv")) == joinpath(dir, "new.csv")
            @test !isfile(joinpath(dir, "new.csv"))
            # rerun directories are suffixed, never reused
            first_dir = CD.Provenance.unique_run_dir(dir, "run_abc")
            second_dir = CD.Provenance.unique_run_dir(dir, "run_abc")
            third_dir = CD.Provenance.unique_run_dir(dir, "run_abc")
            @test basename(first_dir) == "run_abc"
            @test basename(second_dir) == "run_abc_r2" && isdir(second_dir)
            @test basename(third_dir) == "run_abc_r3"
        end
    end

    @testset "Provenance: comment-invariant run identifiers" begin
        mktempdir() do dir
            a = joinpath(dir, "a.toml")
            b = joinpath(dir, "b.toml")
            c = joinpath(dir, "c.toml")
            write(a, "[grid]\nT_obs = 1.0e5\nf_min = 1.0e-3\n")
            # same values: reordered keys, comments, blank lines
            write(
                b,
                "# leading comment\n[grid]\n\nf_min = 1.0e-3 # trailing\nT_obs = 1.0e5\n",
            )
            # one value changed
            write(c, "[grid]\nT_obs = 2.0e5\nf_min = 1.0e-3\n")
            @test run_id_from_config(a) == run_id_from_config(b)
            @test run_id_from_config(a) != run_id_from_config(c)
            @test startswith(run_id_from_config(a), "run_")
        end
    end

    @testset "Provenance: base + overlay configurations" begin
        mktempdir() do dir
            base = joinpath(dir, "base.toml")
            overlay = joinpath(dir, "overlay.toml")
            mono = joinpath(dir, "monolithic.toml")
            write(
                base,
                """
[grid]
T_obs = 1.0e5
f_min = 1.0e-3
f_max = 2.0e-3
[hardware]
gpu_backend = "none"
max_threads = 4
hessian_chunk = 0
[[sweeps]]
name = "a"
theta_0 = [1, 1, 1, 0, 0, 0]
u_dir = [0, 1, 0, 0, 0, 0]
""",
            )
            write(
                overlay,
                """
base_config = "base.toml"
[hardware]
gpu_backend = "auto"
hessian_chunk = 3
[[sweeps]]
name = "b"
theta_0 = [1, 1, 1, 0, 0, 0]
u_dir = [0, 0, 1, 0, 0, 0]
""",
            )
            write(
                mono,
                """
[grid]
T_obs = 1.0e5
f_min = 1.0e-3
f_max = 2.0e-3
[hardware]
gpu_backend = "auto"
max_threads = 4
hessian_chunk = 3
[[sweeps]]
name = "b"
theta_0 = [1, 1, 1, 0, 0, 0]
u_dir = [0, 0, 1, 0, 0, 0]
""",
            )
            # merge semantics: sub-tables recurse, scalars and arrays of
            # tables replace, the base_config key is stripped
            eff = effective_config(overlay)
            @test !haskey(eff, "base_config")
            @test eff["hardware"]["max_threads"] == 4
            @test eff["hardware"]["gpu_backend"] == "auto"
            @test length(eff["sweeps"]) == 1 && eff["sweeps"][1]["name"] == "b"
            @test eff["grid"]["T_obs"] == 1.0e5
            # identity: an overlay hashes exactly like the equivalent monolithic file
            @test run_id_from_config(overlay) == run_id_from_config(mono)
            @test run_id_from_config(overlay) != run_id_from_config(base) # sweeps differ
            # execution sections never enter the identity: the same physical
            # case on another machine (threads, backend, budgets, monitoring)
            # keeps its run identifier
            exec_variant = joinpath(dir, "exec_variant.toml")
            write(exec_variant, read(mono, String) * """
[safety]
max_ram_gb = 200.0
max_vram_gb = 48.0
[monitoring]
enabled = true
""")
            @test run_id_from_config(exec_variant) == run_id_from_config(mono)
            hw_variant = joinpath(dir, "hw_variant.toml")
            write(
                hw_variant,
                replace(read(mono, String), "max_threads = 4" => "max_threads = 64",
                    "gpu_backend = \"auto\"" => "gpu_backend = \"none\""),
            )
            @test run_id_from_config(hw_variant) == run_id_from_config(mono)
            @test !any(
                haskey(identity_config(effective_config(exec_variant)), s)
                for s in ("hardware", "safety", "monitoring")
            )
            @test haskey(identity_config(effective_config(exec_variant)), "grid")
            cfg = load_and_validate_config(overlay)
            @test cfg.gpu_backend === :auto && cfg.max_threads == 4 &&
                  cfg.hessian_chunk == 3
            @test [s.name for s in cfg.sweeps] == ["b"]
            # the snapshot of an overlay is the self-contained merged table and
            # rehashes to the run's identifier; plain files are copied verbatim
            run_dir = mktempdir(dir)
            snap = CD.Provenance.snapshot_config(overlay, run_dir)
            @test !haskey(TOML.parsefile(snap), "base_config")
            @test run_id_from_config(snap) == run_id_from_config(overlay)
            @test_throws ArgumentError CD.Provenance.snapshot_config(overlay, run_dir)
            run_dir2 = mktempdir(dir)
            snap2 = CD.Provenance.snapshot_config(mono, run_dir2)
            @test read(snap2, String) == read(mono, String)
            # one overlay level only; a missing base fails loudly
            nested = joinpath(dir, "nested.toml")
            write(nested, "base_config = \"overlay.toml\"\n")
            @test_throws ArgumentError effective_config(nested)
            @test_throws ArgumentError load_and_validate_config(nested)
            absent = joinpath(dir, "absent.toml")
            write(absent, "base_config = \"no_such_file.toml\"\n")
            @test_throws ArgumentError effective_config(absent)
            write(absent, "base_config = 3\n")
            @test_throws ArgumentError effective_config(absent)
        end
        # the shipped GPU variants are thin [hardware] overlays on their bases
        configs = joinpath(dirname(@__DIR__), "configs")
        for (overlay, base) in (("production_gpu.toml", "production_cpu.toml"),
            ("production_oneapi.toml", "production_cpu.toml"),
            ("quickstart_gpu.toml", "quickstart.toml"))
            raw = TOML.parsefile(joinpath(configs, overlay))
            @test raw["base_config"] == base
            @test collect(keys(raw)) ⊆ ["base_config", "hardware"]
            eff = effective_config(joinpath(configs, overlay))
            @test eff["sweeps"] == TOML.parsefile(joinpath(configs, base))["sweeps"]
            # same physics, other execution settings: same run identifier
            @test run_id_from_config(joinpath(configs, overlay)) ==
                  run_id_from_config(joinpath(configs, base))
        end
        # per-host overlays: thin execution-only files on the production base,
        # each a complete, validating configuration with the base's identity
        hosts = joinpath(configs, "hosts")
        host_files = filter(f -> endswith(f, ".toml"), readdir(hosts))
        @test !isempty(host_files)
        production_id = run_id_from_config(joinpath(configs, "production_cpu.toml"))
        for host in host_files
            raw = TOML.parsefile(joinpath(hosts, host))
            @test raw["base_config"] == "../production_cpu.toml"
            @test collect(keys(raw)) ⊆ ["base_config", "hardware", "safety"]
            cfg_host = load_and_validate_config(joinpath(hosts, host))
            @test length(cfg_host.sweeps) == 7 && length(cfg_host.maps) == 7
            @test run_id_from_config(joinpath(hosts, host)) == production_id
        end
    end

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
    end

    @testset "Orchestrator: resource planning, seeding, mirroring" begin
        mktempdir() do dir
            path = joinpath(dir, "c.toml")
            grid = "[grid]\nT_obs = 1.0e5\nf_min = 1.0e-3\nf_max = 2.0e-3\n"
            write(path, grid * "[safety]\nmax_ram_gb = 0.001\n")
            cfg = load_and_validate_config(path)
            @test_throws CD.Orchestrator.ResourceBudgetError CD.Orchestrator.plan_resources(
                cfg, 10^6, 2, CPU())
            # a budget holding exactly two tasks of a million-bin grid
            # downscales the concurrency (needs ≥ 3 threads to be observable)
            if Threads.nthreads() >= 3
                write(
                    path,
                    grid * "[safety]\nmax_ram_gb = 1.3\n[hardware]\nmax_threads = 8\n",
                )
                cfg2 = load_and_validate_config(path)
                sweep_tasks, map_tasks, est =
                    @test_logs (:warn, r"Concurrency downscaled") CD.Orchestrator.plan_resources(
                        cfg2, 10^6, 2, CPU())
                @test sweep_tasks == 2 && map_tasks == 2 && est <= 1.3
            end
        end

        # multi-start streams: deterministic per work item, distinct across
        # starts, and pinned so a change of derivation cannot silently alter
        # the perturbations recorded under one rng_seed
        stream(k) = CD.Orchestrator.multi_start_stream(42, "sweep", 3, k)
        @test randn(stream(2), 3) == randn(stream(2), 3)
        @test randn(stream(2)) != randn(stream(3))
        @test randn(stream(2)) ≈ -1.5863254334055301 rtol = 1e-15

        # mirroring: K and g are even, r_box is re-evaluated on the negated
        # direction (the box is asymmetric), degenerate and capped flags follow
        entries = [(phi = 0.0, K = 16.0, g = 1.0, wall = 0x00),
            (phi = π / 2, K = 1e-320, g = 1e-9, wall = 0x00)]
        box = (-1.5, 2.0, -0.5, 0.5)
        r_math_of(K) = K > CD.Geometry.K_UNDERFLOW ? (16.0 / K)^(1 / 4) : Inf
        polar = CD.Orchestrator.mirror_to_full_circle(entries, box, r_math_of, 1e-6)
        @test polar.angle ≈ [0.0, π / 2, π, 3π / 2]
        @test polar.K_raw == [16.0, 1e-320, 16.0, 1e-320]
        @test polar.r_math[1] == polar.r_math[3] == 1.0 && isinf(polar.r_math[2])
        @test polar.r_box ≈ [2.0, 0.5, 1.5, 0.5]
        @test polar.r_cap ≈ [1.0, 0.5, 1.0, 0.5]
        @test polar.prior_limited == [false, true, false, true]
        @test polar.degenerate == [false, true, false, true]
        # a crossover vertex marked on one side only: that side is snapped onto
        # the wall and flagged although r_math sits a hair inside it; the
        # antipode keeps the plain comparison
        entries_w = [(phi = 0.0, K = 16.0 / (2.0 * (1 - 1e-7))^4, g = 1.0,
            wall = CD.Orchestrator.WALL_SELF)]
        polar_w = CD.Orchestrator.mirror_to_full_circle(entries_w, box, r_math_of, 1e-6)
        @test polar_w.r_cap[1] == 2.0 && polar_w.prior_limited[1]
        @test polar_w.r_cap[2] ≈ 1.5 && polar_w.prior_limited[2] # r_math ≈ 2 > 1.5 anyway
        @test CD.Orchestrator.wall_bit(0.5) == CD.Orchestrator.WALL_SELF
        @test CD.Orchestrator.wall_bit(4.0) == CD.Orchestrator.WALL_ANTIPODE
        @test CD.Orchestrator.wall_bit(-0.5) == CD.Orchestrator.WALL_ANTIPODE

        # box-corner insertion: a ray through a finite box corner that is capped
        # there becomes a boundary vertex; the four corners fold onto two
        # half-circle directions
        entries_c = [(phi = 0.0, K = 1.0, g = 1.0, wall = 0x00),
            (phi = π / 2, K = 1.0, g = 1.0, wall = 0x00)]
        n_inserted = CD.Orchestrator.insert_box_corner_vertices!(entries_c,
            phis -> [(1.0, 1.0) for _ in phis], (alpha, K) -> true,
            (-1.0, 1.0, -1.0, 1.0))
        @test n_inserted == 2 && length(entries_c) == 4
        @test any(e -> isapprox(e.phi, π / 4), entries_c) &&
              any(e -> isapprox(e.phi, 3π / 4), entries_c)
        @test issorted([e.phi for e in entries_c])
        # a symmetric box folds each corner pair onto one direction carrying
        # both wall bits (the direction and its antipode end on a corner)
        @test all(e -> e.wall == 0x03, filter(e -> !(e.phi in (0.0, π / 2)), entries_c))
        @test all(e -> e.wall == 0x00, filter(e -> e.phi in (0.0, π / 2), entries_c))
        # an uncapped corner inserts nothing
        @test CD.Orchestrator.insert_box_corner_vertices!(copy(entries_c),
            phis -> [(1.0, 1.0) for _ in phis], (alpha, K) -> false, (-1.0, 1.0, -1.0, 1.0),
        ) == 0

        # boundary runs: contiguous same-class edges join into NaN-separated
        # polylines; the wrap-around edge starts a new run
        xs, ys = CD.Plotting.boundary_runs([0.0, 1.0, 2.0, 3.0], [0.0, 0.0, 1.0, 1.0],
            [true, true, false, true], true)
        @test isequal(xs, [0.0, 1.0, 2.0, NaN, 3.0, 0.0])
        @test isequal(ys, [0.0, 0.0, 1.0, NaN, 1.0, 0.0])
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
        # …and 10¹ as plain 10 (products fold: 2×10¹ → 20)
        @test l0[2].s == "\$10\$"
        v125b, l125b = CD.Plotting.log_ticks_125(5.0, 60.0)
        @test l125b[findfirst(≈(10.0), v125b)].s == "\$10\$"
        @test l125b[findfirst(≈(20.0), v125b)].s == "\$20\$"
        # annotations go scientific outside exponents −1..1, 3 significant digits
        @test CD.Plotting.sci_latex(0.00173) == "1.73\\times 10^{-3}"
        @test CD.Plotting.sci_latex(23.4) == "23.4"

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

        # fit coefficients: mantissa ×10ⁿ with two decimals outside exponents −1..1
        # slope annotation: decimals follow the uncertainty (≥ 3, ≤ 6)
        @test CD.Plotting.slope_latex(3.99984, 0.00012) == "3.9998 \\pm 0.0001"
        @test CD.Plotting.slope_latex(3.8241, 0.0432) == "3.824 \\pm 0.043"
        @test CD.Plotting.slope_latex(3.9812, 0.0081) == "3.981 \\pm 0.008"
        @test CD.Plotting.slope_latex(4.0, NaN) == "4.000"
        @test CD.Plotting.coef_latex(0.001278) == "1.28\\times 10^{-3}"
        @test CD.Plotting.coef_latex(-0.0235) == "-2.35\\times 10^{-2}"
        @test CD.Plotting.coef_latex(1.5) == "1.50" # ×10⁰ factor omitted
        @test CD.Plotting.coef_latex(14.46) == "14.5" # 2×10¹-style products fold
        @test CD.Plotting.coef_latex(0.2346) == "0.235" # …and 10⁻¹ likewise
        @test CD.Plotting.coef_latex(-14.46) == "-14.5"
        # per-tick common-exponent labels never show a power in −1..1
        plain = CD.Plotting.sci_tick_labels([-1.5, 0.0, 0.5])
        @test plain[1].s == "\$-1.5\$" && plain[3].s == "\$0.5\$"
        @test !any(l -> occursin("times", l.s), plain)
    end

    @testset "End-to-end minimal pipeline" begin
        mktempdir() do dir
            cfg_path = joinpath(dir, "config.toml")
            write(
                cfg_path,
                """
[pipeline]
optimizer = "ipnewton"
rng_seed = 11
[pipeline.sweep_settings]
n_deltas = 6
min_log_delta_ratio = -2.0
max_log_delta_ratio = 0.3
n_starts = 2
[grid]
T_obs = 1.0e5
f_min = 1.0e-3
f_max = 3.0e-3
[physics]
time_scale = 100.0
ecliptic_longitude = 3.1415
ecliptic_latitude = 0.5238
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
""",
            )
            out_base = run_pipeline(cfg_path, dir, "outputs")
            @test isdir(out_base)
            @test isfile(joinpath(out_base, "config.toml"))
            @test isfile(joinpath(out_base, "run.log"))
            meta = TOML.parsefile(joinpath(out_base, "metadata.toml"))
            # hardware-provenance sidecar: host fingerprint always present
            @test isfile(joinpath(out_base, "hardware.txt"))
            @test occursin("BLAS threads", read(joinpath(out_base, "hardware.txt"), String))
            @test haskey(meta, "git") && haskey(meta, "finished")
            @test meta["failed_stages"] == ""

            sdir = joinpath(out_base, "sweeps", "mini_sweep")
            res = CSV.read(joinpath(sdir, "results.csv"), DataFrame)
            @test nrow(res) == 6
            @test all(
                hasproperty.(Ref(res), [:BestFit_Spin1, :Converged, :Iterations, :AtBound]),
            )
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
            res_u = CSV.read(
                joinpath(out_base, "sweeps", "mini_unequal", "results.csv"),
                DataFrame,
            )
            ratio_u = res_u.D2_Numerical[1:3] ./ res_u.D2_Theoretical[1:3]
            @test all(0.85 .< ratio_u .< 1.15)
            meta_u = TOML.parsefile(
                joinpath(out_base, "sweeps", "mini_unequal", "sweep_meta.toml"),
            )
            @test meta_u["amp_ratio"] == 0.5
            # δ_min inverts the amplitude-prefactored law (p/16) K δ⁴ = ρ²
            @test meta_u["delta_min"] ≈
                  (16 * meta_u["rho_sq"] / ((2 * 0.5 / 1.5)^2 * meta_u["K_u_norm"]))^(1 / 4) rtol =
                1e-12

            for map_name in ("mini_spin_map", "mini_mass_time")
                mdir = joinpath(out_base, "maps", map_name)
                cm = CSV.read(joinpath(mdir, "confusion_contour.csv"), DataFrame)
                n = nrow(cm)
                @test iseven(n)
                half = n ÷ 2
                # the MATHEMATICAL radius is exactly centrally symmetric by
                # construction (mirrored evaluation); the capped radius need
                # not be — the prior box may be asymmetric about the base point
                @test cm.R_Math[1:half] == cm.R_Math[(half+1):end]
                @test cm.K_Raw[1:half] == cm.K_Raw[(half+1):end]
                @test all(cm.Angle[(half+1):end] .≈ cm.Angle[1:half] .+ π)
                @test isfile(joinpath(mdir, "confusion_zone.pdf"))
                @test isfile(joinpath(mdir, "confusion_zone.png"))
                # every vertex sitting on a prior wall carries the flag (the
                # bisected crossover vertices included), so the drawn wall
                # segment reaches its corners
                on_wall = isfinite.(cm.R_Box) .& (cm.R_Capped .== cm.R_Box)
                @test all(cm.Prior_Limited[on_wall])
                @test !any(cm.Prior_Limited[.!on_wall])
                # exact prior-crossover corners: at every capped/uncapped
                # transition along the boundary polygon, one of the two rows
                # must be the bisection-inserted vertex with r_math ≈ r_box —
                # otherwise the polygon chord chamfers the zone corner
                n_trans = 0
                for i in 1:n
                    j = mod1(i + 1, n)
                    cm.Prior_Limited[i] == cm.Prior_Limited[j] && continue
                    n_trans += 1
                    rel = [
                        abs(cm.R_Math[k] - cm.R_Box[k]) / cm.R_Box[k]
                        for
                        k in (i, j) if isfinite(cm.R_Box[k]) && isfinite(cm.R_Math[k])
                    ]
                    @test any(<(1e-3), rel)
                end
                count(cm.Prior_Limited) in (0, n) || @test n_trans >= 2
            end

            # exact box-corner vertex: whenever both walls of the mass-time box
            # are active, the boundary must pass through their corner exactly
            cmt = CSV.read(
                joinpath(out_base, "maps", "mini_mass_time", "confusion_contour.csv"),
                DataFrame)
            x_wall = abs.(cmt.X_Bound .+ 1.5) .< 1e-9
            y_wall = abs.(cmt.Y_Bound .+ 2.0) .< 1e-9
            if any(x_wall) && any(y_wall)
                @test any(x_wall .& y_wall)
            end

            # the spin map must be capped inside the physical box
            cm = CSV.read(
                joinpath(out_base, "maps", "mini_spin_map", "confusion_contour.csv"),
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
            rcsv =
                joinpath(out_base, "maps", "mini_spin_map", "confusion_contour_rho2p0.csv")
            @test isfile(rcsv)
            cm2 = CSV.read(rcsv, DataFrame)
            @test all(cm2.R_Capped .<= cm2.R_Box .+ 1e-12)
            # where neither threshold is box-capped, radii scale exactly by √2
            free = .!cm.Prior_Limited .& .!cm2.Prior_Limited
            if any(free)
                @test all(
                    isapprox.(
                        cm2.R_Capped[free],
                        sqrt(2.0) .* cm.R_Capped[free];
                        rtol = 1e-10,
                    ),
                )
            end
            @test isfile(
                joinpath(out_base, "maps", "mini_spin_map", "confusion_zone_rho2p0.png"),
            )

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
            cmd_collect =
                addenv(`$jlbin --startup-file=no $collect_script $browse_dir $out_base`,
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

            # an overlay (base_config + partial tables) runs end to end: the
            # merged [pipeline]/[mapping] values take effect, the snapshot is
            # self-contained and rehashes to the run's identifier, and the
            # metadata records both files
            overlay_path = joinpath(dir, "overlay.toml")
            write(
                overlay_path,
                """
base_config = "config.toml"
[pipeline]
run_1d_sweeps = false
[mapping]
n_angles = 16
max_refine_levels = 1
corner_bisect_iters = 0
""",
            )
            out3 = run_pipeline(overlay_path, dir, "outputs")
            @test startswith(basename(out3), run_id_from_config(overlay_path))
            snap = TOML.parsefile(joinpath(out3, "config.toml"))
            @test !haskey(snap, "base_config")
            @test snap["pipeline"]["run_1d_sweeps"] == false
            @test snap["pipeline"]["rng_seed"] == 11
            @test snap["mapping"]["n_angles"] == 16
            @test run_id_from_config(joinpath(out3, "config.toml")) ==
                  run_id_from_config(overlay_path)
            meta3 = TOML.parsefile(joinpath(out3, "metadata.toml"))
            @test meta3["config_file"] == "overlay.toml"
            @test meta3["base_config"] == "config.toml"
            @test meta["config_file"] == "config.toml" && meta["base_config"] == ""
            @test isfile(joinpath(out3, "maps", "mini_spin_map", "confusion_contour.csv"))
            @test !isdir(joinpath(out3, "sweeps"))
        end
    end

end
