# Noise model, waveform, post-Newtonian phase and detector response.
# Included by runtests.jl.

@testset "Noise PSD (Robson 2019)" begin
    # instrumental part must reproduce the reference values to 1e-14 relative
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
    # Eq. 14 is a two-channel sensitivity-level fit; it enters the
    # per-channel PSD through the Eq. 9 response R(f) = (3/10)/(1 + 0.6 (f/f★)²)
    f_star = CD.Physics.transfer_frequency(np.arm_length)
    R = sky_averaged_response(f, np.arm_length)
    @test R ≈ 0.3 / (1 + 0.6 * (f / f_star)^2) rtol = 1e-14
    @test analytic_noise_psd(f) ≈ analytic_noise_psd(f; noise = FIX_NOISE_OFF) + R * sc rtol =
        1e-14
    @test sc > 0
    @test 0 < R < 0.3
    @test sky_averaged_response(0.0, np.arm_length) == 0.3
    # the confusion term more than doubles the PSD at 1 mHz
    @test analytic_noise_psd(1e-3) > 2 * analytic_noise_psd(1e-3; noise = FIX_NOISE_OFF)

    # closure at the sensitivity level: the per-channel PSD divided by R(f)
    # is the published sensitivity curve, Eq. 13 (instrumental) + Eq. 14
    L = np.arm_length
    for fx in (2e-4, 1e-3, 3e-3, 1e-2, 3e-2)
        p_oms = np.oms_amplitude^2 * (1 + (np.oms_reddening_freq / fx)^4)
        p_acc =
            np.acc_amplitude^2 * (1 + (np.acc_knee_low / fx)^2) *
            (1 + (fx / np.acc_knee_high)^4)
        s13 =
            10 / (3 * L^2) *
            (p_oms + 2 * (1 + cos(fx / f_star)^2) * p_acc / (2 * π * fx)^4) *
            (1 + 0.6 * (fx / f_star)^2)
        s14 =
            np.confusion_amp * fx^(-7 / 3) *
            exp(
                -(fx^np.confusion_alpha) +
                np.confusion_beta * fx * sin(np.confusion_kappa * fx),
            ) *
            (1 + tanh(np.confusion_gamma * (np.confusion_knee_freq - fx)))
        @test analytic_noise_psd(fx) / sky_averaged_response(fx, L) ≈ s13 + s14 rtol =
            1e-12
    end

    # Table-1 selection by observation time
    @test robson_confusion_params(SECONDS_PER_YEAR).confusion_beta == 292.0
    @test robson_confusion_params(4 * SECONDS_PER_YEAR).confusion_beta == -221.0
    @test robson_confusion_params(0.4 * SECONDS_PER_YEAR).confusion_knee_freq == 0.00258

    @test analytic_noise_psd(0.0) == 1e-30
    @test analytic_noise_psd(-1.0) == 1e-30

    # instrumental parameters (Eq. 10-12) are configurable; defaults
    # reproduce the fixture PSD (test above), and the OMS term must respond
    # quadratically where it dominates (high f)
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
    # harmonic k carries −(k/2)φ_c: the waveform is 4π-periodic in the
    # phase, and a 2π shift reverses the odd harmonics unless the masses
    # are equal
    θ_ph = [1.0, 1.5, 2.0, 0.3, 0.5, 0.3]
    phase_shifted(Δ) = θ_ph .+ Δ .* (1:6 .== 4)
    for (eta, two_pi_periodic) in ((2 / 9, false), (0.25, true))
        wp = waveform_params(; merge(FIX_PHYS, (; eta))...)
        A0, E0 = channel_strain(θ_ph, FIX_FREQS, wp)
        A4, E4 = channel_strain(phase_shifted(4π), FIX_FREQS, wp)
        A2, E2 = channel_strain(phase_shifted(2π), FIX_FREQS, wp)
        @test isapprox(A4, A0; rtol = 1e-10) && isapprox(E4, E0; rtol = 1e-10)
        @test isapprox(A2, A0; rtol = 1e-10) == two_pi_periodic
        @test isapprox(E2, E0; rtol = 1e-10) == two_pi_periodic
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
    # constellation center on the 1 AU circle, the plane tilted by 60°
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
    # long-wavelength 3/10 of a 60° Michelson (Robson et al. 2019, Eq. 9)
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

@testset "2PN phase: spin–spin coefficient and the lifted spin degeneracy" begin
    Mc = 1.5
    # σ and ψ₄ against exact rational evaluations of the literature forms
    # (Arun et al. 2009 with its erratum, reduced to aligned spins, and
    # Khan et al. 2016, Appendix B)
    for (eta, chi1, chi2, sigma_ref, psi4_ref) in (
        (1 / 4, 1 / 2, 1 / 2, 5 / 4, 8571995 / 254016),
        (2 / 9, 7 / 10, 1 / 2, 28961 / 14400, 550410989 / 22861440),
        (4 / 25, 7 / 10, -1 / 5, 2749 / 2000, 1655708177 / 63504000),
        (2 / 9, 3 / 10, -3 / 10, 89 / 1600, 997479149 / 22861440),
    )
        sigma = spin_sigma(chi1, chi2, eta)
        @test sigma ≈ sigma_ref rtol = 1e-13
        @test CD.Physics.phase_coefficient_2pn(eta, sigma) ≈ psi4_ref rtol = 1e-13
    end
    # per body: spin1–spin2 coupling 79/8 = (721 − 247)/48, then
    # quadrupole–monopole 5 plus self-spin 1/16 weighted by (mᵢ/M)²
    @test spin_sigma(0.9, -0.4, 0.16) ≈
          (79 / 8) * 0.16 * 0.9 * (-0.4) +
          (5 + 1 / 16) * (0.8^2 * 0.9^2 + 0.2^2 * 0.4^2) rtol = 1e-13
    # exchanging the bodies at equal mass leaves σ unchanged
    @test spin_sigma(0.7, -0.2, 0.25) ≈ spin_sigma(-0.2, 0.7, 0.25) rtol = 1e-14
    # the 2PN term adds (3/(128 η v⁵)) ψ₄ v⁴ to the phase and nothing else;
    # with ψ₄ = 0 the phase is the 1.5PN one
    for eta in (0.25, 0.16)
        beta = spin_beta(0.7, -0.2, eta)
        psi4 = CD.Physics.phase_coefficient_2pn(eta, spin_sigma(0.7, -0.2, eta))
        v = (π * total_mass(Mc, eta) * 3e-3)^(1 / 3)
        @test pn_phase(3e-3, Mc, eta, beta, psi4) - pn_phase(3e-3, Mc, eta, beta) ≈
              3 * psi4 / (128 * eta * v) rtol = 1e-10
        @test pn_phase(3e-3, Mc, eta, beta, 0.0) == pn_phase(3e-3, Mc, eta, beta)
        @test harmonic_phase(3e-3, 3, Mc, eta, 200.0, 0.3, beta, 0.0) ==
              harmonic_phase(3e-3, 3, Mc, eta, 200.0, 0.3, beta)
    end
    @test (@inferred pn_phase(1e-3, 1.5, 0.16, 2.0, 30.0)) isa Float64
    @test (@inferred spin_sigma(0.5, 0.3, 0.16)) isa Float64

    # the model switch: 1.5PN by default, the listed orders on request
    @test FIX_WP.phase_pn_order == 1.5
    wp_2pn = waveform_params(; FIX_PHYS..., phase_pn_order = 2)
    @test wp_2pn.phase_pn_order == 2.0
    @test isbitstype(typeof(wp_2pn))
    @test_throws ArgumentError waveform_params(; FIX_PHYS..., phase_pn_order = 2.25)
    @test_throws ArgumentError waveform_params(; FIX_PHYS..., phase_pn_order = 1.0)
    @test (@inferred channel_strain(THETA0, FIX_FREQS, wp_2pn)) isa
          NTuple{2,Vector{ComplexF64}}
    # the default order evaluates the 1.5PN bracket itself: the 2PN term
    # with an exact zero changes no bit, for values and for dual numbers
    for eta in (0.25, 0.16)
        beta = spin_beta(0.7, -0.2, eta)
        bracket_15pn(F) =
            let v = (π * total_mass(Mc, eta) * F)^(1 / 3)
                (3 / (128 * eta * v^5)) *
                (1 + (3715 / 756 + 55 * eta / 9) * v^2 + (4 * beta - 16 * π) * v^3)
            end
        @test pn_phase(3e-3, Mc, eta, beta) == bracket_15pn(3e-3)
        @test ForwardDiff.derivative(F -> pn_phase(F, Mc, eta, beta), 3e-3) ==
              ForwardDiff.derivative(bracket_15pn, 3e-3)
    end
    # naming the default order changes nothing
    @test channel_strain(THETA0, FIX_FREQS, FIX_WP) ==
          channel_strain(THETA0, FIX_FREQS,
        waveform_params(; FIX_PHYS..., phase_pn_order = 1.5))
    @test channel_strain(THETA0, FIX_FREQS, wp_2pn)[1] !=
          channel_strain(THETA0, FIX_FREQS, FIX_WP)[1]

    # Tangent rank. At 1.5PN the spins enter through β alone: rank 5 at
    # every base point. At 2PN they enter through β and σ, and the rank is
    # 6 except where ∇σ ∥ ∇β, the line χ₂/χ₁ = q(807 − 713q)/(807q − 713),
    # q = m₁/m₂, through the origin of the spin plane: χ₂ = χ₁ at equal
    # mass (exchange symmetry), χ₂ = −(1238/901)χ₁ at η = 2/9
    tangent_rank(theta, wp) =
        length(compute_tangent_basis(theta, FIX_FREQS, FIX_SN, FIX_DF, wp))
    base_point(chi1, chi2) = [1.0, 1.5, 2.0, 0.0, chi1, chi2]
    for (eta, slope) in ((0.25, 1.0), (2 / 9, -1238 / 901))
        wp15 = waveform_params(; merge(FIX_PHYS, (; eta))...)
        wp2 = waveform_params(; merge(FIX_PHYS, (; eta, phase_pn_order = 2))...)
        on_line = base_point(0.5, 0.5 * slope)
        off_line = base_point(0.5, 0.5 * slope + 0.3)
        @test tangent_rank(on_line, wp15) == 5
        @test tangent_rank(off_line, wp15) == 5
        @test tangent_rank(on_line, wp2) == 5
        @test tangent_rank(off_line, wp2) == 6
        @test tangent_rank(base_point(0.0, 0.0), wp2) == 5
    end

    # Fisher norm along the 1.5PN null direction at equal mass, u = e₅ − e₆
    # (χ_a → χ_a + s at fixed χ_s): β is unchanged, σ = 5χ_s² + χ_a²/16
    # changes at the rate χ_a/8, and only the k = 2 harmonic radiates, so
    # ∂_u h̃ = i (χ_a/8) [30/(128 η v)] h̃ and
    # g(u,u) = (χ_a/8)² · 4 df Σ_C Σ_f [30/(128 η v)]² |h̃_C|²/S_n
    let chi1 = 0.8, chi2 = 0.2, eta = 0.25
        theta = base_point(chi1, chi2)
        u = [0.0, 0.0, 0.0, 0.0, 1.0, -1.0]
        A, E = channel_strain(theta, FIX_FREQS, wp_2pn)
        M = total_mass(theta[2] * wp_2pn.mass_scale, eta)
        closed =
            ((chi1 - chi2) / 2 / 8)^2 * 4 * FIX_DF *
            sum(
                (30 / (128 * eta * (π * M * f)^(1 / 3)))^2 * (abs2(a) + abs2(e)) / s
                for (f, a, e, s) in zip(FIX_FREQS, A, E, FIX_SN)
            )
        _, g_2pn =
            compute_extrinsic_curvature(theta, u, FIX_FREQS, FIX_SN, FIX_DF, wp_2pn)
        _, g_15pn =
            compute_extrinsic_curvature(theta, u, FIX_FREQS, FIX_SN, FIX_DF, FIX_WP)
        @test g_2pn ≈ closed rtol = 1e-9
        @test g_15pn < 1e-18 * g_2pn
    end
end

@testset "TaylorF2 phase to 4.5PN: coefficients, orders, spin degeneracy" begin
    # coefficients against exact evaluations of the literature forms at three
    # points (notes on the sources in the documentation, Waveform Physics §2);
    # fields: p2, p3, p4, p5, p5L, p6, p6L, p7, p8L, p8L2, p9, p9L
    reference = Dict(
        (2 / 9, 0.8, 0.8) => (6.272045855379189, -24.63585282780706,
            12.14331560041712, -48.26149699618888, -144.7844909885667,
            705.5777629344754, -326.0952380952381, -3537.735296498236,
            -6560.388494361212, 620.6955572352398, 1863.297001420149,
            -2048.916808741229),
        (2 / 9, 0.3, -0.3) => (6.272045855379189, -46.49881579077003,
            43.63151004486157, 122.8847650065546, 368.6542950196638,
            -774.0574789881386, -326.0952380952381, 372.3098396129402,
            -826.1415395472197, 620.6955572352398, 1863.297001420149,
            -2048.916808741229),
        (1 / 4, 0.5, 0.5) => (6.441798941798942, -34.59881579077003,
            33.74588608591585, 31.94476295303460, 95.83428885910380,
            -80.84873279886557, -326.0952380952381, -1511.268814396760,
            -3451.814730053968, 661.7626606198035, 1231.070858379564,
            -2048.916808741229),
    )
    # Two points with χ_s χ_a ≠ 0 (χ₁ = 0.7, χ₂ = −0.2), at which every mixed
    # spin term and the δ-weighted χ_a terms contribute, η = 2/9 (δ = 1/3) and
    # η = 4/25 (δ = 3/5); exact rational evaluations of the same literature forms,
    # rounded from 60 digits.
    mixed = Dict(
        (2 / 9, 0.7, -0.2) => (6.2720458553791887, -36.606223198177433,
            36.009982267083788, 42.861861812355109, 128.58558543706533,
            -124.68631514269648, -326.09523809523810, -1441.8703435546501,
            -3672.7424490246482, 620.69555723523977, 1863.2970014201487,
            -2048.9168087412290),
        (4 / 25, 0.7, -0.2) => (5.8917989417989418, -31.692149124103358,
            26.072502157344419, 0.45428318668630907, 1.3628495600589272,
            423.13011030733959, -326.09523809523810, -2737.7633469852079,
            -6362.4182011668335, 528.70524565381708, 3286.3328462714924,
            -2048.9168087412290),
    )
    names = (:p2, :p3, :p4, :p5, :p5L, :p6, :p6L, :p7, :p8L, :p8L2, :p9, :p9L)
    for ((eta, chi1, chi2), ref) in reference
        c = phase_coefficients(chi1, chi2, eta, 4.5)
        for (name, value) in zip(names, ref)
            @test getfield(c, name) ≈ value rtol = 1e-12
        end
    end
    # every order switches on the terms of its own column, with the values of
    # the full set, and nothing else
    order_of = (p2 = 1.5, p3 = 1.5, p4 = 2.0, p5 = 2.5, p5L = 2.5, p6 = 3.0, p6L = 3.0,
        p7 = 3.5, p8L = 4.0, p8L2 = 4.0, p9 = 4.5, p9L = 4.5)
    for ((eta, chi1, chi2), ref) in mixed, order in CD.Physics.PHASE_PN_ORDERS
        c = phase_coefficients(chi1, chi2, eta, order)
        for (name, value) in zip(names, ref)
            if order_of[name] > order
                @test iszero(getfield(c, name))
            else
                @test getfield(c, name) ≈ value rtol = 1e-12
            end
        end
    end
    # an order outside the set is not checked here and acts as the order below
    @test phase_coefficients(0.5, 0.3, 0.2, 2.25) == phase_coefficients(0.5, 0.3, 0.2, 2.0)
    # stationary-phase time from the phase of each order: its offset from t_c
    # is (1/2π) dψ_PN/dF, here by automatic differentiation of pn_phase; with a
    # unit bracket it is the Newtonian time
    let Mc = 1.5, eta = 0.16, tc = 200.0
        for order in CD.Physics.PHASE_PN_ORDERS, F in (1e-4, 3e-3, 2e-2)
            c = phase_coefficients(0.7, -0.2, eta, order)
            dψ = ForwardDiff.derivative(x -> pn_phase(x, Mc, eta, c, order), F)
            @test CD.Physics.spa_time(F, Mc, eta, tc, c, order) - tc ≈ dψ / (2π) rtol =
                1e-12
        end
        unit = CD.Physics.PhaseCoefficients(ntuple(_ -> 0.0, 12)...)
        @test CD.Physics.spa_time(3e-3, Mc, eta, tc, unit, 4.5) - tc ≈
              CD.Physics.spa_time(3e-3, Mc, tc) - tc rtol = 1e-13
    end
    # the response follows it only on request
    let wp_n = waveform_params(; FIX_PHYS..., phase_pn_order = 3.5),
        wp_p = waveform_params(; FIX_PHYS..., phase_pn_order = 3.5, pn_spa_time = true)

        @test !waveform_params(; FIX_PHYS...).pn_spa_time
        θ = [1.0, 1.5, 2.0, 0.0, 0.7, -0.2]
        a_n, a_p =
            channel_strain(θ, FIX_FREQS, wp_n)[1], channel_strain(θ, FIX_FREQS, wp_p)[1]
        @test a_n != a_p
        @test maximum(abs, a_n .- a_p) < 0.1 * maximum(abs, a_n)
    end
    # harmonic k evaluates the (2,2) phase at 2f/k, at every order
    let f = 3e-3, Mc = 1.5, eta = 0.16, tc = 200.0, φc = 0.3
        for order in CD.Physics.PHASE_PN_ORDERS, k in (1, 2, 3)
            c = phase_coefficients(0.7, -0.2, eta, order)
            @test harmonic_phase(f, k, Mc, eta, tc, φc, c, order) ≈
                  (k / 2) * pn_phase(2f / k, Mc, eta, c, order) + 2π * f * tc -
                  (k / 2) * φc - π / 4 rtol = 1e-14
        end
    end
    @test_throws ArgumentError waveform_params(; FIX_PHYS..., phase_pn_order = 5.0)
    @test_throws ArgumentError waveform_params(; FIX_PHYS..., phase_pn_order = 2.25)
    # orders 1.5 and 2 through the coefficient set reproduce the β/ψ₄ form, bit
    # for bit; each higher order adds its term (3/(128ηv⁵)) Δ v^i
    Mc, F = 1.5, 3e-3
    for eta in (0.25, 2 / 9)
        beta = spin_beta(0.7, -0.2, eta)
        psi4 = CD.Physics.phase_coefficient_2pn(eta, spin_sigma(0.7, -0.2, eta))
        @test pn_phase(F, Mc, eta, phase_coefficients(0.7, -0.2, eta, 1.5), 1.5) ==
              pn_phase(F, Mc, eta, beta)
        @test pn_phase(F, Mc, eta, phase_coefficients(0.7, -0.2, eta, 2.0), 2.0) ==
              pn_phase(F, Mc, eta, beta, psi4)
        c = phase_coefficients(0.7, -0.2, eta, 4.5)
        v = (π * total_mass(Mc, eta) * F)^(1 / 3)
        lv = log(v)
        pref = 3 / (128 * eta * v^5)
        term = Dict(2.5 => (c.p5 + c.p5L * lv) * v^5, 3.0 => (c.p6 + c.p6L * lv) * v^6,
            3.5 => c.p7 * v^7, 4.0 => (c.p8L * lv + c.p8L2 * lv^2) * v^8,
            4.5 => (c.p9 + c.p9L * lv) * v^9)
        for (lower, upper) in ((2.0, 2.5), (2.5, 3.0), (3.0, 3.5), (3.5, 4.0), (4.0, 4.5))
            @test pn_phase(F, Mc, eta, c, upper) - pn_phase(F, Mc, eta, c, lower) ≈
                  pref * term[upper] rtol = 1e-9
        end
    end
    @test (@inferred phase_coefficients(0.5, 0.3, 0.16, 4.5)) isa
          CD.Physics.PhaseCoefficients{Float64}
    dual_spin = ForwardDiff.Dual(0.5, 1.0)
    @test (@inferred phase_coefficients(dual_spin, 0.3, 0.16, 3.5)) isa
          CD.Physics.PhaseCoefficients{typeof(dual_spin)}
    @test (@inferred pn_phase(1e-3, 1.5, 0.16,
        phase_coefficients(0.5, 0.3, 0.16, 4.5), 4.5)) isa Float64

    # Tangent rank. From 2.5PN the spin–orbit coefficient of that order is
    # linear in a third spin combination, so for unequal masses the rank is 6
    # also at zero spins and on the 2PN residual line; at equal mass and equal
    # spins the exchange symmetry keeps rank 5 at every order
    tangent_rank(theta, wp) =
        length(compute_tangent_basis(theta, FIX_FREQS, FIX_SN, FIX_DF, wp))
    on_2pn_line = [1.0, 1.5, 2.0, 0.0, 0.5, 0.5 * (-1238 / 901)]
    zero_spins = [1.0, 1.5, 2.0, 0.0, 0.0, 0.0]
    for order in (2.5, 3.0, 3.5, 4.0, 4.5)
        wp_u =
            waveform_params(; merge(FIX_PHYS, (; eta = 2 / 9, phase_pn_order = order))...)
        @test tangent_rank(on_2pn_line, wp_u) == 6
        @test tangent_rank(zero_spins, wp_u) == 6
        wp_e = waveform_params(; FIX_PHYS..., phase_pn_order = order)
        @test tangent_rank(THETA0, wp_e) == 5
    end
    # the waveform at the highest order runs through the per-bin core and the
    # kernel alike
    wp45 = waveform_params(; merge(FIX_PHYS, (; eta = 2 / 9, phase_pn_order = 4.5))...)
    @test (@inferred channel_strain(THETA0, FIX_FREQS, wp45)) isa
          NTuple{2,Vector{ComplexF64}}
    c45 = channel_strain(THETA0, FIX_FREQS, wp45)
    c15 = channel_strain(THETA0, FIX_FREQS,
        waveform_params(; merge(FIX_PHYS, (; eta = 2 / 9))...))
    @test c45[1] != c15[1]
end
