using Pkg
const PROJECT_ROOT = dirname(@__DIR__)
Pkg.activate(PROJECT_ROOT; io=devnull)

using TwoWaveformDistinguishability
using Test
using LinearAlgebra
using KernelAbstractions

@testset "TwoWaveformDistinguishability.jl" begin
    
    @testset "Hardware Module" begin
        backend = get_best_backend()
        @test backend isa KernelAbstractions.Backend
        
        # Test fallback conversion
        arr = [1.0, 2.0, 3.0]
        dev_arr = to_backend(arr, backend)
        @test length(dev_arr) == 3
    end
    
    @testset "Physics Module" begin
        # Test Noise PSD
        @test analytic_noise_psd(1e-3) > 0.0
        @test analytic_noise_psd(0.0) == 1e-30 # DC fallback
        
        # Test Waveform Model (6 parameters)
        freqs = collect(1e-3:1e-4:1e-2)
        theta = [1.0, 2.0, 1.0, 0.0, 0.5, 0.5]
        h = scaled_waveform_model(theta, freqs)
        
        @test length(h) == length(freqs)
        @test eltype(h) <: Complex
    end
    
    @testset "Detector Module" begin
        freqs = collect(1e-3:1e-4:1e-2)
        theta = [1.0, 2.0, 1.0, 0.0, 0.5, 0.5]
        h = scaled_waveform_model(theta, freqs)
        
        A, E, T = project_to_tdi(h, freqs, theta)
        
        @test length(A) == length(freqs)
        @test length(E) == length(freqs)
        @test length(T) == length(freqs)
        # T channel is a null channel at low frequencies, so it should be near zero
        @test sum(abs.(T)) < 1e-10
    end
    
    @testset "Geometry Module" begin
        freqs = collect(1e-3:1e-4:1e-2)
        df = 1e-4
        Sn = analytic_noise_psd.(freqs)
        
        h1 = scaled_waveform_model([1.0, 2.0, 1.0, 0.0, 0.0, 0.0], freqs)
        h2 = scaled_waveform_model([1.0, 2.0, 1.0, 0.0, 0.0, 0.0], freqs)
        
        # Inner product of a vector with itself should be strictly positive
        ip = inner_product(h1, h2, Sn, df)
        @test ip > 0.0
        
        # Decoupled Extrinsic Curvature Calculation
        theta_0 = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
        u_dir = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
        
        basis = compute_tangent_basis(theta_0, freqs, Sn, df)
        @test length(basis) > 0
        
        K_u, g_uu = compute_extrinsic_curvature_from_basis(theta_0, u_dir, basis, freqs, Sn, df)
        
        @test K_u >= 0.0 # Curvature is a squared norm
        @test g_uu > 0.0
    end
    
    @testset "Inference Module" begin
        freqs = collect(1e-3:1e-4:1e-2)
        df = 1e-4
        Sn = analytic_noise_psd.(freqs)
        
        # Perfect match should yield near-zero distance
        theta_true = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
        h_true = scaled_waveform_model(theta_true, freqs)
        data = project_to_tdi(h_true, freqs, theta_true)
        
        theta_guess = [1.0, 2.0, 1.0, 0.0, 0.0, 0.0]
        dist, best_fit, _ = calculate_numerical_distance(data, theta_guess, freqs, Sn, df; iterations=10)
        
        @test dist < 1e-5
    end
    
end
