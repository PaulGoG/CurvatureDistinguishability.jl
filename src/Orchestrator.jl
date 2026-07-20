module Orchestrator

using Printf
using Plots
using ProgressMeter
using CSV
using DataFrames
using Dates
using Random
using LaTeXStrings
using SHA
using TOML
using KernelAbstractions
using Statistics

using ..Hardware
using ..Physics
using ..Detector
using ..Geometry
using ..Inference

export run_pipeline

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
function format_time(seconds)
    m, s = divrem(seconds, 60)
    h, m = divrem(m, 60)
    return @sprintf("%02d:%02d:%02d", h, m, s)
end

function log_and_print(msg, logfile)
    println(msg)
    flush(stdout)
    write(logfile, msg * "\n")
    flush(logfile)
end

# -----------------------------------------------------------------------------
# MODULE 1: 1D Parameter Sweeps (Delta^4 Validation)
# -----------------------------------------------------------------------------
function run_1d_sweeps_module(sweeps, out_base_dir, freqs, Sn_vals, df, phys_kwargs, sweep_settings, start_time_total, active_threads, backend)
    n_deltas = get(sweep_settings, "n_deltas", 20)
    min_log_delta = get(sweep_settings, "min_log_delta", -4.5)
    max_log_delta = get(sweep_settings, "max_log_delta", -0.5)
    
    println("\n>>> INITIATING 1D PARAMETER SWEEPS ($(length(sweeps)) configurations)")
    
    for (s_idx, sweep) in enumerate(sweeps)
        sweep_name = sweep["name"]
        theta_0 = Float64.(sweep["theta_0"])
        raw_dir = Float64.(sweep["u_dir"])
        
        out_dir = joinpath(out_base_dir, "sweeps", sweep_name)
        mkpath(out_dir)
        
        log_io = open(joinpath(out_dir, "sweep.log"), "w")
        start_time_sweep = time()
        
        log_and_print("\n" * "─"^80, log_io)
        log_and_print("  🚀 [Sweep $s_idx/$(length(sweeps))] : $sweep_name", log_io)
        log_and_print("  ▶ Base Parameters : $theta_0", log_io)
        log_and_print("  ▶ Direction Vector: $raw_dir", log_io)
        log_and_print("─"^80, log_io)
        
        log_and_print("\n  [1/3] Computing Manifold Geometry (Theoretical K(u))...", log_io)
        K_u, g_uu = compute_extrinsic_curvature(theta_0, raw_dir, freqs, Sn_vals, df; phys_kwargs...)
        
        if g_uu < 1e-12
            log_and_print("  [⚠ WARNING] The chosen direction is completely degenerate (Fisher Norm ≈ 0).", log_io)
            log_and_print("  [⚠ WARNING] The sources cannot be separated in this direction. Skipping sweep.", log_io)
            close(log_io)
            continue
        end
        scale_factor = 1.0 / sqrt(g_uu)
        u_norm = raw_dir .* scale_factor
        K_u_norm = K_u * (scale_factor^4)
        
        log_and_print(@sprintf("        Fisher Norm g(u,u)                 : %.5f", 1.0), log_io)
        log_and_print(@sprintf("        Extrinsic Curvature K(u)           : %.5e", K_u_norm), log_io)
        
        rho_sq_thresh = 1.0 
        delta_min = ( (16.0 * rho_sq_thresh) / K_u_norm )^(1/4)
        log_and_print(@sprintf("        Fundamental Discernibility (δ_min) : %.5e", delta_min), log_io)
        
        deltas = 10 .^ range(min_log_delta, stop=max_log_delta, length=n_deltas)
        D2_num = zeros(n_deltas)
        D2_theo = zeros(n_deltas)
        max_idx = n_deltas
        
        log_and_print("\n  [2/3] Running Optimization Sweep across parameter separation...", log_io)
        
        p = Progress(n_deltas; desc="        Optimizing: ", showspeed=false, barlen=40)
        
        best_fit_largest_delta = zeros(length(theta_0))
        data_stream_largest_delta_A = zeros(ComplexF64, length(freqs))

        # Memory Throttling: Use asyncmap with ntasks limit to smoothly cap concurrency.
        asyncmap(1:n_deltas; ntasks=active_threads) do i
            fetch(Threads.@spawn begin
                d = deltas[i]
                p1 = theta_0
                p2 = theta_0 .+ d .* u_norm
                h1 = scaled_waveform_model(p1, freqs; phys_kwargs...)
                h2 = scaled_waveform_model(p2, freqs; phys_kwargs...)
                A1, E1, T1 = project_to_tdi(h1, freqs, p1; phys_kwargs...)
                A2, E2, T2 = project_to_tdi(h2, freqs, p2; phys_kwargs...)
                data_stream_A = A1 .+ A2
                data_stream_E = E1 .+ E2
                data_stream_T = T1 .+ T2
                data_stream = (data_stream_A, data_stream_E, data_stream_T)
                
                p_guess = theta_0 .+ (0.5 * d) .* u_norm
                p_guess[1] *= 2.0 
                
                dist, best_fit, _ = calculate_numerical_distance(data_stream, p_guess, freqs, Sn_vals, df; g_tol=1e-12, backend=backend, phys_kwargs...)
                
                D2_num[i] = dist
                D2_theo[i] = (1.0 / 16.0) * K_u_norm * (d^4)
                
                if i == max_idx
                    best_fit_largest_delta .= best_fit
                    data_stream_largest_delta_A .= data_stream_A
                end

                current_total_time = format_time(floor(Int, time() - start_time_total))
                msg1 = @sprintf("δ = %.2e | Num D²: %.2e | Theo D²: %.2e", d, dist, D2_theo[i])
                msg2 = "$current_total_time"
                next!(p; showvalues=[(:Latest, msg1), (:Uptime, msg2)])
            end)
        end
        
        log_and_print("\n  [3/3] Saving outputs and generating publication-grade plots...", log_io)
        
        df_results = DataFrame(Delta = deltas, D2_Numerical = D2_num, D2_Theoretical = D2_theo, K_u_Norm = fill(K_u_norm, n_deltas))
        CSV.write(joinpath(out_dir, "results.csv"), df_results)
        
        min_x_pow = floor(Int, log10(minimum(deltas)))
        max_x_pow = ceil(Int, log10(maximum(deltas)))
        x_ticks = 10.0 .^ (min_x_pow:max_x_pow)
        x_labels = [latexstring("10^{$(Int(log10(v)))}") for v in x_ticks]

        min_y_pow = floor(Int, log10(minimum(vcat(D2_num, D2_theo))))
        max_y_pow = ceil(Int, log10(maximum(vcat(D2_num, D2_theo))))
        y_step = max(1, round(Int, (max_y_pow - min_y_pow) / 6))
        y_ticks = 10.0 .^ (min_y_pow:y_step:max_y_pow)
        y_labels = [latexstring("10^{$(Int(log10(v)))}") for v in y_ticks]

        plt1 = plot(deltas, D2_num, xscale=:log10, yscale=:log10, seriestype=:scatter, label="Numerical Optimization", marker=:circle, markersize=5, color=:blue, legend=:topleft, xlabel=latexstring("\\mathrm{Parameter\\ Separation}\\ \\mathrm{\\delta}"), ylabel=latexstring("\\mathrm{Squared\\ Distance}\\ \\mathrm{D}^2"), title="Quartic Scaling: $(replace(sweep_name, "_" => " "))", dpi=300, framestyle=:box, grid=:both, gridstyle=:dash, gridalpha=0.3, gridcolor=:gray, xticks=(x_ticks, x_labels), yticks=(y_ticks, y_labels), minorticks=false, fontfamily="Computer Modern", tickfontsize=10, guidefontsize=12, legendfontsize=10, titlefontsize=14, margin=8Plots.mm)
        plot!(plt1, deltas, D2_theo, label=latexstring("\\mathrm{Theoretical\\ Prediction}\\ \\propto \\mathrm{\\delta}^4"), linestyle=:dash, color=:red, lw=2.5)
        savefig(plt1, joinpath(out_dir, "scaling_plot.png"))

        h_best_fit = scaled_waveform_model(best_fit_largest_delta, freqs; phys_kwargs...)
        A_best_fit, _, _ = project_to_tdi(h_best_fit, freqs, best_fit_largest_delta; phys_kwargs...)
        residual_A = abs.(data_stream_largest_delta_A .- A_best_fit)
        
        noise_amplitude = sqrt.(Sn_vals ./ (4.0 * df))
        sig_mag = abs.(data_stream_largest_delta_A) ./ noise_amplitude
        bf_mag = abs.(A_best_fit) ./ noise_amplitude
        res_mag = residual_A ./ noise_amplitude
        
        # Calculate smooth plotting envelope to prevent GR backend ribbon-aliasing
        window = max(1, length(freqs) ÷ 1500)
        n_windows = length(freqs) ÷ window
        f_plot = [mean(freqs[((i-1)*window + 1):(i*window)]) for i in 1:n_windows]
        
        function get_envelope(arr)
            e_mean = [mean(arr[((i-1)*window + 1):(i*window)]) for i in 1:n_windows]
            e_min = [minimum(arr[((i-1)*window + 1):(i*window)]) for i in 1:n_windows]
            e_max = [maximum(arr[((i-1)*window + 1):(i*window)]) for i in 1:n_windows]
            return e_mean, e_mean .- e_min, e_max .- e_mean
        end
        
        sig_m, sig_l, sig_u = get_envelope(sig_mag)
        bf_m, bf_l, bf_u = get_envelope(bf_mag)
        res_m, res_l, res_u = get_envelope(res_mag)
        
        p2_top = plot(f_plot, sig_m, ribbon=(sig_l, sig_u), fillalpha=0.3, label="Two-Source Superposition", lw=2.0, color=:black, ylabel=latexstring("\\mathrm{SNR\\ Density}"), title="Largest Separation: Signal vs Best-Fit", framestyle=:box, grid=:both, gridstyle=:dash, gridalpha=0.3, gridcolor=:gray, minorticks=false, fontfamily="Computer Modern", legend=:topright, tickfontsize=10, guidefontsize=12, legendfontsize=10, titlefontsize=14, margin=8Plots.mm)
        plot!(p2_top, f_plot, bf_m, ribbon=(bf_l, bf_u), fillalpha=0.3, label="Best-Fit Single Source", lw=2.0, linestyle=:dash, color=:dodgerblue)

        p2_bottom = plot(f_plot, res_m, ribbon=(res_l, res_u), fillalpha=0.4, label="Unabsorbed Residual", lw=2.0, color=:crimson, xlabel=latexstring("\\mathrm{Frequency\\ [Hz]}"), ylabel=latexstring("\\mathrm{Residual\\ SNR}"), framestyle=:box, grid=:both, gridstyle=:dash, gridalpha=0.3, gridcolor=:gray, minorticks=false, fontfamily="Computer Modern", legend=:topright, tickfontsize=10, guidefontsize=12, legendfontsize=10, margin=8Plots.mm)

        plt2 = plot(p2_top, p2_bottom, layout=(2, 1), size=(800, 600), dpi=300)
        savefig(plt2, joinpath(out_dir, "residual_plot.png"))
        
        elapsed_sweep = time() - start_time_sweep
        log_and_print("\n  [✓] Sweep '$sweep_name' completed successfully in $(format_time(floor(Int, elapsed_sweep))).", log_io)
        close(log_io)
    end
end

# -----------------------------------------------------------------------------
# MODULE 2: 2D Confusion Mapping (Discernibility Ellipses)
# -----------------------------------------------------------------------------
function run_2d_mapping_module(maps, out_base_dir, freqs, Sn_vals, df, phys_kwargs, start_time_total, active_threads, backend)
    println("\n>>> INITIATING 2D CONFUSION MAPPING ($(length(maps)) configurations)")
    flush(stdout)
    
    for (m_idx, map_cfg) in enumerate(maps)
        map_name = map_cfg["name"]
        px = map_cfg["param_x"]
        py = map_cfg["param_y"]
        n_angles = get(map_cfg, "n_angles", 2000)
        rho_thresh = get(map_cfg, "rho_thresh", 1.0)
        rho_sq_thresh = rho_thresh^2
        base_theta = Float64.(map_cfg["theta_0"])
        
        out_dir = joinpath(out_base_dir, "maps", map_name)
        mkpath(out_dir)
        
        log_io = open(joinpath(out_dir, "mapping.log"), "w")
        start_time_map = time()
        
        param_names_raw = ["Amplitude", "Chirp Mass", "Time", "Phase", "Spin 1", "Spin 2"]
        param_names_latex = ["Amplitude", "Chirp\\ Mass", "Time", "Phase", "Spin\\ 1", "Spin\\ 2"]
        
        log_and_print("\n" * "─"^80, log_io)
        log_and_print("  🌐 [Map $m_idx/$(length(maps))] : $map_name", log_io)
        log_and_print("  ▶ Plane     : $(param_names_raw[px]) vs $(param_names_raw[py])", log_io)
        log_and_print("  ▶ Threshold : SNR = $rho_thresh", log_io)
        log_and_print("─"^80, log_io)
        
        angles = range(0, 2π, length=n_angles)
        x_bounds = zeros(n_angles)
        y_bounds = zeros(n_angles)
        K_vals = zeros(n_angles)
        
        log_and_print("\n  [1/2] Computing Orthogonal Tangent Basis (Jacobian)...", log_io)
        tangent_basis = compute_tangent_basis(base_theta, freqs, Sn_vals, df; phys_kwargs...)
        
        log_and_print("\n  [2/2] Starting angular sweep ($n_angles directions)...", log_io)
        
        p = Progress(n_angles; desc="        Mapping: ", showspeed=false, barlen=40)
        
        # Memory Throttling: Smooth asynchronous execution capping concurrency
        asyncmap(1:n_angles; ntasks=active_threads) do i
            fetch(Threads.@spawn begin
                phi = angles[i]
                raw_dir = zeros(6)
                raw_dir[px] = cos(phi)
                raw_dir[py] = sin(phi)
                
                K_u, g_uu = compute_extrinsic_curvature_from_basis(base_theta, raw_dir, tangent_basis, freqs, Sn_vals, df; phys_kwargs...)
                
                if g_uu < 1e-12
                    K_vals[i] = NaN
                    x_bounds[i] = NaN
                    y_bounds[i] = NaN
                else
                    scale_factor = 1.0 / sqrt(g_uu)
                    K_u_norm = K_u * (scale_factor^4)
                    K_vals[i] = K_u_norm
                    delta_min = ( (16.0 * rho_sq_thresh) / K_u_norm )^(1/4)
                    x_bounds[i] = delta_min * raw_dir[px] * scale_factor
                    y_bounds[i] = delta_min * raw_dir[py] * scale_factor
                end
                
                current_total_time = format_time(floor(Int, time() - start_time_total))
                msg2 = "$current_total_time"
                next!(p; showvalues=[(:Uptime, msg2)])
            end)
        end
        
        log_and_print("\n  Saving numerical outputs to CSV...", log_io)
        df_results = DataFrame(Angle=angles, X_Bound=x_bounds, Y_Bound=y_bounds, K=K_vals)
        CSV.write(joinpath(out_dir, "confusion_contour.csv"), df_results)
        
        push!(x_bounds, x_bounds[1])
        push!(y_bounds, y_bounds[1])
        
        log_and_print("  Generating publication-grade contour plot...", log_io)
        plt = plot(x_bounds, y_bounds, seriestype=:shape, fillalpha=0.3, color=:dodgerblue, lw=0, label="Zone of Confusion (Indistinguishable)", xlabel=latexstring("\\mathrm{\\Delta\\ $(param_names_latex[px])}"), ylabel=latexstring("\\mathrm{\\Delta\\ $(param_names_latex[py])}"), title="Fundamental Discernibility Ellipse: $(replace(map_name, "_" => " "))", dpi=300, framestyle=:box, grid=:both, gridstyle=:dash, gridalpha=0.3, gridcolor=:gray, fontfamily="Computer Modern", aspect_ratio=:equal, margin=8Plots.mm)
        savefig(plt, joinpath(out_dir, "confusion_zone.png"))
        
        elapsed_map = time() - start_time_map
        log_and_print("\n  [✓] Map '$map_name' completed successfully in $(format_time(floor(Int, elapsed_map))).", log_io)
        close(log_io)
    end
end

"""
    run_pipeline(config_path::String, project_root::String, output_dir::String)

The unified orchestrator function. Reads the TOML configuration, allocates hardware, 
generates the physics grid, and safely routes processing to the 1D Sweep or 2D Mapping 
modules while enforcing memory safety.
"""
function run_pipeline(config_path::String, project_root::String, output_dir::String)
    if !isfile(config_path)
        error("Configuration file not found: $config_path")
    end
    config = TOML.parsefile(config_path)
    
    # Extract blocks
    pipeline_cfg = get(config, "pipeline", Dict())
    run_1d_sweeps = get(pipeline_cfg, "run_1d_sweeps", true)
    run_2d_mapping = get(pipeline_cfg, "run_2d_mapping", true)
    
    sweep_settings = get(pipeline_cfg, "sweep_settings", Dict())
    grid_cfg = get(config, "grid", Dict())
    phys_cfg = get(config, "physics", Dict())
    sweeps = get(config, "sweeps", [])
    maps = get(config, "maps", [])
    
    # Generate unique run ID
    run_id = "run_" * first(bytes2hex(sha256(string(now()))), 8)
    out_base_dir = joinpath(project_root, output_dir, run_id)
    mkpath(out_base_dir)
    
    start_time_total = time()
    
    hardware_cfg = get(config, "hardware", Dict())
    max_vram_gb = get(hardware_cfg, "max_vram_gb", 8.0)
    max_threads = get(hardware_cfg, "max_threads", Threads.nthreads())
    os_vram_overhead_gb = get(hardware_cfg, "os_vram_overhead_gb", 1.0)
    force_cpu = get(hardware_cfg, "force_cpu", false)
    
    backend = force_cpu ? KernelAbstractions.CPU() : get_best_backend()
    backend_str = if backend isa KernelAbstractions.CPU
        "$(Threads.nthreads()) CPU Threads"
    elseif isdefined(Main, :CUDA) && backend isa Main.CUDA.CUDABackend
        "NVIDIA CUDA GPU"
    elseif isdefined(Main, :AMDGPU) && backend isa Main.AMDGPU.ROCBackend
        "AMD ROCm GPU"
    elseif isdefined(Main, :Metal) && backend isa Main.Metal.MetalBackend
        "Apple Metal GPU"
    elseif isdefined(Main, :oneAPI) && backend isa Main.oneAPI.oneAPIBackend
        "Intel oneAPI GPU"
    else
        "Unknown Accelerator"
    end
    
    # Setup Simulation Grid
    T_obs = get(grid_cfg, "T_obs", 3.15e7) # 1 year
    df = 1.0 / T_obs
    f_min = get(grid_cfg, "f_min", 1.0e-3)
    f_max = get(grid_cfg, "f_max", 0.01)
    
    # Generate and move to backend
    freqs_cpu = collect(f_min:df:f_max)
    Sn_vals_cpu = analytic_noise_psd.(freqs_cpu)
    
    freqs = to_backend(freqs_cpu, backend)
    Sn_vals = to_backend(Sn_vals_cpu, backend)
    
    # Extract physics kwargs
    phys_kwargs = (
        mass_scale = get(phys_cfg, "mass_scale", 10.0),
        time_scale = get(phys_cfg, "time_scale", 1000.0),
        amp_scale = get(phys_cfg, "amp_scale", 1e-21),
        eta = get(phys_cfg, "eta", 0.25),
        amp_33_factor = get(phys_cfg, "amp_33_factor", 0.1),
        sky_theta = get(phys_cfg, "sky_theta", 1.047),
        sky_phi = get(phys_cfg, "sky_phi", 0.0),
        inclination = get(phys_cfg, "inclination", 0.523),
        polarization = get(phys_cfg, "polarization", 0.0)
    )
    
    # Dynamic Memory & Concurrency Manager
    if backend isa KernelAbstractions.CPU
        # CPU uses the optimized sum(1:N) allocation-free loop. 
        # The memory footprint per thread is basically zero, so no throttling is needed.
        safe_concurrency = max_threads
    else
        # For a 6-parameter AD Hessian evaluation via array broadcasting (GPU), 
        # ForwardDiff allocates arrays of Dual{Dual} numbers.
        # 1 ComplexF64 = 16 bytes. A 6-param Complex Hessian Dual is ~1500 bytes.
        # With 3 channels and intermediate broadcast arrays, we conservatively estimate 5000 bytes.
        available_vram_gb = max(1.0, max_vram_gb - os_vram_overhead_gb)
        bytes_per_bin_per_thread = 5000 
        gb_per_thread = (length(freqs) * bytes_per_bin_per_thread) / (1024^3)
        safe_concurrency = max(1, floor(Int, available_vram_gb / gb_per_thread))
    end
    
    active_threads = min(Threads.nthreads(), max_threads, safe_concurrency)
    
    println("================================================================================")
    println("  🌌 Two-Waveform Distinguishability Unified Pipeline")
    println("================================================================================")
    println("  ▶ Run ID           : $run_id")
    println("  ▶ Simulation Grid  : $(length(freqs)) frequency bins ($(f_min) to $(f_max) Hz)")
    if backend isa KernelAbstractions.CPU
        println("  ▶ Allocation-Free  : CPU loop active (Safe Concurrency: Infinite)")
    else
        println("  ▶ GPU VRAM Limit   : $(max_vram_gb) GB (Safe Concurrency: $safe_concurrency tasks)")
    end
    println("  ▶ Compute Backend  : $backend_str (Throttled to $active_threads active threads)")
    println("================================================================================")
    flush(stdout)
    
    if run_1d_sweeps && !isempty(sweeps)
        run_1d_sweeps_module(sweeps, out_base_dir, freqs, Sn_vals, df, phys_kwargs, sweep_settings, start_time_total, active_threads, backend)
    end

    if run_2d_mapping && !isempty(maps)
        run_2d_mapping_module(maps, out_base_dir, freqs, Sn_vals, df, phys_kwargs, start_time_total, active_threads, backend)
    end
    
    total_elapsed = time() - start_time_total
    println("\n" * "="^80)
    println("  ✅ PIPELINE ORCHESTRATION COMPLETE")
    println("  ▶ Total Uptime     : $(format_time(floor(Int, total_elapsed)))")
    println("  ▶ Results Vaulted  : $(relpath(out_base_dir, project_root))/")
    println("="^80)
    flush(stdout)
end

end # module