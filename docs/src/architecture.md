# TwoWaveformDistinguishability: Complete Architectural Context for AI Agents

**Target Audience:** Future AI Agents and Human Collaborators
**Purpose:** This document provides a comprehensive, high-level structural breakdown of the `TwoWaveformDistinguishability` codebase. It serves as an exact state-of-the-union context file to immediately onboard any future AI agent or developer contributing to this project.

---

## 1. Project Objective and Theoretical Foundation
This pipeline computationally proves a fundamental theorem in gravitational wave (GW) data analysis: The squared distance ($D^2$) between a true two-source GW signal and the best-fitting single-source model scales **quartically** ($\delta^4$) with respect to the parameter separation ($\delta$) between the two sources, rather than quadratically ($\delta^2$) as a linear Fisher Information Matrix would suggest.

This happens because the single-source manifold "flexes" to absorb the linear and quadratic differences. The unabsorbable residual lies entirely in the orthogonal normal space, and its magnitude is strictly governed by the **Extrinsic Curvature ($K(u)$)** of the signal manifold in the direction $u$.

**The Primary Formula:**
$$ D^2 \approx \frac{1}{16} K(u) \delta^4 $$
From this, the pipeline calculates the absolute **Fundamental Discernibility Limit ($\delta_{\mathrm{min}}$)**—the exact parameter boundary forming the "Zone of Confusion."

---

## 2. Directory Structure and Modules
The project follows strict Julian `PkgTemplates` standards.

*   **`config.toml`:** The absolute source of truth. Contains `[hardware]`, `[grid]`, `[physics]`, `[[sweeps]]` (1D tests), and `[[maps]]` (2D contours). **There are zero hardcoded physical parameters in the Julia source code.**
*   **`scripts/pipeline.jl`:** The unified CLI orchestrator. It parses the TOML, initializes the simulation grid, and routes to the modules. Use this for live, interactive terminal runs.
*   **`scripts/launch_campaign.jl`:** A robust background runner. It spawns `pipeline.jl` asynchronously (`run(..., wait=false)`), instantly returning terminal control to the user while dynamically tracking and vaulting standard output logs directly into `data/logs/` to prevent buffering issues.
*   **`src/TwoWaveformDistinguishability.jl`:** The primary module file that includes and exports the submodules.
*   **`docs/`:** A complete, locally-built `Documenter.jl` website containing the external markdown notes and automatically-scraped API references. (The HTML builds are stored in `docs/build/index.html` and are `.gitignore`d).

### The 5 Core Sub-Modules
1.  **`Hardware.jl` (The Abstraction Layer):** Uses `KernelAbstractions.jl` to dynamically probe the environment. It detects `CUDA`, `AMDGPU`, `Metal`, or `oneAPI` GPUs. If `force_cpu = true` is set in the TOML, it safely falls back to CPU threading.
2.  **`Physics.jl`:** Contains the Robson et al. (2019) analytic noise PSD (`analytic_noise_psd`) and the core `scaled_waveform_model`. 
    *   *Note on Scaling:* The 6 parameters (Amp, Mass, Time, Phase, Spin1, Spin2) are internally scaled to $\mathcal{O}(1)$ to prevent the LBFGS optimizer from crashing due to an ill-conditioned Fisher matrix.
    *   *Physics:* Includes the 1.5PN Spin-Orbit coupling "hang-up" effect and injects the asymmetric higher harmonic ($l=3, m=3$) mode.
3.  **`Detector.jl`:** Implements Time Delay Interferometry (TDI). Projects the source strain into noise-orthogonal `A`, `E`, and `T` channels. It dynamically applies Doppler phase shifts and Antenna Pattern amplitude modulations based on the detector's orbit.
4.  **`Geometry.jl`:** The mathematical heart. Uses `ForwardDiff.jl` Dual numbers to compute exact analytical derivatives.
    *   `compute_tangent_basis`: Generates the massive Jacobian and orthonormalizes the tangent vectors via multi-channel Gram-Schmidt.
    *   `compute_extrinsic_curvature_from_basis`: Takes the pre-computed basis and calculates the Directional Hessian to find the normal projection ($K(u)$). *This decoupling prevents massive Out-Of-Memory (OOM) crashes.*
5.  **`Inference.jl`:** Uses `Optim.jl` (LBFGS) to numerically find the best-fit single source. Features a highly-optimized, allocation-free `sum(1:N) do i` loop for CPU execution to completely bypass Garbage Collection lock contention across 20+ threads.
6.  **`Orchestrator.jl`:** Houses the execution loops (`run_1d_sweeps_module`, `run_2d_mapping_module`) and the **Dynamic Memory Manager**. It reads the physical RAM from the TOML, estimates the Dual Number memory overhead ($\approx 1000$ bytes/bin), and intelligently throttles `asyncmap` concurrency to protect the system.

---

## 3. The Workflows (1D vs 2D)

### 1D Parameter Sweeps (`[[sweeps]]`)
*   **Purpose:** Mathematically validates the $\delta^4$ theory.
*   **Process:** Injects two identical sources, separates them along a multi-dimensional vector $u$ by a distance $\delta$, and numerically optimizes a single-source template to fit them. 
*   **Output:** Generates a log-log scatter plot (`scaling_plot.png`) proving the numerical $D^2$ perfectly tracks the theoretical Extrinsic Curvature line. It also generates a smooth-envelope `residual_plot.png` visualizing the unabsorbed energy in the frequency domain.

### 2D Confusion Mapping (`[[maps]]`)
*   **Purpose:** Draws the continuous physical boundary of the "Zone of Confusion."
*   **Process:** Sweeps an angle $\phi$ across a 2D parameter plane (e.g., Mass vs. Time). It uses the pre-computed Tangent Basis to evaluate the Directional Hessian $K(u)$ instantly for thousands of angles.
*   **Output:** Generates a continuous, filled red ellipse (`confusion_zone.png`). Any secondary source whose parameters fall inside that ellipse is operationally indistinguishable from the primary source.

---

## 4. Best Practices for Future AI Development
If you are an AI agent tasked with modifying this codebase:
1.  **Do not touch `Inference.jl`'s CPU loop unless absolutely necessary.** It is deliberately un-vectorized (`sum(...) do i`) because allocating arrays inside a 22-thread `ForwardDiff` AD loop will instantly choke the Julia Garbage Collector.
2.  **Respect the Dynamic Memory Manager:** If you add more channels or physics, increase the `bytes_per_bin_per_thread` in `config.toml` so the orchestrator knows to throttle the active threads further to prevent `OOM Killed` OS crashes.
3.  **No Hardcoding:** If you add a new physical parameter (e.g., symmetric mass ratio $\eta$), expose it in `config.toml`, parse it in `pipeline.jl`, and pass it down via `phys_kwargs...`. 
4.  **GPU Compiler Warnings (`oneAPI.jl`):** There is a known bug in experimental GPU compilers (like Intel's `oneAPI.jl`) where passing `ForwardDiff.Dual` arrays via `Ref(p)` throws an `InvalidIRError` (passing non-bitstype argument) because the GPU compiler fails to statically infer the deeply nested derivatives. Until `oneAPI.jl` matures, massive AD runs should rely on the allocation-free CPU loop (`force_cpu = true`). Mature compilers like NVIDIA's `CUDA.jl` should handle the AD broadcasting natively.

This pipeline is currently highly-optimized, 100% thread-safe, mathematically rigorous, and ready for publication-level data generation.