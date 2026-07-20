# Two-Waveform Distinguishability 🌊

## 📂 Project Structure
```text
.
├── config.toml           # The absolute orchestrator configurations
├── benchmarks/
│   └── run_benchmarks.jl # Profilers and timing suite
├── scripts/
│   └── pipeline.jl       # Unified CLI orchestrator entry point
├── src/
│   ├── Detector.jl       # TDI projection & dynamic orbital/antenna modulations
│   ├── Geometry.jl       # Core tensor mathematics (Jacobian, Gram-Schmidt)
│   ├── Hardware.jl       # Dynamic GPU/CPU backend detection
│   ├── Inference.jl      # High-precision optimization (LBFGS)
│   ├── Orchestrator.jl   # Execution and dynamic memory manager
│   ├── Physics.jl        # Analytic PSD and scaled 6-parameter waveform model
│   └── TwoWaveformDistinguishability.jl # Main package module
└── test/
    └── runtests.jl       # Comprehensive test suite
```

This project contains a high-performance Julia simulation pipeline to validate the theoretical quartic scaling law ($\delta^4$) for two-waveform distinguishability and geometrically map the "Zone of Confusion" for the LISA mission.

## 📖 Theoretical Intent

When observing data containing two closely overlapping gravitational wave signals with a small parameter separation $\delta$, a standard approach might try to fit them with a single-source model. A key theoretical result dictates that the squared distance $D^2$ (residual power) between the two-source signal and the best-fit single-source manifold scales **quartically**:

$$ D^2 \approx \frac{A^2}{16} K(u) \delta^4 $$

Where $K(u)$ is the **extrinsic curvature** of the signal manifold in the direction $u$. This project proves this mathematically and Maps the absolute limit of distinguishability ($\delta_{\mathrm{min}}$). 
*For full technical details, see `scientific_context.md`.*

## 🚀 Architecture

The pipeline avoids numerical precision floors (underflow) that plague standard parameter estimation by:
1.  **Operating on an $\mathcal{O}(1)$ Scaled Manifold:** Ensures the Fisher Matrix is perfectly well-conditioned.
2.  **High-Precision AD:** Uses `ForwardDiff.jl` to compute exact analytical gradients and Hessians.
3.  **Hardware Abstraction (HAL):** Uses `KernelAbstractions.jl` to target GPUs (CUDA, AMD, Metal) or fall back to an ultra-fast, allocation-free multi-threaded CPU loop to avoid GC lock contention.
4.  **Realistic Physics:** Includes Spin-Orbit coupling, Higher Harmonics ($33$ mode), and full Time Delay Interferometry (TDI) with Orbital Doppler shifts.

## ⚙️ Usage

The entire project is controlled via the `config.toml` file. You define your physical parameters, the 1D parameter sweeps you want to run, and the 2D contour maps you want to draw.

Because the path resolution is fully dynamic, you can execute the unified pipeline from anywhere:

```bash
julia --threads auto TwoWaveformDistinguishability/scripts/pipeline.jl
```

*(Note: The pipeline automatically suppresses package activation logs and provides a beautiful, aesthetic real-time wall-clock in your terminal).*

### Options:
*   `--config`: Path to your TOML configuration file (default: `config.toml`).
*   `--workers`: Number of distributed cluster nodes to spawn (default: `0` for local multi-threading).

## 📊 Visual Outputs

Outputs are cleanly vaulted into unique `data/outputs/run_<ID>` directories, separated into `/sweeps/` and `/maps/`.

1.  **`scaling_plot.png` (1D Sweep):**
    *   A log-log plot proving the residual signal energy shrinks quartically, perfectly tracking the theoretical $\delta^4$ curve.
2.  **`residual_plot.png` (1D Sweep):**
    *   A 2-layered plot showing the true two-source signal vs. the best-fit single source, and isolating the unabsorbable Extrinsic Curvature residual on its own linear scale.
3.  **`confusion_zone.png` (2D Map):**
    *   A continuous, filled ellipse showing the Fundamental Discernibility boundary ($\delta_{\mathrm{min}}$) for a given parameter plane (e.g., Mass vs. Time). Any secondary source inside this red zone is operationally indistinguishable from the primary source.
