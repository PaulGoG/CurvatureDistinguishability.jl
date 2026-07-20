# Complex Run Parameters Documentation
**Context:** LISA Two-Waveform Distinguishability Simulation

This document outlines the theoretical rationale and parameter choices for the ultimate 6-dimensional and high-resolution 2D mapping configurations executed by the pipeline. It is meant to be a companion piece for interpreting the data generated during the massive computational stress test.

## 1. Global Simulation Grid
To push the theory to its limits and validate the code's computational capacity, the grid was vastly expanded:
*   **Observation Time ($T_{\mathrm{obs}}$):** $3.15576 \times 10^7$ seconds (1 Julian year — the same constant that drives the orbital Doppler modulation).
*   **Frequency Range ($f_{\mathrm{min}}$ to $f_{\mathrm{max}}$):** $10^{-4}$ Hz to $0.05$ Hz.
*   **Resolution:** at a spacing of $1/T_{\mathrm{obs}}$ this grid spans $\approx 1.57$ million frequency bins. The `[safety].max_ram_gb` budget is checked against this size before anything is allocated.

## 2. 1D Parameter Separation Sweeps

The pipeline validates the fundamental geometric distance $D^2 \propto \delta^4$ across 4 distinct physical scenarios.

### Scenario A: The 6-Dimensional Diagonal Stress
*   **Base Source ($\theta_0$):** `[1.0, 1.25, 300.0, 1.57, 0.9, -0.4]`
*   **Direction ($u$):** `[0.0, 0.707, -0.5, 0.3, 0.4, -0.1]`
*   **Rationale:** Instead of moving along a single parameter axis, this injection forces the second black hole to simultaneously become heavier, merge earlier, have a different phase, spin faster on its primary axis, and slow down on its secondary axis. 
*   **Significance:** This tests the true cross-coupling of the Fisher Information Matrix. It proves that no matter how complex the multi-dimensional trajectory is, the residual unabsorbable energy is strictly governed by the Extrinsic Curvature of the signal manifold.

### Scenario B: Massive Binary, Mass–Time Direction (`massive_binary_mass_time`)
*   **Base Source ($\theta_0$):** `[1.0, 1.5, 200.0, 0.0, 0.8, 0.8]`
*   **Direction ($u$):** `[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]`
*   **Rationale:** a representative massive ($\sim 3\times 10^6 M_\odot$ via `mass_scale`) equal-mass binary with strongly aligned spins, merging weeks into the observation; the chirp-rate/arrival-time direction dominates realistic overlapping-source populations.

### Scenario C: Extreme Spin-Orbit Coupling
*   **Base Source ($\theta_0$):** `[1.0, 2.0, 150.0, 1.57, 0.95, -0.95]`
*   **Direction ($u$):** `[0.0, 0.0, 0.0, 1.0, 0.5, 0.5]`
*   **Rationale:** The base state features maximal, anti-aligned spins ($\chi_1 = 0.95, \chi_2 = -0.95$). The separation sweeps the absolute phase and spin dimensions.
*   **Significance:** Maximizes the non-linear 1.5PN "hang-up" effect. It visually demonstrates how spin twists the signal manifold, generally breaking classical degeneracies.

### Scenario B': Unequal-Amplitude Pair (`unequal_amplitude_mass_time`)
*   **Base Source ($\theta_0$):** as Scenario B; **Direction ($u$):** as Scenario B; **`amp_ratio` $q = 1/2$.**
*   **Rationale:** the second source carries half the amplitude of the first. The theoretical prediction acquires the harmonic-mean amplitude prefactor, $D^2 = \tfrac{1}{16}K_{\rm num}\delta^4\,(2q/(1+q))^2$, and the optimizer is initialized at the weighted midpoint with effective amplitude $(1+q)A$.
*   **Significance:** validates the unequal-amplitude branch of the distinguishability law ($A_{\rm eff}=A_{\rm harm}$) at the level of the prefactor — with $q=1/2$, using the equal-amplitude law instead would misplace the ratio panel by a factor $9/4$.

### Scenario D: Low-Mass Time Shift (`low_mass_time_shift`)
*   **Base Source ($\theta_0$):** `[0.5, 0.1, 500.0, 0.0, 0.9, 0.0]`
*   **Direction ($u$):** `[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]`
*   **Rationale:** a low-mass system ($\sim 2 \times 10^5 M_\odot$) with a single spinning component, merging late in the observation. The direction is a pure coalescence-time shift.
*   **Significance:** isolates the Time Delay Interferometry (TDI) and Doppler mechanics — shifting the signal in time places the detector at a different point of its solar orbit at every contributing frequency, changing antenna patterns and Doppler phases.

## 3. 2D Zone of Confusion Mappings

These configurations evaluate the extrinsic curvature over a 2D plane to draw the $\delta_{\mathrm{min}}$ boundary. The angular resolution comes from the global `[mapping].n_angles` (per-map override allowed); the solver computes half the directions and mirrors ($K$ is exactly even), refining adaptively near boundary spikes, and caps every direction at the physical prior box (`Prior_Limited` flags in the CSV).

1.  **Mass vs. Time (`mass_vs_time_degeneracy`, 2 vs 3)** — the classic chirp-rate / arrival-time degeneracy; base $\theta_0 = [1.0, 1.5, 200.0, 0.0, 0.8, 0.8]$.
2.  **Spin 1 vs. Spin 2 (`spin1_vs_spin2_coupling`, 5 vs 6)** — the waveform sees only $\chi_{\mathrm{eff}} = \frac{1}{2}(\chi_1 + \chi_2)$, so the anti-symmetric direction is exactly flat: the mathematical zone diverges there and the published zone is limited by the spin prior $[-1, 1]$ (a wedge, not an ellipse).
3.  **Mass vs. Spin 1 (`mass_vs_spin1_twist`, 2 vs 5)** — mass/spin phasing trade-off; the zone crosses the $\chi_1 \le 1$ bound and is capped there.
4.  **Time vs. Phase (`time_vs_phase_doppler`, 3 vs 4)** — absolute time translation vs orbital phase; the ultimate Doppler test. The pure-phase direction is quasi-degenerate ($\partial^2_\varphi h \parallel$ tangent space), producing a near-singular boundary spike that the adaptive refinement resolves.
5.  **Mass vs. Phase, low-mass system (`mass_vs_phase_low_mass`, 2 vs 4)** — the chirp-rate/phase degeneracy at the low-mass base $\theta_0 = [0.5, 0.1, 500.0, 0.0, 0.9, 0.0]$.