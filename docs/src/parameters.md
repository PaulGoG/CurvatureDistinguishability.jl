# Complex Run Parameters Documentation
**Context:** LISA Two-Waveform Distinguishability Simulation

This document outlines the theoretical rationale and parameter choices for the ultimate 6-dimensional and high-resolution 2D mapping configurations executed by the pipeline. It is meant to be a companion piece for interpreting the data generated during the massive computational stress test.

## 1. Global Simulation Grid
To push the theory to its limits and validate the code's computational capacity, the grid was vastly expanded:
*   **Observation Time ($T_{\mathrm{obs}}$):** $3.15 \times 10^7$ seconds (1 year).
*   **Frequency Range ($f_{\mathrm{min}}$ to $f_{\mathrm{max}}$):** $10^{-4}$ Hz to $0.05$ Hz.
*   **Resolution:** At a spacing of $1/T_{\mathrm{obs}}$, this grid spans exactly **1,577,880 frequency bins**. This dense array forces the Automatic Differentiation (AD) engine to process massive Dual Number arrays, rigorously testing the system's memory allocation and multi-threading safety.

## 2. 1D Parameter Separation Sweeps

The pipeline validates the fundamental geometric distance $D^2 \propto \delta^4$ across 4 distinct physical scenarios.

### Scenario A: The 6-Dimensional Diagonal Stress
*   **Base Source ($\theta_0$):** `[1.0, 1.25, 300.0, 1.57, 0.9, -0.4]`
*   **Direction ($u$):** `[0.0, 0.707, -0.5, 0.3, 0.4, -0.1]`
*   **Rationale:** Instead of moving along a single parameter axis, this injection forces the second black hole to simultaneously become heavier, merge earlier, have a different phase, spin faster on its primary axis, and slow down on its secondary axis. 
*   **Significance:** This tests the true cross-coupling of the Fisher Information Matrix. It proves that no matter how complex the multi-dimensional trajectory is, the residual unabsorbable energy is strictly governed by the Extrinsic Curvature of the signal manifold.

### Scenario B: Standard LDC "Radler" MBHB
*   **Base Source ($\theta_0$):** `[1.0, 1.5, 200.0, 0.0, 0.8, 0.8]`
*   **Direction ($u$):** `[0.0, 1.0, 0.5, 0.0, 0.0, 0.0]`
*   **Rationale:** Simulates a standard $\sim 3\times 10^6 M_\odot$ equal-mass binary merging 23 days into the observation with highly aligned spins. It represents the quintessential, loud LISA detection.

### Scenario C: Extreme Spin-Orbit Coupling
*   **Base Source ($\theta_0$):** `[1.0, 2.0, 150.0, 1.57, 0.95, -0.95]`
*   **Direction ($u$):** `[0.0, 0.0, 0.0, 1.0, 0.5, 0.5]`
*   **Rationale:** The base state features maximal, anti-aligned spins ($\chi_1 = 0.95, \chi_2 = -0.95$). The separation sweeps the absolute phase and spin dimensions.
*   **Significance:** Maximizes the non-linear 1.5PN "hang-up" effect. It visually demonstrates how spin twists the signal manifold, generally breaking classical degeneracies.

### Scenario D: EMRI Analog Time-Shift
*   **Base Source ($\theta_0$):** `[0.5, 0.1, 500.0, 0.0, 0.9, 0.0]`
*   **Direction ($u$):** `[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]`
*   **Rationale:** A very low-mass system ($\sim 2 \times 10^5 M_\odot$) with asymmetric spin, analogous to an Extreme Mass Ratio Inspiral (EMRI). It merges very late in the year. The direction is a pure time shift.
*   **Significance:** Tests the Time Delay Interferometry (TDI) and Doppler shift mechanics. Shifting a signal in time means the detector is at a different physical location in its orbit around the Sun, resulting in different antenna patterns and Doppler phases.

## 3. 2D Zone of Confusion Mappings

These configurations evaluate the Extrinsic Curvature across an entire 2D plane to draw the continuous $\delta_{\mathrm{min}}$ boundary contour.

1.  **Mass vs. Phase (`param_x = 2, param_y = 4`)**
    *   Maps the notorious degeneracy between the overall chirp rate and the absolute orbital phase. 
2.  **Spin 1 vs. Spin 2 (`param_x = 5, param_y = 6`)**
    *   Maps the spin-orbit coupling plane. Because the waveform relies heavily on the *effective spin* $\chi_{\mathrm{eff}} = \frac{1}{2}(\chi_1 + \chi_2)$, this maps exactly how indistinguishable two sources are if one black hole speeds up its spin while the other slows down.
3.  **Time vs. Phase (`param_x = 3, param_y = 4`)**
    *   Maps the geometry of absolute time-translation versus orbital phase. This is the ultimate test of the orbital Doppler effect, as purely shifting the time alters the geometric arrival of the signal on the LISA cartwheel.