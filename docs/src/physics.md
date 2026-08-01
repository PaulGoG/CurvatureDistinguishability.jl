# Deep Dive: Physics and Waveform Modeling
**Context:** LISA Two-Source Distinguishability Simulation

This document provides a highly detailed, equation-level breakdown of the physical models and signal generation architecture used in the `CurvatureDistinguishability` pipeline. 

While full Numerical Relativity (NR) or complete Effective One-Body (EOB) waveforms are computationally prohibitive for millions of geometric manifold evaluations, our pipeline utilizes a customized, highly optimized **Frequency-Domain TaylorF2-analog** inspiral model. This model selectively incorporates the exact non-linear physical effects—such as spin-orbit coupling, higher harmonics, and orbital Doppler shifts—that are most critical for testing parameter degeneracies and the Extrinsic Curvature of the signal manifold.

---

## 1. The Core State Vector
The waveform generator maps a 6-dimensional parameter vector $\vec{\theta}$ into the frequency domain. To maintain a well-conditioned Fisher Information Matrix, the internal engine evaluates parameters scaled to $\mathcal{O}(1)$.

1. **Amplitude ($A$)**: Scales the overall signal strain (typically $\sim 10^{-21}$).
2. **Chirp Mass ($\mathcal{M}$)**: The primary mass parameter driving the frequency evolution.
3. **Time of Coalescence ($t_c$)**: The merger time, acting as a linear phase shift across the frequency band.
4. **Coalescence Phase ($\phi_c$)**: The absolute orbital phase at merger.
5. **Primary Spin ($\chi_1$)**: Dimensionless aligned spin of the primary mass $[-1, 1]$.
6. **Secondary Spin ($\chi_2$)**: Dimensionless aligned spin of the secondary mass $[-1, 1]$.

For the internal calculations, we assume a symmetric mass ratio $\eta = 0.25$ (equal mass system), which allows us to simplify the effective spin parameter to $\chi_{\mathrm{eff}} = \frac{1}{2}(\chi_1 + \chi_2)$.

---

## 2. Intrinsic Waveform: Phase Evolution & Spin-Orbit Coupling
The base waveform is generated in the frequency domain using the Stationary Phase Approximation (SPA). 

### The Dominant Quadrupole Mode ($l=2, m=2$)
The amplitude of the dominant 22-mode scales purely with the Newtonian leading order:
$$ \tilde{h}_{22}(f) = A \cdot f^{-7/6} e^{i \Psi_{22}(f)} $$

The phase $\Psi_{22}(f)$ is where the non-linear physics occurs. It is expanded as a Post-Newtonian (PN) series. Our model includes the leading Newtonian term ($0$PN) and the **Spin-Orbit Coupling term ($1.5$PN)**. 

Defining the orbital velocity parameter $v = (\pi \mathcal{M} f)^{1/3}$, the phase is computed as:
$$ \Psi_{22}(f) = 2\pi f t_c - \phi_c - \frac{3}{128} v^{-5} \left[ 1 - 4\beta v^3 \right] $$

Where the spin-orbit coupling coefficient $\beta$ is defined as:
$$ \beta = \frac{1}{4} \left( \frac{113}{3} - \frac{76}{3}\eta \right) \chi_{\mathrm{eff}} $$

**Physical Significance:** If $\chi_{\mathrm{eff}}$ is positive (spins aligned with orbital angular momentum), the "hang-up" effect occurs. The binary takes longer to merge, which slows down the rate of phase accumulation. This highly non-linear modification is critical for breaking the degeneracy between Chirp Mass and Time.

---

## 3. Higher-Order Harmonics ($l=3, m=3$)
A waveform containing only the 22-mode suffers from severe parameter degeneracies. To make the "Zone of Confusion" mapping realistic, we inject the first sub-dominant harmonic: the $33$-mode.

The frequency of the $33$-mode evolves exactly $1.5\times$ faster than the $22$-mode. Therefore, at a given frequency bin $f$, the phase of the $33$-mode is exactly:
$$ \Psi_{33}(f) = 1.5 \cdot \Psi_{22}(f) $$

Its amplitude is suppressed relative to the dominant mode by a factor proportional to the orbital velocity $v$ and a phenomenological scaling constant (e.g., $10\%$):
$$ |\tilde{h}_{33}(f)| = 0.1 \cdot A \cdot f^{-7/6} \cdot v $$

The total source-frame strain is the linear superposition:
$$ \tilde{h}_{\mathrm{source}}(f) = \tilde{h}_{22}(f) + \tilde{h}_{33}(f) $$

**Physical Significance:** The frequency asymmetry between these two modes means that a change in $t_c$ or $\mathcal{M}$ affects the two modes differently. A single-source template attempting to mimic two distinct sources cannot simultaneously match the phase evolution of both the 22 and 33 modes, forcing a massive increase in the Extrinsic Curvature $K(u)$.

---

## 4. Detector Dynamics: Orbit and Doppler Modulation
LISA is not a stationary detector; it is a cartwheeling constellation orbiting the Sun at 1 AU ($R_{\mathrm{orbit}} \approx 499$ light-seconds) with an orbital angular velocity $\Omega_{\mathrm{orbit}} = 2\pi / 1 \text{ year}$.

To project the source-frame strain into the detector, we must calculate where the detector is in its orbit *at the exact moment* a specific frequency $f$ arrives. 

### The Time-Frequency Relation $t(f)$
Using the SPA, the time at which the binary emits GWs at frequency $f$ is the derivative of the phase. In our model, this is approximated as:
$$ t(f) = t_c - \frac{5 \mathcal{M}}{256 v^8} $$

### The Doppler Phase Shift
The detector's motion toward or away from the source causes a time-dependent phase shift. Given the source's Ecliptic Colatitude ($\theta_{\mathrm{sky}}$) and Longitude ($\phi_{\mathrm{sky}}$), the orbital phase of the detector is $\Phi_{\mathrm{orb}}(f) = \Omega_{\mathrm{orbit}} t(f)$. The Doppler shift applied to the waveform is:
$$ \Delta\Phi_{\mathrm{Doppler}}(f) = 2\pi f R_{\mathrm{orbit}} \sin(\theta_{\mathrm{sky}}) \cos(\Phi_{\mathrm{orb}}(f) - \phi_{\mathrm{sky}}) $$

---

## 5. Antenna Patterns & TDI Projection
As the detector orbits, it also cartwheels, changing its sensitivity to the "plus" ($+$) and "cross" ($\times$) polarizations of the wave. 

The intrinsic polarizations are defined by the orbital inclination ($\iota$):
$$ A_+ = \frac{1}{2}(1 + \cos^2\iota), \quad A_\times = \cos\iota $$

The low-frequency envelope of the LISA antenna patterns $F_+$ and $F_\times$ are highly oscillatory functions of the detector's orbital phase $\Phi_{\mathrm{orb}}$ and the source's polarization angle $\psi$:
$$ F_+(f) = \frac{1}{2}(1 + \cos^2\theta_{\mathrm{sky}}) \cos(2\Phi_{\mathrm{orb}}) \cos(2\psi) - \cos(\theta_{\mathrm{sky}}) \sin(2\Phi_{\mathrm{orb}}) \sin(2\psi) $$
*(And similarly for $F_\times$)*.

### Time Delay Interferometry (TDI) Channels
Finally, the modulated strain is projected into the noise-orthogonal A, E, and T channels. In the low-frequency limit, A and E act as two independent $90^\circ$ interferometers rotated by $45^\circ$, and T is the null channel.

$$ \tilde{h}_A(f) = \frac{\sqrt{3}}{2} \left[ F_+(f) A_+ - i F_\times(f) A_\times \right] e^{i \Delta\Phi_{\mathrm{Doppler}}(f)} \tilde{h}_{\mathrm{source}}(f) $$
$$ \tilde{h}_E(f) = \frac{1}{2} \left[ F_\times(f) A_+ + i F_+(f) A_\times \right] e^{i \Delta\Phi_{\mathrm{Doppler}}(f)} \tilde{h}_{\mathrm{source}}(f) $$
$$ \tilde{h}_T(f) \approx 0 $$

---

## 6. The Noise Profile (Robson et al. 2019)
The inner products computing the Extrinsic Curvature $K(u)$ are weighted by the one-sided PSD $S_n(f)$, implemented exactly from Robson, Cornish & Liu (2019), arXiv:1803.01944:

**Instrumental noise (Eq. 12)** — the Michelson-channel PSD (the sky-averaged $10/3$ response factor of their Eq. 13 is *not* applied, because this pipeline models the antenna response explicitly in `Detector.jl`):
$$ P_n(f) = \frac{P_{\mathrm{OMS}}}{L^2} + 2\left(1 + \cos^2(f/f_*)\right)\frac{P_{\mathrm{acc}}}{(2\pi f)^4 L^2} $$
with $P_{\mathrm{OMS}}$ (Eq. 10) the optical-metrology noise and $P_{\mathrm{acc}}$ (Eq. 11) the test-mass acceleration noise.

**Galactic confusion noise (Eq. 14)** — the unresolved white-dwarf foreground:
$$ S_c(f) = A\, f^{-7/3}\, e^{-f^{\alpha} + \beta f \sin(\kappa f)} \left[ 1 + \tanh\!\big(\gamma (f_k - f)\big) \right], \qquad A = 9\times 10^{-45}, $$
with $(\alpha, \beta, \kappa, \gamma, f_k)$ from Table 1, selected by the observation time (1 yr: $\alpha=0.171$, $\beta=292$, $\kappa=1020$, $\gamma=1680$, $f_k=2.15$ mHz) and overridable via the `[noise]` config section.

By utilizing this framework, the pipeline ensures the "Zone of Confusion" mappings are directly applicable to genuine space-based gravitational wave astronomy.