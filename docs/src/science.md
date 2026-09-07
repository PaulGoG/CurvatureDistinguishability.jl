# Scientific Context: Two-Source Distinguishability in LISA

Theoretical companion to the `CurvatureDistinguishability.jl` pipeline: the
geometric formulation, the discernibility criterion and the modelling
assumptions behind the computed results.

## 1. Theoretical Background: The Geometry of Source Confusion

When observing a data stream containing two distinct, closely overlapping gravitational wave (GW) signals (a "two-source" hypothesis) with a small parameter separation ``\delta``, standard analysis pipelines often attempt to fit them using a single-source model. 

Standard Fisher Information Matrix (linear) approximations suggest that the residual power—the squared distance ``D^2`` between the true two-source data and the best-fit single-source manifold—should scale quadratically (``\delta^2``).

However, the mathematical backbone of this project demonstrates a non-linear geometric reality: the manifold of single-source waveforms ``\mathcal{M}_1`` is "flexible." Because an optimizer is allowed to adjust the single source's parameters to fit the data, it perfectly absorbs the linear (``\mathcal{O}(\delta)``) and quadratic (``\mathcal{O}(\delta^2)``) differences (assuming equal amplitudes).

The unabsorbable residual lies strictly in the normal space orthogonal to the tangent space of the signal manifold. The magnitude of this orthogonal projection is governed by the **Extrinsic Curvature** ``K(u)`` of the manifold in the direction of the parameter separation ``u``. 

Therefore, the distance ``D^2`` scales **quartically**:
```math
D^2 \approx \frac{1}{16} K(u) \delta^4
```
*(Note: The amplitude ``A^2`` is absorbed natively into the inner product of ``K(u)`` in this pipeline).*

## 2. The Discernibility Limit (``\delta_{\mathrm{min}}``)

The criterion is a residual signal-to-noise ratio: two sources count as
distinguishable from a single one when the unabsorbable residual power
exceeds a threshold ``\rho_{\mathrm{threshold}}``,

```math
D^2 \ge \rho_{\mathrm{threshold}}^2
```
Substituting the geometric scaling law, we can analytically solve for the absolute mathematical limit of discernibility (the "Zone of Confusion" boundary):

```math
\delta_{\mathrm{min}}(u) = \left( \frac{16 \cdot \rho_{\mathrm{threshold}}^2}{K(u)} \right)^{1/4}
```
Because of the ``1/4`` exponent, lowering ``\rho_{\mathrm{threshold}}`` yields strongly
diminishing returns for resolving overlapping sources. The shipped
``\rho_{\mathrm{threshold}} = 1`` is the leading-order resolvability scale, not a
detection-significance threshold: with the likelihood identity
``\Delta\ln\mathcal{L}_{\max} = D^2/2`` it corresponds to ``\Delta\ln\mathcal{L} = 1/2``,
and a Bayesian model-selection statement would in addition carry the Occam
penalty of the six extra parameters of the two-source hypothesis.

## 3. Physical Model

The pipeline implements a reduced six-parameter frequency-domain inspiral
model, documented equation by equation in [Physics and Waveform Model](physics.md).

**The State Vector (`theta`):**
1.  Amplitude (``A``)
2.  Chirp Mass (``\mathcal{M}``)
3.  Time of Coalescence (``t_c``)
4.  Phase (``\phi_c``)
5.  Primary Spin (``\chi_1``)
6.  Secondary Spin (``\chi_2``)

**Key Physical Features:**
*   **Spin–orbit coupling:** the phasing carries the leading-order 1.5PN
    spin–orbit term, governed by the effective spin ``\chi_{\mathrm{eff}}`` at
    equal mass; it couples the spins to the chirp mass and the coalescence
    time and sets the curvature of the spin directions.
*   **Sub-dominant harmonic:** a ``(3,3)`` harmonic with its own
    stationary-phase phase and a phenomenological amplitude breaks the
    amplitude–mass–time degeneracies of a single-harmonic model.

## 4. Detector Dynamics: Time Delay Interferometry (TDI)

`Detector.jl` projects the source-frame strain into the LISA TDI observables — orbital Doppler modulation, time-dependent antenna patterns, and the noise-orthogonal A/E channels — evaluated per frequency bin through the stationary-phase time–frequency relation. The full detector model is documented in [Physics and Waveform Model](physics.md).

## 5. Physical Bounds and the Capped Zone of Confusion

The local differential geometry is blind to global parameter bounds, and along quasi-degenerate directions the mathematical boundary legitimately diverges: the waveform depends on the spins only through ``\chi_{\mathrm{eff}} = (\chi_1+\chi_2)/2`` (exactly, at equal mass), so along the anti-symmetric combination ``\chi_a`` the manifold is flat and ``\delta_{\mathrm{min}} \to \infty``. Physically, however, ``|\chi| \le 1``, amplitudes/masses/times are non-negative, and phase deviations live on ``[-\pi, \pi]``.

The pipeline therefore computes, per direction ``\varphi``, both the mathematical radius ``r_{\mathrm{math}} = (16\rho^2/K_{\mathrm{raw}})^{1/4}`` (equal source amplitudes are assumed throughout the mapping stage) and the distance to the physical prior box ``r_{\mathrm{box}}``, and publishes the **capped** boundary ``\min(r_{\mathrm{math}}, r_{\mathrm{box}})`` — the exact intersection of the mathematical zone with the prior. Directions where the prior takes over are flagged (`Prior_Limited`) and drawn distinctly in the figures. Both radii are persisted, so the raw mathematical zone remains fully recoverable from the data.

## 6. Computational Architecture (Software Engineering)

Resolving distances down to ``D^2 \sim 10^{-20}`` without precision loss rests on four pillars: ``\mathcal{O}(1)``-rescaled parameters, fused nested-dual automatic differentiation, mirrored and adaptively refined angular mapping with exact corner vertices, and box-constrained interior-point Newton optimization inside the physical priors. The implementation is documented in [Architecture](architecture.md).
