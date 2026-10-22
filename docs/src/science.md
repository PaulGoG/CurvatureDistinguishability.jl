# Scientific Context

The results computed by `CurvatureDistinguishability.jl` rest on the
geometric formulation, the discernibility criterion and the modelling
assumptions set out below.

## 1. Geometry of source confusion

Consider a data stream that contains two distinct, closely overlapping
gravitational-wave (GW) signals with a small parameter separation ``\delta``
(the two-source hypothesis) and is fitted with a single-source model. The
linear Fisher-matrix approximation predicts that the residual
power, the squared distance ``D^2`` between the true two-source data and the
best-fit single-source manifold, scales quadratically as ``\delta^2``.

The manifold of single-source waveforms ``\mathcal{M}_1`` adapts to the data.
The optimizer adjusts the parameters of the single source and so absorbs the
linear (``\mathcal{O}(\delta)``) and quadratic (``\mathcal{O}(\delta^2)``)
differences, assuming equal amplitudes.

The unabsorbable residual lies in the normal space, orthogonal to the tangent
space of the signal manifold. The magnitude of this orthogonal projection is
set by the extrinsic curvature ``K(u)`` of the manifold along the direction
``u`` of the parameter separation, and the distance ``D^2`` scales quartically:
```math
D^2 \approx \frac{1}{16} K(u) \delta^4
```
The signal amplitude enters ``K(u)`` through the noise-weighted inner product,
so no separate amplitude factor appears in the law.

## 2. Discernibility limit

The criterion is a residual signal-to-noise ratio: two sources count as
distinguishable from a single one when the unabsorbable residual power
exceeds a threshold ``\rho_{\mathrm{threshold}}``,

```math
D^2 \ge \rho_{\mathrm{threshold}}^2
```
Substituting the geometric scaling law gives the analytic limit of
discernibility, which is the boundary of the zone of confusion:

```math
\delta_{\mathrm{min}}(u) = \left( \frac{16 \cdot \rho_{\mathrm{threshold}}^2}{K(u)} \right)^{1/4}
```
Because of the ``1/4`` exponent, lowering ``\rho_{\mathrm{threshold}}``
yields strongly diminishing returns for resolving overlapping sources. The
shipped ``\rho_{\mathrm{threshold}} = 1`` is the leading-order resolvability
scale, not a detection-significance threshold. With the likelihood identity
``\Delta\ln\mathcal{L}_{\max} = D^2/2`` it corresponds to
``\Delta\ln\mathcal{L} = 1/2``, and a Bayesian model-selection statement would
in addition carry the Occam penalty of the six extra parameters of the
two-source hypothesis.

## 3. Physical model

The pipeline implements a six-parameter frequency-domain inspiral of an
aligned-spin massive black-hole binary, with TaylorF2 phasing to 1.5PN order,
0.5PN harmonic amplitudes and an innermost-stable-orbit window, observed
through the long-wavelength response of the LISA constellation on its
analytic orbits. The model is documented equation by equation in
[Waveform Physics](physics.md). The state vector `theta` holds, in
order, the luminosity distance ``D_L``, the chirp mass ``\mathcal{M}``, the
coalescence time ``t_c``, the coalescence phase ``\phi_c`` and the aligned
spins ``\chi_1`` and ``\chi_2``.

The phasing carries the complete 1.5PN spin–orbit term of Poisson and Will,
which couples the spins to the chirp mass and the coalescence time through the
single coefficient ``\beta``. The spin combination with
``\mathrm{d}\beta = 0`` is therefore an exact flat direction at every mass
ratio.

The ``(1,1)`` and ``(3,3)`` harmonics are physical: they enter with their
0.5PN amplitudes, proportional to the mass asymmetry ``\sqrt{1 - 4\eta}``,
and break the distance–mass–time degeneracies of a single-harmonic model
without any phenomenological factor.

## 4. Detector response

`Detector.jl` projects the source-frame strain into the A and E channel
combinations of the constellation, which are noise-orthogonal for equal arms
and equal link noise. The projection uses long-wavelength Michelson antenna
patterns on the analytic spacecraft orbits rather than a time-delay
interferometry simulation, and includes the orbital Doppler phase and the
finite-arm roll-off. The response is evaluated per frequency bin and per
harmonic at the stationary-phase emission time. The full detector model is
documented in [Waveform Physics](physics.md).

## 5. Physical bounds and the capped zone of confusion

The local differential geometry is blind to global parameter bounds, and
along quasi-degenerate directions the mathematical boundary legitimately
diverges. The 1.5PN phase sees the spins only through ``\beta``, so along the
spin combination with ``\mathrm{d}\beta = 0`` (``\chi_a`` at equal mass, a
tilted line otherwise) the manifold is exactly flat and
``\delta_{\mathrm{min}} \to \infty``. The physical parameters are bounded
nonetheless: ``|\chi| \le 1``, distances, masses and times are non-negative,
and phase deviations live on ``[-\pi, \pi]``.

For each direction ``\varphi`` the pipeline computes both the mathematical
radius ``r_{\mathrm{math}} = (16\rho^2/K_{\mathrm{raw}})^{1/4}`` and the
distance ``r_{\mathrm{box}}`` to the physical prior box. Equal source
amplitudes are assumed throughout the mapping stage. The published boundary is
the capped radius ``\min(r_{\mathrm{math}}, r_{\mathrm{box}})``, the exact
intersection of the mathematical zone with the prior. Directions where the
prior takes over are flagged (`Prior_Limited`) and drawn distinctly in the
figures. Both radii are persisted, so the raw mathematical zone remains fully
recoverable from the data.

## 6. Computational design

Resolving distances down to ``D^2 \sim 10^{-20}`` without precision loss
rests on ``\mathcal{O}(1)``-rescaled parameters, fused nested-dual automatic
differentiation, mirrored and adaptively refined angular mapping with exact
corner vertices, and box-constrained interior-point Newton optimization inside
the physical priors. The implementation is documented in
[Architecture](architecture.md).
