# Physics and Waveform Model

Equation-level description of the model implemented in `Physics.jl` and
`Detector.jl`. The model is a deliberately reduced frequency-domain
inspiral: it keeps the effects that shape the signal-manifold geometry
under study — the Newtonian chirp, the 1.5PN spin–orbit phasing, one
sub-dominant harmonic, and the detector's orbital motion — and omits
everything else. It is a testbed for the geometric law, not a substitute
for a complete waveform approximant; §7 states its domain of validity.

## 1. Parameters

The waveform depends on ``\vec{\theta} = (A, \mathcal{M}, t_c, \phi_c, \chi_1, \chi_2)``:
the strain amplitude, the chirp mass in geometrised seconds
(``1\,\mathrm{s} \approx 2.03\times 10^{5}\,M_\odot``), the coalescence time,
the coalescence phase, and the two dimensionless aligned spins
``\chi_i \in [-1, 1]``. Internally every component is ``\mathcal{O}(1)``, the
physical values being restored through the `[physics]` scales, so the
Fisher matrix stays well conditioned. The amplitude is an independent
parameter: the relation ``A \propto \mathcal{M}^{5/6}/D_L`` of a physical
source is not imposed, so the tangent space carries no
``\partial A/\partial\mathcal{M}`` contribution.

The model assumes equal masses. The symmetric mass ratio ``\eta = 1/4`` is
enforced by the configuration: the spin–orbit coefficient below is the
symmetric-spin part of the full 1.5PN coefficient, which is complete only
at equal mass, and the exact ``\chi_a`` degeneracy discussed in
[Scientific Context](science.md) rests on the same restriction.

## 2. Intrinsic phasing of the dominant harmonic

The ``(2,2)`` harmonic is
```math
\tilde{h}_{22}(f) = A f^{-7/6} e^{i \Psi_{2}(f)}, \qquad
\Psi_{2}(f) = 2\pi f t_c - \phi_c + \frac{3}{128} v^{-5}\left[1 + \sigma v^{3}\right],
\qquad v = (\pi \mathcal{M} f)^{1/3},
```
with the TaylorF2 sign conventions (``\tilde h(f) \propto e^{i\Psi}``): the
stationary-phase time–frequency map follows from the phase,
``\mathrm{d}\Psi_2/\mathrm{d}f = 2\pi\, t(f)`` with
``t(f) = t_c - 5\mathcal{M}/(256 v^{8})``, and the detector model of §4
evaluates the orbital motion on exactly this map. The Newtonian term is
exact in the chirp-mass velocity ``v``. The 1.5PN spin–orbit term of
TaylorF2 is ``4\beta v_M^{3}`` in the total-mass velocity
``v_M = (\pi M f)^{1/3}``; with ``M = \mathcal{M}\eta^{-3/5}`` this is
``\sigma v^{3}`` with
```math
\sigma = 4\beta\,\eta^{-3/5}, \qquad
\beta = \frac{1}{4}\left(\frac{113}{3} - \frac{76}{3}\eta\right)\chi_{\mathrm{eff}},
\qquad \chi_{\mathrm{eff}} = \tfrac{1}{2}(\chi_1 + \chi_2).
```
``\beta`` is the symmetric-spin part of the Poisson–Will coefficient
``\beta = \tfrac{1}{12}\sum_i [113 (m_i/M)^2 + 75\eta]\chi_i``; the
antisymmetric part ``113\,\delta\,\chi_a/12`` (``\delta = (m_1-m_2)/M``)
vanishes at equal mass. Aligned spins (``\chi_{\mathrm{eff}} > 0``) lengthen
the inspiral, the orbital hang-up, and the term is what couples the spins
to the chirp mass and the coalescence time. The 1PN term and the 1.5PN
tail term of TaylorF2 are not modelled.

## 3. The (3,3) harmonic

A single-harmonic model leaves ``A``, ``\mathcal{M}`` and ``t_c`` strongly
degenerate. A sub-dominant ``(3,3)`` harmonic is therefore superposed,
with the stationary-phase phase of harmonic ``m``,
``\Psi_m(f) = 2\pi f t_c - \tfrac{m}{2}\phi_c + \tfrac{m}{2}\,\psi_{\mathrm{PN}}(2f/m)``,
where ``\psi_{\mathrm{PN}}`` is the post-Newtonian part of ``\Psi_2`` and
``2f/m`` the ``(2,2)`` frequency at which harmonic ``m`` radiates at ``f``:
```math
\Psi_{3}(f) = 2\pi f t_c - \tfrac{3}{2}\phi_c
 + \frac{3}{128}\Big[\big(\tfrac{3}{2}\big)^{8/3} v^{-5} + \big(\tfrac{3}{2}\big)^{5/3}\sigma v^{-2}\Big].
```
The arrival-time term is common to every harmonic. The amplitude,
```math
|\tilde{h}_{33}(f)| = a_{33}\, A\, f^{-7/6}\, v, \qquad a_{33} = \texttt{amp\_33\_factor},
```
is phenomenological: the physical ``(3,3)`` amplitude is proportional to
``\delta m/M`` and vanishes at the equal masses assumed above, so the term
is a deliberate degeneracy-breaking perturbation of adjustable strength
(``0.1`` in the shipped configurations), not a physical mode amplitude.
The source-frame strain is ``\tilde h = \tilde h_{22} + \tilde h_{33}``.

## 4. Detector motion

The detector orbits the Sun at ``R = 1\,\mathrm{AU} = 499.005`` light-seconds
with ``\Omega = 2\pi/\mathrm{yr}``. The orbital phase at which the signal at
frequency ``f`` is received is ``\Phi_{\mathrm{orb}}(f) = \Omega\, t(f)`` with the
map of §2, and the source azimuth in the rotating detector frame is
``\varphi(f) = \Phi_{\mathrm{orb}}(f) - \phi_{\mathrm{sky}}``. The orbital Doppler
phase is
```math
\Delta\Phi_{\mathrm{D}}(f) = 2\pi f R \sin\theta_{\mathrm{sky}} \cos\varphi(f).
```

## 5. Antenna response and TDI channels

The antenna patterns are those of a 90° interferometer rotating once per
year in the ecliptic plane, evaluated at the same detector-frame azimuth
``\varphi(f)`` as the Doppler term:
```math
F_+ = \tfrac{1}{2}(1 + \cos^2\theta_{\mathrm{sky}}) \cos 2\varphi \cos 2\psi
      - \cos\theta_{\mathrm{sky}} \sin 2\varphi \sin 2\psi, \qquad
F_\times = \tfrac{1}{2}(1 + \cos^2\theta_{\mathrm{sky}}) \cos 2\varphi \sin 2\psi
      + \cos\theta_{\mathrm{sky}} \sin 2\varphi \cos 2\psi,
```
with the polarisation angle ``\psi``. This is a reduced model of the
constellation's cartwheel: the 60° inclination of the LISA plane and the
resulting time dependence of the source's polar angle in the detector
frame (Cutler 1998) are not modelled. The inclination factors are
``A_+ = \tfrac{1}{2}(1 + \cos^2\iota)`` and ``A_\times = \cos\iota``.

In the long-wavelength limit the noise-orthogonal channels A and E behave
as two 90° interferometers rotated by 45° with respect to each other and
carry the same ``\sqrt{3}/2`` normalisation; T is the null channel:
```math
\tilde{h}_{A}(f) = \frac{\sqrt{3}}{2}\,\mathcal{T}(f)\left[ F_+ A_+ - i F_\times A_\times \right] e^{i \Delta\Phi_{\mathrm{D}}(f)} \tilde{h}(f), \qquad
\tilde{h}_{E}(f) = \frac{\sqrt{3}}{2}\,\mathcal{T}(f)\left[ F_\times A_+ + i F_+ A_\times \right] e^{i \Delta\Phi_{\mathrm{D}}(f)} \tilde{h}(f), \qquad
\tilde{h}_{T} = 0,
```
where ``\mathcal{T}(f) = [1 + 0.6 (f/f_\star)^2]^{-1/2}`` is the finite-arm
transfer roll-off of [Robson2019](@cite) (their Eq. 13) with
``f_\star = c/(2\pi L) = 19.09`` mHz for ``L = 2.5\times 10^{9}`` m; the arm
length is shared with the noise model through `[noise].arm_length`.

## 6. Noise model

Inner products are weighted by the one-sided PSD ``S_n(f)`` of
[Robson2019](@cite): the Michelson-channel instrumental noise of their
Eq. 12,
```math
P_n(f) = \frac{P_{\mathrm{OMS}}}{L^2} + 2\left(1 + \cos^2(f/f_\star)\right)\frac{P_{\mathrm{acc}}}{(2\pi f)^4 L^2},
```
with ``P_{\mathrm{OMS}}`` (Eq. 10) the optical-metrology noise and
``P_{\mathrm{acc}}`` (Eq. 11) the test-mass acceleration noise, plus the
galactic confusion fit of Eq. 14,
```math
S_c(f) = A\, f^{-7/3}\, e^{-f^{\alpha} + \beta f \sin(\kappa f)} \left[ 1 + \tanh\!\big(\gamma (f_k - f)\big) \right], \qquad A = 9\times 10^{-45},
```
with ``(\alpha, \beta, \kappa, \gamma, f_k)`` from their Table 1, selected by
the observation time (1 yr: ``\alpha=0.171``, ``\beta=292``, ``\kappa=1020``,
``\gamma=1680``, ``f_k=2.15`` mHz) and overridable in `[noise]`. The
sky-averaged response ``3/10`` of their Eq. 13 is not applied to the noise,
because the response is modelled explicitly in §5, where its roll-off
factor appears instead. The confusion fit is calibrated for observation
times between 0.5 and 4 yr; shorter grids use the nearest column.

## 7. Domain of validity

The model has no merger cutoff: every quantity is a smooth function of
``\vec\theta`` over the whole configured band, which is what the
differential-geometry engine requires. Physically, however, the inspiral
description holds only below the innermost stable circular orbit,
``f_{\mathrm{ISCO}} = (6^{3/2}\pi M)^{-1}``. For the production base points
(``\mathcal{M} = 15`` s at ``\eta = 1/4``, i.e. ``M = 34.5`` s and
``M \approx 7\times 10^{6}\,M_\odot``) this is ``0.63`` mHz, while the
production band extends to ``50`` mHz, where ``v > 1``; the phasing above
``f_{\mathrm{ISCO}}`` is therefore a smooth extrapolation rather than a
physical inspiral, and the long-wavelength response is used up to
``2.6 f_\star``. The band and the mass scale are campaign choices
([Complex Run Parameters](parameters.md)); the geometric law itself makes
no assumption about them.
