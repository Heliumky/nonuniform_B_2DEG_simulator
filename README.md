# Nonuniform-\(B\) 2DEG simulator

Numerical solver for one fixed-\(k_y\) channel of a two-dimensional electron
gas in a linearly varying perpendicular magnetic field. It calculates static
bands and eigenstates, and propagates a DC-biased eigenstate under a sinusoidal
or cosine electric drive. Internally it uses atomic units; figures are labelled
in nm and meV where appropriate.

This is a single-particle, fixed-channel model. It does **not** include
electron--electron interactions, disorder, Zeeman coupling, spin, boundaries
in \(y\), or a sum over occupied \(k_y\) channels.

## Model and sign convention

Electrons have physical charge \(q=-e\), with \(e>0\). The code uses the
positive magnitude `ELEMENTARY_CHARGE = e`; do not replace it by `-1`. For

\[
\mathbf B(x)=\frac{B_0x}{L}\hat{\mathbf z},\qquad
\mathbf A(x)=\frac{B_0x^2}{2L}\hat{\mathbf y},\qquad
\phi(x,t)=-\frac{V_e(t)}{L}x,
\]

translation invariance in \(y\) makes \(p_y=\hbar k_y\) a conserved quantum
number. In each channel the Hamiltonian is

\[
\begin{aligned}
H_{k_y}(t)&=\frac{p_x^2}{2m^*}+V_{k_y}(x,t),\\
V_{k_y}(x,t)&=
\frac{1}{2m^*}\left(\hbar k_y+\frac{eB_0x^2}{2L}\right)^2
+\frac{e}{L}[V_{\rm DC}+V_{\rm AC}f(\omega_{\rm ac}t)]x.
\end{aligned}
\]

This is the expression in `src/hamiltonian.py`. In particular, the electric
term has a plus sign: \(q\phi=(-e)(-V_ex/L)=+eV_ex/L\). Expanding the square
gives the implemented \(x^2\), \(x^4\), and constant terms. For zero electric
bias, positive \(k_y\) gives one central well; negative \(k_y\) gives two
classical minima. A DC bias tilts the two wells.

## Numerical method

`src/basis.py` represents operators in an \(N\)-state harmonic-oscillator
(SHO) basis. The analytic matrices for \(x\), \(x^2\), \(x^4\), and \(p_x^2\)
are projected directly into that basis. This is a Galerkin truncation on the
whole real line, not a finite-difference grid with hard walls.

Static eigenpairs are obtained either by dense Hermitian diagonalization
(`scipy.linalg.eigh`) or ARPACK's Hermitian Lanczos routine (`eigsh`).
Eigenstates are sorted by energy: `n=0` is always the ground state.

For the driven problem, the code splits

\[
H(t)=T+V(t),\qquad T=p_x^2/(2m^*),
\]

and advances it with the five-exponential, fully commutator-free sixth-order
\(\Upsilon^{[6]}_3\) / `Y6_3` scheme from `method.pdf`. It samples \(V(t)\)
at the three sixth-order Gauss--Legendre nodes in every step and evaluates each
matrix-exponential action with `scipy.sparse.linalg.expm_multiply`. The
propagator is unitary up to the tolerance of that exponential action.

`AC_WAVEFORM="sin"` is continuous at \(t=0\), since \(f(0)=0\). With
`"cos"`, the same DC-prepared eigenstate is used but the AC term is already
maximal at \(t=0\); this is intentionally a quench.

The current plot uses the charge-current density per unit length in \(y\),

\[
J_x=-\frac{e\hbar}{m^*}\operatorname{Im}(\psi^*\partial_x\psi),\qquad
J_y=-\frac{e}{m^*}\left(\hbar k_y+\frac{eB_0x^2}{2L}\right)|\psi|^2.
\]

The second expression includes the vector-potential (diamagnetic) term.

## Requirements

Use Python 3 with NumPy, SciPy, Matplotlib, and Pillow (the GIF writer):

```bash
python -m pip install numpy scipy matplotlib pillow
```

Run commands from the repository root so that the `src`, `statics`, and
`dynamics` packages can be imported.

## Configure and run

The static and driven workflows currently have separate configuration modules:

- `statics/config.py` controls the spectrum, static state, and potential plots.
- `dynamics/config.py` controls the driven calculation and its animations.

Keep the shared material constants identical between those two files when the
figures are meant to describe the same physical system. The main driven
parameters are `BASIS_SIZE`, `KY`, `V_DC`, `V_AC`, `AC_FREQUENCY`,
`INITIAL_STATE_INDEX`, `NUMBER_OF_STEPS`, `TOTAL_TIME`, and `AC_WAVEFORM`.
All `KY` values passed to the core Hamiltonian are in inverse bohr; the
configuration converts a value stated in nm\(^{-1}\) with
`KY = value_in_nm_inverse * AU_TO_NM`.

```bash
# Driven propagation: prints coefficient-array shape and norm range
python dynamics/run_dynamics.py

# Driven figures (each reruns the propagation)
python dynamics/run_probability_animation.py
python dynamics/run_current_animation.py
python dynamics/run_potential_animation.py

# Static figures
python statics/run_spectrum.py
python statics/run_wave_function.py
python statics/run_potential.py
```

Outputs are written below `figures/dynamics/` and `figures/statics/`. Names
encode the principal numerical settings, which avoids silently overwriting a
different run. The two animation commands each independently perform the full
propagation; generate one trajectory once and reuse it externally if runtime
becomes important.

## Checks before interpreting a result

No automated test suite is currently included. The following convergence
checks are therefore part of a reliable calculation:

1. Increase `BASIS_SIZE` until every reported eigenvalue, relevant density,
   and observable are unchanged at the required precision. The basis scale
   (`omega = CYCLOTRON_FREQUENCY`) is numerical, so this check is essential.
2. Increase `NUMBER_OF_STEPS` (for example, double it) at fixed `TOTAL_TIME`.
   Compare the final state up to a global phase and compare observables over
   the whole trajectory. Do not infer sixth-order accuracy solely from the
   formal order of the integrator.
3. For ARPACK spectra, repeat representative points with `SPECTRUM_SOLVER =
   "dense"` and/or a larger `LANCZOS_DIMENSION`. In the driven workflow, the
   distinct `INITIAL_EIGSH_NCV` setting controls only the ARPACK solve used to
   prepare the initial state.
4. Reconstruct the wavefunction on a wider, finer plotting grid and verify
   \(\int|\psi|^2dx\simeq1\) and negligible probability near the displayed
   edges. The plotting grid is only for reconstruction; it does not confine
   the Hamiltonian.
5. Monitor `np.linalg.norm(states, axis=1)`. It should stay near one. For
   current diagnostics, also check
   \(\partial_t[-e|\psi|^2]+\partial_xJ_x\simeq0\) using
   `dynamics.current.continuity_residual` away from finite-difference edges.

For the present default physical parameters, direct dense checks gave stable
lowest four zero-bias \(k_y=-0.15\,\mathrm{nm}^{-1}\) energies from 101 through
501 SHO states. For the configured DC-biased spectrum, the lowest 12 levels
at the demanding \(k_y=-0.5\,\mathrm{nm}^{-1}\) endpoint require at least
about 201 states; 101 is not converged for the upper displayed levels. These
are useful sanity checks, not a replacement for convergence tests after a
parameter change.

## Review notes and limitations

- The Hamiltonian construction, its electron-sign convention, and the
  mechanical-momentum current are mutually consistent.
- A standalone noncommuting-matrix comparison against a high-accuracy ODE
  solution showed the implemented `Y6_3` error shrink by about \(2^6\) when
  halving the step, as expected. This verifies the coefficient ordering, not
  every physical parameter choice.
- `AU_TO_MEV = 2.7e4` and `AU_TO_NM = 5.29e-2` are rounded conversion factors.
  They are adequate for the current plots but should be replaced by consistent
  CODATA values if sub-per-mille absolute labels matter.
- The driven exponential action uses SciPy's adaptive `expm_multiply`, not a
  user-configurable Lanczos exponential. Its action-approximation tolerance is
  fixed internally near double-precision round-off; reduce the physical time
  step with `NUMBER_OF_STEPS` to test propagation accuracy.
- Near-degenerate double-well states can rotate within their nearly degenerate
  subspace as parameters change. Compare densities or the subspace rather
  than treating a floating-point eigenvector sign/phase as physical.

For a longer derivation of the gauge choice, well structure, propagator, and
current formulae, see `non_BTD.md`.
