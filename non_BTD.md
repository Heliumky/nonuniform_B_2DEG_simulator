---
marp: true
theme: default
paginate: true
math: katex
size: 16:9
style: |
  section {
    font-family: "Noto Sans", "Noto Sans CJK TC", "Arial", sans-serif;
    font-size: 22px;
    line-height: 1.28;
    color: #17212b;
    padding: 34px 52px;
  }
  h1 { font-size: 42px; color: #17324d; }
  h2 { font-size: 32px; color: #17324d; }
  h3 { font-size: 25px; color: #17324d; }
  code { font-family: "JetBrains Mono", "Consolas", monospace; }
  pre { font-size: 15px; line-height: 1.12; }
  .small { font-size: 18px; }
  .tiny { font-size: 15px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; align-items: start; }
  .box { border: 2px solid #2b5c8a; border-radius: 14px; padding: 12px 16px; background: #f7fbff; }
  .warn { border: 2px solid #bd3e3e; border-radius: 14px; padding: 12px 16px; background: #fff2f2; }
  .ok { border: 2px solid #2f8f46; border-radius: 14px; padding: 12px 16px; background: #edf8ef; }
  .center { text-align: center; }
  table { font-size: 18px; }
  section.title {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }
  section.title h1,
  section.title h2,
  section.title h3 {
    margin: 0.2em 0;
  }
---

<!-- _class: title -->

# Time-Dependent Dynamics of a 2DEG in a Linearly Nonuniform Magnetic Field

### National Tsing Hua University

### Qiancan Chen (Jerry Chen)

---

# Questions addressed in this report

1. How does the full two-dimensional electromagnetic Hamiltonian reduce to the one-dimensional quartic Hamiltonian used in the code?
2. How are the static eigenvalues and eigenstates computed numerically in a simple-harmonic-oscillator (SHO) basis?
3. What exactly is the status of the claimed static zero-energy analytic solution?
4. How does the sixth-order time-dependent algorithm in `method.pdf` work, and where is each of its ingredients used in the code?
5. Starting from a gauge-covariant Schrödinger field Lagrangian, what is the charge-current density, and how is it evaluated for each $k_y$ channel?

---

# Geometry, fields, and conventions

We consider electrons ($q=-e$, $e>0$) confined to the $x$–$y$ plane. The prescribed external fields are

$$
\mathbf B(x)=\frac{B_0x}{L}\hat{\mathbf z},
\qquad
\mathbf E(t)=\frac{V_e(t)}{L}\hat{\mathbf x}.
$$

The magnetic field changes sign at $x=0$. $L$ is the magnetic-gradient length: $|B(\pm L)|=B_0$.

We use the gauge

$$
\mathbf A(x)=A_y(x)\hat{\mathbf y}
=\frac{B_0x^2}{2L}\hat{\mathbf y},
\qquad
\phi(x,t)=-\frac{V_e(t)}Lx.
$$

Indeed, $\partial_xA_y=B_0x/L$ and $-\partial_x\phi=V_e(t)/L$.

---

# Full two-dimensional Hamiltonian

For charge $q=-e$, minimal coupling gives

$$
H(t)=\frac{1}{2m^*}\left[-i\hbar\boldsymbol\nabla+e\mathbf A(x)\right]^2-e\phi(x,t).
$$

With the chosen scalar potential,

$$
H(t)=\frac{p_x^2}{2m^*}
+\frac{1}{2m^*}\left(p_y+\frac{eB_0x^2}{2L}\right)^2
+\frac{eV_e(t)}Lx.
$$

The sign of the linear term is important. For an electron, its potential energy is $q\phi=-e\phi=+eV_ex/L$, even though the physical electric field is $+V_e\hat x/L$.

---

# Reduction to independent $k_y$ channels

The full Hamiltonian contains no $y$, so

$$
[H,p_y]=0.
$$

In a box of length $L_y$, choose the normalized channel decomposition

$$
\Psi(x,y,t)=\frac{e^{ik_yy}}{\sqrt{L_y}}\psi_{k_y}(x,t),
\qquad p_y\Psi=\hbar k_y\Psi.
$$

Every $k_y$ therefore evolves independently:

$$
i\hbar\partial_t\psi_{k_y}(x,t)=H_{k_y}(t)\psi_{k_y}(x,t),
$$

with $\int dx\,|\psi_{k_y}|^2=1$ for one occupied channel.

---

# Effective one-dimensional Hamiltonian

Substituting $p_y\to\hbar k_y$ gives

$$
H_{k_y}(t)=\frac{p_x^2}{2m^*}+V_{k_y}(x,t),
$$

$$
V_{k_y}(x,t)=
\frac{1}{2m^*}\left(\hbar k_y+\frac{eB_0x^2}{2L}\right)^2
+\frac{eV_e(t)}Lx.
$$

This is the effective Hamiltonian studied in arXiv:2601.05064. It is quartic in $x$, bounded from below, and has no generic closed-form spectrum.

---

# Explicit polynomial form used by the code

Expanding the effective potential,

$$
\begin{aligned}
H_{k_y}(t)=&\frac{p_x^2}{2m^*}
+\frac{\hbar^2k_y^2}{2m^*}I
+\frac{\hbar k_yeB_0}{2m^*L}x^2\\
&+\frac{e^2B_0^2}{8m^*L^2}x^4
+\frac{eV_e(t)}Lx.
\end{aligned}
$$

This is exactly the coefficient structure in `hamiltonian.py`:

```python
c_p2 = 1 / (2*m_star)
c_x  = e*Ve / L
c_x2 = hbar*ky*e*B0 / (2*m_star*L)
c_x4 = e**2*B0**2 / (8*m_star*L**2)
c_x0 = hbar**2*ky**2 / (2*m_star)
```

---

# Static potential: derive the one-well / two-well structure

Set $V_e(t)=V_e$ constant and first take $V_e=0$. Define

$$
\alpha=\frac{eB_0}{2L}=\frac{m^*\omega_c}{2L},
\qquad
V_0(x)=\frac{(\hbar k_y+\alpha x^2)^2}{2m^*}.
$$

Its derivative is

$$
\frac{dV_0}{dx}=\frac{2\alpha x}{m^*}(\hbar k_y+\alpha x^2).
$$

Thus the stationary points are $x=0$ and, only if $k_y<0$,

$$
x_\pm=\pm\sqrt{-\frac{\hbar k_y}{\alpha}}
=\pm\sqrt{\frac{-2\hbar k_yL}{m^*\omega_c}}.
$$

---

# Why $k_y>0$ gives one well whereas $k_y<0$ gives two

At the central stationary point,

$$
V_0''(0)=\frac{2\alpha\hbar k_y}{m^*}.
$$

Therefore:

<div class="cols">
<div class="box">

### $k_y>0$

$V_0''(0)>0$, and the nonzero stationary-point condition has no real solution. Hence $x=0$ is the only minimum: one interface-localized, snake-like well.

</div>
<div class="box">

### $k_y<0$

$V_0''(0)<0$, so $x=0$ is a barrier. The two real points $x_\pm=\pm|x_0|$ are degenerate minima, because

$$
\hbar k_y+\alpha x_\pm^2=0.
$$

</div>
</div>

The two wells are related by $x\to-x$ only at $V_e=0$; tunnel coupling makes the exact eigenstates even/odd combinations, with an exponentially small splitting when the barrier is high.

---

# Local approximation: what is being expanded?

Pick **one** minimum, $x_\sigma=\sigma|x_0|$ with $\sigma=+1$ (right well) or $-1$ (left well), and write a nearby position as

$$
x=x_\sigma+\xi,\qquad |\xi|\ll |x_0|.
$$

Because $x_\sigma$ is a minimum, $V_0'(x_\sigma)=0$. The Taylor expansion therefore begins as

$$
V_0(x_\sigma+\xi)
\simeq V_0(x_\sigma)+\frac12V_0''(x_\sigma)\xi^2.
$$

This is compared with the harmonic-oscillator potential

$$
V_{\rm HO}(\xi)=\frac12m^*\omega_{\rm loc}^2\xi^2.
$$

Thus the curvature is simply the local spring constant:

$$
\boxed{m^*\omega_{\rm loc}^2=V_0''(x_\sigma).}
$$

---

# Compute the local oscillator frequency step by step

Let $g(x)=\hbar k_y+\alpha x^2$, so $V_0=g^2/(2m^*)$. At either minimum, $g(x_\sigma)=0$; hence

$$
V_0''(x_\sigma)=\frac{[g'(x_\sigma)]^2}{m^*}
=\frac{(2\alpha x_\sigma)^2}{m^*}
=\frac{4\alpha^2x_0^2}{m^*}.
$$

Comparing with $m^*\omega_{\rm loc}^2$ gives

$$
\omega_{\rm loc}=\frac{2\alpha|x_0|}{m^*}
=\omega_c\frac{|x_0|}{L}.
$$

So a deeper-separated well ($|x_0|$ larger) is locally a stiffer oscillator and has a larger level spacing.

---

# Local energy and its $k_y$ slope

Ignoring tunnelling between the two wells, the local quantized levels are

$$
E_{n,\sigma}^{(0)}\simeq
\hbar\omega_{\rm loc}(n+\tfrac12)
=\hbar\omega_c\frac{|x_0|}{L}(n+\tfrac12).
$$

For $k_y<0$,

$$
|x_0|=\sqrt{\frac{-2\hbar k_yL}{m^*\omega_c}}
\ \propto\sqrt{-k_y}.
$$

Therefore $E_n\propto\sqrt{-k_y}$. Increasing $k_y$ from a negative value toward zero makes $\sqrt{-k_y}$ smaller, so

$$
\frac{dE_n}{dk_y}<0.
$$

By $v_y=(1/\hbar)dE/dk_y$, this is the negative-$y$ gradient-$B$ drift branch.

---

# Add the constant electric tilt

The electric potential is $V_E=eV_ex/L$. Evaluated at the two minima,

$$
V_E(x_\sigma)=\sigma\frac{eV_e|x_0|}{L}.
$$

To harmonic order, the two local branches are therefore

$$
\boxed{
E_{n,\sigma}(k_y)\simeq
\frac{|x_0(k_y)|}{L}
\left[\hbar\omega_c(n+\tfrac12)+\sigma eV_e\right]
}\,,\qquad k_y<0.
$$

This equation—not just the qualitative statement that the potential tilts—is the direct reason for the reported flat branch.

---

# Where the nearly flat branch comes from

For positive $V_e$, choose the lower-$x$ well, $\sigma=-1$. The preceding two-well energy estimate becomes

$$
E_{n,-}(k_y)\simeq
\frac{|x_0(k_y)|}{L}
\left[\hbar\omega_c(n+\tfrac12)-eV_e\right].
$$

At

$$
\boxed{\ eV_e=(n+\tfrac12)\hbar\omega_c\ },
$$

the coefficient multiplying $|x_0(k_y)|$ vanishes. Thus the leading $\sqrt{-k_y}$ dispersion cancels for one localized branch:

$$
E_{n,-}(k_y)\simeq0
\quad\text{for all $k_y<0$ within the separated-well approximation.}
$$

For negative $V_e$, the corresponding flat branch is localized in the upper ($\sigma=+1$) well.

---

# Why the result is “nearly” flat in the full calculation

The two-well harmonic estimate neglects several effects:

- tunnelling between the two wells;
- anharmonic corrections beyond the quadratic expansion;
- the displacement of the minima caused by the linear tilt;
- finite SHO-basis truncation in the numerical calculation.

Therefore the exact full-line eigenvalue is not required to be literally zero or exactly $k_y$ independent. The numerical claim should be that its residual bandwidth is much smaller than the relevant Landau-like gap, after convergence in `nmax`.

---

# Static transport benchmark: band slope

Differentiating the effective Hamiltonian with respect to $k_y$ gives

$$
\frac{\partial H_{k_y}}{\partial k_y}
=\frac{\hbar}{m^*}\left(\hbar k_y+eA_y(x)\right).
$$

For a normalized stationary state, Hellmann–Feynman gives

$$
\frac1\hbar\frac{\partial E_n(k_y)}{\partial k_y}
=\left\langle\frac{\hbar k_y+eA_y(x)}{m^*}\right\rangle
=\langle v_y\rangle.
$$

Thus a flat band has zero **integrated mean velocity**, not necessarily zero local current density. The band-slope identity above is a stringent static check for the current implementation.

---

# What is $\psi_{\rm form}$?

$\psi_{\rm form}$ means **formal wavefunction**. It is not obtained from the SHO numerical diagonalization and it is not an arbitrary trial function.

At the special static field

$$
eV_e=\frac12\hbar\omega_c,
$$

the differential Hamiltonian has a special algebraic form. One can solve a first-order equation instead of the usual second-order Schrödinger equation.

The paper calls the resulting expression a zero-energy solution. We denote it by $\psi_{\rm form}$ first because we must still check whether it is normalizable and satisfies the physical domain conditions.

---

# Step 1a: write the Hamiltonian at the critical field

Define

$$
f(x)=k_y+\frac{m^*\omega_c}{2\hbar L}x^2.
$$

At $eV_e=\hbar\omega_c/2$, the effective Hamiltonian is explicitly

$$
H_{k_y}=-\frac{\hbar^2}{2m^*}\partial_x^2
+\frac{1}{2m^*}
\left(\hbar k_y+\frac{m^*\omega_c}{2L}x^2\right)^2
+\frac{\hbar\omega_c}{2L}x.
$$

The first term is kinetic energy, the squared term is the magnetic effective potential, and the last term is the critical electric tilt. The aim is to recover these three terms from the product of two first-order operators.

---

# Step 1b: expand the operator product

Let the product act on an arbitrary test function $\chi(x)$:

$$
\begin{aligned}
(\partial_x+f)(-\partial_x+f)\chi
&=(\partial_x+f)\big[-\chi'+f\chi\big]\\
&=-\chi''+f'\chi+f\chi'-f\chi'+f^2\chi\\
&=\left[-\partial_x^2+f'(x)+f(x)^2\right]\chi.
\end{aligned}
$$

The $f\chi'$ terms cancel. Crucially, $\partial_x$ differentiates $f(x)\chi(x)$ and creates the $f'(x)$ term.

---

# Step 1c: match the three terms

For

$$
f(x)=k_y+\frac{m^*\omega_c}{2\hbar L}x^2,
$$

the two new pieces are

$$
\frac{\hbar^2}{2m^*}f^2
=\frac1{2m^*}\left(\hbar k_y+\frac{m^*\omega_c}{2L}x^2\right)^2,
$$

$$
\frac{\hbar^2}{2m^*}f'
=\frac{\hbar\omega_c}{2L}x.
$$

Together with $-\hbar^2\partial_x^2/(2m^*)$, these are exactly the three terms in the critical Hamiltonian displayed two slides earlier. Therefore

$$
\boxed{H_{k_y}=\frac{\hbar^2}{2m^*}(\partial_x+f)(-\partial_x+f).}
$$

---

# Step 1d: obtain the zero-mode condition

This factorization is analogous to writing a harmonic oscillator Hamiltonian as a product of ladder-operator-like first-order operators. If a state satisfies

$$
(-\partial_x+f)\psi(x)=0,
$$

then the product above immediately gives $H_{k_y}\psi=0$.

---

# Step 2: solve the first-order equation

The first-order equation is

$$
\partial_x\psi(x)=f(x)\psi(x).
$$

Divide by $\psi$ and integrate:

$$
\log\psi(x)=\int^x f(s)\,ds+\log A
=k_yx+\frac{m^*\omega_c}{6\hbar L}x^3+\log A.
$$

Therefore

$$
\boxed{
\psi_{\rm form}(x)=A\exp\!\left[k_yx+
\frac{m^*\omega_c}{6\hbar L}x^3\right]
}.
$$

So $\psi_{\rm form}$ is simply the solution of this factorized **first-order** zero-mode equation.

---

# Limitation of the formal zero mode

Its squared amplitude is

$$
|\psi_{\rm form}(x)|^2\propto
\exp\!\left(2k_yx+\frac{m^*\omega_c}{3\hbar L}x^3\right).
$$

For the physical sign $B_0/L>0$, it decays as $x\to-\infty$ but diverges as $x\to+\infty$. Consequently,

<div class="warn">

On the full real line, the displayed exponential expression is not in $L^2(\mathbb R)$ and is not a normalizable bound eigenstate of the self-adjoint quartic Hamiltonian. It should not be presented as a globally physical exact eigenfunction without specifying a different domain and boundary condition.

</div>

The factorization above still explains why a left-localized state can become exponentially close to zero energy. The full-line numerical eigenproblem remains the appropriate static calculation.

---

# Consequence for a careful static report

The quartic potential confines as $x\to\pm\infty$. Near the critical field, a state localized in the lowered well is near the formal zero-mode envelope, while tunnelling and the opposite side of the full domain set the residual energy and residual band curvature.

Therefore quote all three quantities:

$$
\Delta E_{\rm band}=\max_{k_y\in W}E_n(k_y)-\min_{k_y\in W}E_n(k_y),
$$

the $k_y$ window $W$, and convergence with basis size $n_{\max}$. “Exactly flat” should be reserved for a result supported to the desired numerical precision and physical domain.

---

# Static numerical method: SHO basis

The code does not discretize real space. It expands the channel wavefunction in $N=n_{\max}$ harmonic-oscillator functions,

$$
\psi_{k_y}(x)=\sum_{n=0}^{N-1}c_n\phi_n(x),
$$

using the reference frequency $\omega_c=eB_0/m^*$ and oscillator length

$$
\ell_0=\sqrt{\frac{\hbar}{m^*\omega_c}},
\qquad x_0=\frac{\ell_0}{\sqrt2}.
$$

This is a computational basis, not a claim that the nonuniform-field problem is a harmonic oscillator. It is convenient because all terms in (5) have sparse, analytic matrix elements.

---

# SHO ladder operators and matrix elements

Define

$$
x=x_0(a+a^\dagger),
\qquad p_x=i\sqrt{\frac{m^*\hbar\omega_c}{2}}(a^\dagger-a),
$$

with $a|n\rangle=\sqrt n|n-1\rangle$. Therefore

$$
\langle n|x|m\rangle=x_0\left(\sqrt{m+1}\,\delta_{n,m+1}
+\sqrt m\,\delta_{n,m-1}\right),
$$

and $p_x^2$, $x^2$, and $x^4$ connect only

$$
\Delta n=0,\ \pm2,\ \pm4
$$

(except $x$, which connects $\Delta n=\pm1$). Hence the static matrix is real, symmetric, and banded before any dense eigensolver is applied.

---

# Where the SHO matrices are built

`basis.py:operators_matrixize(nmax)` constructs

```python
x_mat, x2_mat, x4_mat, p2_mat
```

from the explicit ladder-operator expressions. For example:

```python
x_mat += diag(x0*sqrt(ns[1:]), k=-1)
x_mat += diag(x0*sqrt(ns[1:]), k=+1)

p2_mat += diag(p0**2*(2*ns+1))
p2_mat += diag(-p0**2*sqrt(ns[2:]*ns[1:-1]), k=-2)
p2_mat += diag(-p0**2*sqrt(ns[2:]*ns[1:-1]), k=+2)
```

The $x^4$ implementation likewise fills diagonal, $\Delta n=\pm2$, and $\Delta n=\pm4$ bands. `hamiltonian.py:build_H` combines these matrices using the displayed polynomial coefficients.

---

# Static eigensolvers actually used

For a fixed $(k_y,V_e)$, the truncated problem is

$$
H^{(N)}\mathbf c_n=E_n^{(N)}\mathbf c_n.
$$

The implementation has two paths.

| Use | Code path | Numerical method |
|---|---|---|
| Wavefunction plots and small matrices | `wave_function.py`, `hamiltonian.py` | `scipy.linalg.eigh`: dense Hermitian diagonalization |
| Spectrum sweep, `method="dense"` | `spectrum.py:compute_spectrum` | `scipy.linalg.eigh` |
| Lowest bands of a large sweep | `spectrum.py`, `method="lanczos"` | ARPACK implicitly restarted Lanczos via `scipy.sparse.linalg.eigsh` |

The Lanczos wrapper is `lanczos.py:lanczos_eigenpairs`. It returns the lowest
`state_count` Ritz pairs, ordered by energy: $n=0,ldots,\text{state_count}-1$.

---

# Static convergence and tracking cautions

The basis truncation is variational for the lowest eigenvalue, but each reported quantity must be converged with $N$.

- Increase `nmax` until the selected energies, density, and $\langle x^2\rangle$ are stable.
- Near avoided crossings, energy sorting alone may exchange physical state labels. Track states with overlaps $|\langle\psi_n(k_y)|\psi_m(k_y+\delta k)\rangle|$ if following one band continuously.
- `spectrum.py` does not implement multi-band overlap tracking.
- A large negative $k_y$, strong tilt, or strong drive can require substantially more basis functions than a low-lying state near $k_y=0$.

---

# Units in the implementation

`params.py` uses Hartree atomic units:

$$
\hbar=e=m_e=1,\qquad m^*=0.067.
$$

The code sets

```python
B0 = 1.65/(2.35e5)
w_c = e*B0/m_star
L  = 16*pi**2/AU_TO_NM
```

and converts only for output using `AU_TO_MEV` and `AU_TO_NM`. Equations below retain $\hbar$ and $e$ to display the physics; remove them consistently only inside atomic-unit code.

---

# Time-dependent protocol currently implemented

The present code uses a pure AC scalar-potential drive,

$$
V_e(t)=V_{\rm ac}\sin(\omega_{\rm ac}t).
$$

In the current `dynamics/propagation.py` implementation:

```python
w_ac = 1*w_c
T = 20*pi/w_ac
nstep = 1000
dt = T/nstep

def drive_factor(t):
    return np.sin(w_ac*t)
```

The initial vector is eigenstate $n$ of $H(t=0)=H[V_e=0]$; energy eigenstates are
sorted in increasing energy and $n=0$ is the ground state.

---

# Two distinct physical drive protocols

The current implementation crosses each static field value repeatedly during a
cycle. Thus a static flat-band condition is met, at most, at isolated instants.

To probe the response about one static critical configuration, use instead

$$
V_e(t)=V_{\rm dc}+V_{\rm ac}\cos(\omega_{\rm ac}t),
\qquad eV_{\rm dc}=(n+\tfrac12)\hbar\omega_c.
$$

This DC-plus-AC protocol is not currently exposed as a code option. It is a different experiment: small-amplitude response near a drift-compensated state rather than a full sweep through both tilt directions.

---

# Why time ordering is required

For a time-dependent Hamiltonian,

$$
\psi(t+\tau)=\mathcal T
\exp\!\left[-\frac{i}{\hbar}\int_t^{t+\tau}H(s)\,ds\right]\psi(t).
$$

The naive midpoint propagator

$$
U_{\rm mid}=e^{-i\tau H(t+\tau/2)/\hbar}
$$

is second order. It misses higher Magnus terms containing $[H(t_1),H(t_2)]$. Here they are nonzero because the kinetic operator $p_x^2/(2m^*)$ does not commute with the time-dependent linear potential $x\cos\omega t$.

---

# Hamiltonian split required by `method.pdf`

Write (3) as

$$
H(t)=T+V(t),
$$

$$
T=\frac{p_x^2}{2m^*},
$$

$$
V(t)=V_c(x;k_y)+f(t)V_f(x),
$$

where, for the code,

$$
\begin{aligned}
V_c&=\frac{\hbar^2k_y^2}{2m^*}
+\frac{\hbar k_yeB_0}{2m^*L}x^2
+\frac{e^2B_0^2}{8m^*L^2}x^4,\\
f(t)&=\cos(\omega_{\rm ac}t),\qquad
V_f=\frac{eV_{\rm ac}}Lx.
\end{aligned}
$$

---

# Code realization of the split

`time_coeffs.py:build_split_operators` implements (16):

```python
kinetic = p2_mat / (2*m_star)
static_potential = (
    hbar**2*ky**2/(2*m_star)*np.eye(nmax_value)
    + hbar*ky*e*B0/(2*m_star*L)*x2_mat
    + e**2*B0**2/(8*m_star*L**2)*x4_mat
)
drive_potential = (e*Ve/L)*x_mat
```

`coordinate_potential(static_potential, drive_potential, t)` returns $V_c+\cos(\omega t)V_f$. The name is historical: these are matrices represented in the SHO basis, not diagonal arrays on an $x$ grid.

---

# Sixth-order Gauss–Legendre sampling

The method in `method.pdf` evaluates the potential at three sixth-order Gauss–Legendre nodes:

$$
c_1=\frac12-\frac{\sqrt{15}}{10},\quad
c_2=\frac12,\quad
c_3=\frac12+\frac{\sqrt{15}}{10},
$$

$$
V_j=V(t+c_j\tau),\qquad j=1,2,3.
$$

Direct code correspondence:

```python
Y6_2_C = np.array([0.5 - np.sqrt(15)/10,
                   0.5,
                   0.5 + np.sqrt(15)/10])

V1, V2, V3 = [coordinate_potential(..., t + c*dt)
              for c in Y6_2_C]
```

---

# Commutator-free Magnus construction

The exact propagator is $e^{\Omega}$, where the Magnus exponent contains nested commutators. `method.pdf` replaces it by a product of exponentials of linear combinations of $T$ and $V_j$.

For the optimized fourth-order kernel, define

$$
\bar V_1=a_1\cdot(V_1,V_2,V_3),\qquad
\bar V_2=a_2\cdot(V_1,V_2,V_3),
$$

and obtain $\bar V_3,\bar V_4$ by time reversal. The coefficients are

$$
a_1=\left(\frac{10+\sqrt{15}}{180},-\frac19,
\frac{10-\sqrt{15}}{180}\right),
$$

$$
a_2=\left(\frac{15+8\sqrt{15}}{90},\frac23,
\frac{15-8\sqrt{15}}{90}\right).
$$

---

# Coefficients in the implementation

The constants in `time_coeffs.py` are a direct transcription of Eq. (17) of `method.pdf`:

```python
Y6_2_A1 = np.array([(10 + np.sqrt(15))/180,
                    -1/9,
                    (10 - np.sqrt(15))/180])
Y6_2_A2 = np.array([(15 + 8*np.sqrt(15))/90,
                    2/3,
                    (15 - 8*np.sqrt(15))/90])
```

and the code enforces time reversal by reversing these coefficients:

```python
Vbar3 = A2[2]*V1 + A2[1]*V2 + A2[0]*V3
Vbar4 = A1[2]*V1 + A1[1]*V2 + A1[0]*V3
```

This symmetric arrangement is essential for even order and good long-time geometric behaviour.

---

# The special $[212]$ commutator

For $H=T+V(x,t)$, the nested commutator used by the paper has the special coordinate-space form

$$
[\alpha_2,[\alpha_1,\alpha_2]]\psi
=-\frac{5\tau^3}{3m^*}
\left[V'(x,t+c_3\tau)-V'(x,t+c_1\tau)\right]^2\psi,
$$

in the $\hbar=1$ convention of `method.pdf`.

Unlike generic commutators, this is a multiplication operator in coordinate space. Incorporating it as a force-gradient correction upgrades the symmetric four-exponential scheme from fourth to sixth order.

---

# Optional derivative-assisted sixth-order propagator

Equation (19) of `method.pdf`, in atomic units, is

$$
\begin{aligned}
\Upsilon_2^{[6]}(t,\tau)=&\
e^{-i\tau(\bar V_4+\tau^2\widetilde V)}
e^{-i\tau(T+\bar V_3)/2}\\
&\times e^{-i\tau(T+\bar V_2)/2}
e^{-i\tau(\bar V_1+\tau^2\widetilde V)},
\end{aligned}
$$

where

$$
\widetilde V=-\frac{5}{3m^*\,43200}
\left[V'_3-V'_1\right]^2.
$$

The local error is $O(\tau^7)$ and the global error at fixed final time is $O(\tau^6)$, provided the exponential actions are sufficiently accurate. This remains available as `method="y6_2_lanczos"`, but is no longer the default.

---

# Why the correction is especially simple here

With (16), only the AC part changes between the three nodes:

$$
V'_3-V'_1=[f(t+c_3\tau)-f(t+c_1\tau)]\frac{eV_{\rm ac}}L.
$$

This has no $x$ dependence. Hence

$$
\widetilde V(t,\tau)=
-\frac{[f_3-f_1]^2}{25920m^*}
\left(\frac{eV_{\rm ac}}L\right)^2I.
$$

The quartic static potential does not contribute to $V'_3-V'_1$, because it is identical at all three time nodes. This is why the force-gradient correction remains a scalar identity matrix even in the SHO representation.

---

# Exact code location of the force-gradient term

`time_coeffs.py:y6_2_modified_potential` implements (20):

```python
f1 = np.cos(w_ac*(t + Y6_2_C[0]*tau))
f3 = np.cos(w_ac*(t + Y6_2_C[2]*tau))
drive_gradient = e*Ve/L
scalar = -((f3 - f1)**2)*drive_gradient**2 / (m_star*25920)
return scalar*np.eye(nmax_value)
```

The extra $\tau^2$ multiplying this quantity is applied in the sixth-order propagator displayed above:

```python
Vbar1 + dt**2*V_tilde
Vbar4 + dt**2*V_tilde
```

---

# Exact code location of the four force-gradient exponentials

The following four calls in `time_evolute_y6_2_lanczos` are the four factors of the displayed sixth-order formula, applied from right to left to the ket:

```python
psi = exp_action(Vbar1 + dt**2*V_tilde, psi, -1j*dt/hbar)
psi = exp_action(kinetic + Vbar2,          psi, -1j*dt/(2*hbar))
psi = exp_action(kinetic + Vbar3,          psi, -1j*dt/(2*hbar))
psi = exp_action(Vbar4 + dt**2*V_tilde, psi, -1j*dt/hbar)
```

In the real source, `exp_action` is `lanczos_expm_multiply`, and the positional arguments are written explicitly. The ordering is correct: the rightmost exponential in the displayed formula acts first.

---

# Default: fully commutator-free sixth-order propagator

The simulation defaults to the paper's Eq. (18), $\Upsilon_3^{[6]}$. It has five exponentials and contains no explicit commutator and no force-gradient/potential-derivative term:

$$
\Upsilon_3^{[6]}=
e^{-i\tau\bar V_5}
e^{-i\tau(a_2T+\bar V_4)}
e^{-i\tau(a_3T+\bar V_3)}
e^{-i\tau(a_2T+\bar V_2)}
e^{-i\tau\bar V_1}.
$$

The same three Gauss--Legendre samples $V_1,V_2,V_3$ are used. With $\bar V_i=\sum_j a_{i,j}V_j$, the paper's coefficients are

$$
\begin{aligned}
a_1 &= (0.0199409626509361,0,-0.0199409626509361),\\
a_2 &= (0.4882524910228222,-0.0046136830175631,0.0834019108602183),\\
a_3 &= (-0.2938766241052627,0.4536718104795706,-0.2938766241052627),
\end{aligned}
$$

with $\bar V_4$ and $\bar V_5$ formed by reversing $a_2$ and $a_1$, respectively. The kinetic coefficients are $a_2=\sum_j a_{2,j}=0.5670407188654774$ and $a_3=\sum_j a_{3,j}=-0.1340814377309549$. `time_coeffs.py:time_evolute_y6_3_lanczos` applies these factors right-to-left, and `time_evolute` selects it by default with `method="y6_3_lanczos"`.

---

# Exponential action: what “Lanczos” means in this repository

For each Hermitian matrix $K$, the algorithm needs only

$$
e^{\alpha K}\mathbf c,
$$

not the dense matrix $e^{\alpha K}$. The Krylov idea is to approximate this vector in

$$
\mathcal K_m(K,\mathbf c)=\mathrm{span}\{\mathbf c,K\mathbf c,\ldots,K^{m-1}\mathbf c\}.
$$

`lanczos.py:lanczos_expm_multiply` delegates the action to

```python
scipy.sparse.linalg.expm_multiply(dt * H, psi0)
```

It is therefore a robust SciPy Krylov exponential action, not a hand-written fixed-dimension Lanczos recurrence.

---

# Important implementation distinction

`method.pdf` gains efficiency in a coordinate-grid representation because the outer potential exponentials are diagonal and can be applied pointwise.

This code uses an SHO basis. In that basis $x$, $x^2$, and $x^4$ are matrices, so even the outer $\bar V_1$ and $\bar V_5$ factors are not diagonal. The code therefore evaluates all five $\Upsilon_3$ exponentials with `expm_multiply`.

<div class="box">

The order conditions and the force-gradient formula remain valid because they are operator identities. What is lost is only the FFT/pointwise cost advantage advertised in `method.pdf`, not the mathematical propagator.

</div>

The API parameter `lanczos_k` is currently passed through but ignored by SciPy’s adaptive `expm_multiply`; it is not a fixed Krylov dimension.

---

# Current time-evolution data flow

```text
(ky, Vac, state index n)
        |
        v
build_split_operators: T, Vc, Vf
        |
        v
eigenvector n of H(0), with n=0 the ground state  --> initial coefficients c(0)
        |
        v
for every time step: GL nodes --> V1,V2,V3 --> Vbar1,...,Vbar4
        |                                      + force-gradient Vtilde
        v
four Krylov exponential actions --> c(t + dt)
        |
        +--> c_snapshots --> density, current, harmonics
```

The relevant public entry point is `time_coeffs.py:time_evolute`; `time_visualise.py` and `current_visualise.py` call it.

---

# Verification required for a time-dependent calculation

1. **Norm:** $|\mathbf c^\dagger\mathbf c-1|$ must stay below tolerance.
2. **Time-step convergence:** compare fixed-final-time results at $\tau$, $\tau/2$, and $\tau/4$. In the asymptotic regime, the difference should decrease approximately as $2^6$.
3. **Basis convergence:** increase `nmax`; a driven state can populate much higher SHO levels than the initial static state.
4. **Independent reference:** use `time_evolute_midpoint_dense` with a sufficiently small step for modest $N$.
5. **Analytic regression:** `tests/test_driven_sho.py` compares the evolution pipeline with the exact driven harmonic oscillator motion.

Do not infer sixth-order accuracy from norm conservation alone: a unitary but inaccurate propagator can conserve norm.

---

# Gauge-covariant Schrödinger field theory

We now derive the current density instead of postulating it. Let $\Psi(\mathbf r,t)$ be a complex Schrödinger field of charge $q=-e$ in prescribed background fields $(\phi,\mathbf A)$.

Define covariant derivatives

$$
D_t=\partial_t+\frac{iq}{\hbar}\phi,
\qquad
\mathbf D=\boldsymbol\nabla-\frac{iq}{\hbar}\mathbf A.
$$

Then $-i\hbar\mathbf D=\mathbf p-q\mathbf A$. For $q=-e$, this is precisely $\mathbf p+e\mathbf A$.

---

# Gauge-invariant Lagrangian density

The nonrelativistic matter Lagrangian density is

$$
\mathcal L=
\frac{i\hbar}{2}\left[\Psi^*D_t\Psi-(D_t\Psi)^*\Psi\right]
-\frac{\hbar^2}{2m^*}(\mathbf D\Psi)^*\cdot(\mathbf D\Psi).
$$

Expanding the first term gives

$$
\mathcal L=
\frac{i\hbar}{2}(\Psi^*\partial_t\Psi-\partial_t\Psi^*\Psi)
-q\phi|\Psi|^2
-\frac{\hbar^2}{2m^*}|\mathbf D\Psi|^2.
$$

It is invariant under the local gauge transformation

$$
\Psi\to e^{iq\chi/\hbar}\Psi,\quad
\mathbf A\to\mathbf A+\nabla\chi,\quad
\phi\to\phi-\partial_t\chi.
$$

---

# Euler–Lagrange equation gives the Hamiltonian

Varying $S=\int dt\,d^2r\,\mathcal L$ with respect to $\Psi^*$ yields

$$
i\hbar\partial_t\Psi=
\left[\frac{1}{2m^*}(-i\hbar\nabla-q\mathbf A)^2+q\phi\right]\Psi.
$$

For $q=-e$ and the potentials chosen above,

$$
i\hbar\partial_t\Psi=
\left[\frac{(-i\hbar\nabla+e\mathbf A)^2}{2m^*}
+\frac{eV_e(t)}Lx\right]\Psi,
$$

which is exactly the total Hamiltonian derived at the beginning of this report. Thus the field-theory and single-particle formulations use the same sign conventions.

---

# Noether derivation of probability conservation

The global subgroup $\Psi\to e^{i\alpha}\Psi$ of the gauge symmetry gives the conserved probability density

$$
n(\mathbf r,t)=|\Psi|^2
$$

and probability current

$$
\mathbf J_{\rm prob}=
\frac{\hbar}{m^*}\operatorname{Im}(\Psi^*\mathbf D\Psi).
$$

Using the Euler–Lagrange equation above and its complex conjugate gives

$$
\partial_t n+\nabla\cdot\mathbf J_{\rm prob}=0.
$$

This result is gauge invariant because $\Psi^*\mathbf D\Psi$ is gauge invariant.

---

# Charge density and charge current from the Lagrangian

Multiplying the conserved particle current by charge $q$ gives

$$
\rho=q|\Psi|^2,
$$

$$
\mathbf j=q\mathbf J_{\rm prob}
=\frac{q\hbar}{m^*}\operatorname{Im}(\Psi^*\mathbf D\Psi).
$$

Equivalently, with mechanical momentum $\boldsymbol\pi=-i\hbar\nabla-q\mathbf A$,

$$
\boxed{\ \mathbf j=\frac{q}{m^*}\operatorname{Re}(\Psi^*\boldsymbol\pi\Psi)\ }.
$$

Varying the Lagrangian density with respect to the external vector potential also yields this source current: $\delta\mathcal L/\delta\mathbf A=\mathbf j$. Charge conservation is

$$
\partial_t\rho+\nabla\cdot\mathbf j=0.
$$

---

# Paramagnetic and diamagnetic pieces

Expanding the charge-current expression gives

$$
\mathbf j=
\frac{q\hbar}{m^*}\operatorname{Im}(\Psi^*\nabla\Psi)
-\frac{q^2}{m^*}\mathbf A|\Psi|^2.
$$

The first term is usually called paramagnetic and the second diamagnetic. Neither is separately gauge invariant; their sum is.

For electrons ($q=-e$),

$$
\mathbf j=-\frac{e\hbar}{m^*}\operatorname{Im}(\Psi^*\nabla\Psi)
-\frac{e^2}{m^*}\mathbf A|\Psi|^2.
$$

Omitting the $A_y$ term would give an incorrect $y$ current and would fail the static band-slope identity (8).

---

# Channel-resolved current density

Insert

$$
\Psi=L_y^{-1/2}e^{ik_yy}\psi_{k_y}(x,t)
$$

into (28). Since $A_x=0$ and $A_y=B_0x^2/(2L)$,

$$
j_x(x,t)=-\frac{e\hbar}{m^*L_y}
\operatorname{Im}[\psi^*\partial_x\psi],
$$

$$
j_y(x,t)=-\frac{e}{m^*L_y}
\left(\hbar k_y+\frac{eB_0x^2}{2L}\right)|\psi|^2.
$$

Because the channel is uniform in $y$, $\partial_yj_y=0$ and the channel continuity equation is $\partial_t\rho+\partial_xj_x=0$.

---

# What `current.py` computes and `current_visualise.py` plots

The 1D channel wavefunction is normalized only in $x$, so the code plots the line current

$$
\mathcal J_\mu(x,t)\equiv L_yj_\mu(x,t).
$$

Equations (32)–(33) become

```python
jx_line = -e*hbar/m_star * np.imag(np.conj(psi)*dpsi_dx)
mechanical_py = hbar*ky + e*B0*x**2/(2*L)
jy_line = -e/m_star * mechanical_py[None, :] * density
```

This is implemented by `current.py:line_current_density` and called by `current_visualise.py`. The $[None, :]$ broadcasts the position-dependent mechanical momentum over all time snapshots.

---

# Current directly from SHO coefficients

The code reconstructs

$$
\psi(x,t)=\sum_{n=0}^{N-1}c_n(t)\phi_n(x).
$$

Instead of finite-differencing a sampled wavefunction, it uses the analytic basis derivative

$$
\partial_x\phi_n=
\frac{\sqrt n\,\phi_{n-1}-\sqrt{n+1}\,\phi_{n+1}}
{\sqrt2\ell_0}.
$$

`current.py:sho_basis_and_derivative` implements this derivative identity, and `current.py:wavefunction_and_derivative` then evaluates

```python
psi      = c_snapshots @ basis
dpsi_dx = c_snapshots @ dbasis_dx
density = np.abs(psi)**2
```

This is the appropriate derivative within the truncated SHO representation.

---

# Integrated channel current and velocity

For the normalized channel, define

$$
\langle v_x\rangle=
\frac{\hbar}{m^*}\int dx\,\operatorname{Im}(\psi^*\partial_x\psi),
$$

$$
\langle v_y\rangle=
\frac1{m^*}\int dx\,
\left(\hbar k_y+\frac{eB_0x^2}{2L}\right)|\psi|^2.
$$

Then

$$
\int dx\,\mathcal J_\mu(x,t)=-e\langle v_\mu\rangle.
$$

For a static eigenstate, combining the integrated-current relation with the band-slope identity gives

$$
\int dx\,\mathcal J_y=-\frac{e}{\hbar}\frac{\partial E_n}{\partial k_y}.
$$

This identity should be included as a numerical unit/regression test.

---

# From one channel to a physical many-electron current

One occupied $k_y$ state contributes a density proportional to $1/L_y$. In the thermodynamic limit, the sum over $k_y$ becomes

$$
\frac1{L_y}\sum_{k_y}\longrightarrow\int\frac{dk_y}{2\pi}.
$$

For occupations $f_{n,k_y}$ and spin degeneracy $g_s$,

$$
j_\mu^{\rm tot}(x,t)=g_s\sum_n\int\frac{dk_y}{2\pi}
f_{n,k_y}\,\mathcal J_{\mu,nk_y}(x,t).
$$

The present animation is only one orbital’s contribution $\mathcal J_{\mu,nk_y}$. A DC Hall conductance additionally requires a specification of reservoirs, filling, temperature, disorder, and boundary conditions; it cannot be inferred from a closed single-channel propagation alone.

---

# Time-dependent observables and harmonic analysis

Once $\mathcal J_\mu(x,t)$ has converged, useful outputs are

$$
I_\mu^{(k)}(t)=\int dx\,\mathcal J_\mu(x,t),
$$

and, for a periodic response with period $T_d=2\pi/\omega_{\rm ac}$,

$$
I_\mu(r\omega_{\rm ac})=
\frac1{T_d}\int_{t_0}^{t_0+T_d}
dt\,I_\mu(t)e^{ir\omega_{\rm ac}t}.
$$

For high-harmonic analysis, first remove the switch-on transient. In a closed system, verify that observables are periodic from cycle to cycle before interpreting a finite-window Fourier peak as a steady Floquet response.

---

# Implemented current regression tests

`tests/test_current.py` now executes the following checks on the actual simulator:

| Test | Quantity verified | Numerical setup |
|---|---|---|
| Static band/current test | $-e^{-1}\int\mathcal J_y dx=(1/\hbar)dE/dk_y$ | Dense static eigenstates in both the double-well and snake-state regimes |
| Static $x$-current test | $\mathcal J_x=0$ for a real stationary eigenvector | Same eigenstates |
| Driven Ehrenfest test | $d\langle x\rangle/dt=-e^{-1}\int\mathcal J_xdx$ | Sixth-order CFM propagation with a reduced test basis |
| Driven continuity test | $\partial_t(-e|\psi|^2)+\partial_x\mathcal J_x\simeq0$ | Central finite differences on the saved $x$–$t$ grid |

The reusable implementation is in `current.py`; it defines SHO derivatives, line-current densities, integrated currents, and the sampled continuity residual without importing the animation script.

---

# Current-specific verification criteria

<div class="cols">
<div class="box">

### Static checks

For every selected eigenstate:

$$
\langle v_y\rangle \stackrel{?}{=}
\frac1\hbar\frac{dE}{dk_y},
$$

with an independently converged central difference, and

$$
\int dx\,\mathcal J_y\stackrel{?}{=}-e\langle v_y\rangle.
$$

</div>
<div class="box">

### Dynamic checks

For the propagated state:

$$
\frac{d\langle x\rangle}{dt}\stackrel{?}{=}\langle v_x\rangle,
$$

and the discrete residual

$$
R=\partial_t[-e|\psi|^2]+\partial_x\mathcal J_x
$$

must converge toward zero as $dt$ and $dx$ are refined.

</div>
</div>

---

# Implementation map

| Physics / numerical concept | File and function |
|---|---|
| Atomic-unit parameters | `params.py` |
| SHO $x,x^2,x^4,p_x^2$ matrices | `basis.py:operators_matrixize` |
| Static polynomial $H_{k_y}(V_e)$ | `hamiltonian.py:build_H` |
| Dense and ARPACK static spectra | `spectrum.py:compute_spectrum` |
| SHO wavefunction reconstruction | `wave_function.py:QHO_basis` |
| $T+V_c+f(t)V_f$ split | `time_coeffs.py:build_split_operators` |
| GL nodes and $\Upsilon_3$ coefficients | `Y6_3_C`, `Y6_3_A1`, `Y6_3_A2`, `Y6_3_A3` |
| Default fully commutator-free propagation | `time_evolute_y6_3_lanczos` |
| Optional scalar force-gradient correction | `y6_2_modified_potential` |
| Krylov exponential action | `lanczos.py:lanczos_expm_multiply` |
| Evolved density | `time_coeffs.py:compute_prob_evolute` |
| SHO reconstruction and current observables | `current.py` |
| Current animation frontend | `current_visualise.py` |
| Current regression tests | `tests/test_current.py` |

---

# Conclusions

- The code’s static Hamiltonian is the correct SHO-basis representation of the full 2D minimal-coupling problem after separation of $k_y$.
- Static states are obtained by finite-dimensional Hermitian diagonalization—dense `eigh` or ARPACK `eigsh`—not by the analytic formal solution.
- The formal critical zero mode has an exact algebraic factorization origin but is non-normalizable on the full line; numerical flatness should be quantified by residual bandwidth and basis convergence.
- `time_coeffs.py` defaults to the fully commutator-free $\Upsilon_3^{[6]}$ method from `method.pdf`: three GL samples, symmetric linear combinations, and five Krylov exponential actions per time step. The derivative-assisted $\Upsilon_2^{[6]}$ force-gradient method remains an explicit option.
- The charge current follows uniquely from the gauge-covariant Schrödinger Lagrangian. `current.py` implements the correct mechanical-momentum current, including the $A_y$ (diamagnetic) term; `current_visualise.py` visualizes it.

---

# References and run commands

- Y.-T. Huang, C.-C. Kaun, and C.-H. Chang, *Electrically Controllable Landau Levels in Two-Dimensional Electron Gases under Nonuniform Magnetic Fields*, arXiv:2601.05064v3; local file: `2601.05064v3.pdf`.
- S. Blanes, F. Casas, and A. Murua, *Exponential propagators for the Schrödinger equation*; local file: `method.pdf`. The default propagator is its Eq. (18); Eq. (19) remains available as the derivative-assisted alternative.

```bash
python spectrum.py
python time_visualise.py
python current_visualise.py
python -m unittest discover -s tests -v
```
