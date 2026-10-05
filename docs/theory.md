# From a volatility smile to a quantum potential well

This note derives the mapping implemented in `cpp/src/schrodinger.cpp`. Every step is an exact
change of variables; the only modelling assumption is the parametric form of the effective
local variance in step 1.

## 1. Projecting stochastic volatility onto one dimension

Let $X_t = \ln(F_t/F_0)$ be log forward moneyness under the pricing measure, driven by a
stochastic-volatility model

$$
dX_t = -\tfrac12 v_t\,dt + \sqrt{v_t}\,dW_t .
$$

European option prices depend only on the marginal law of $X_T$. Gyöngy's (1986) Markovian
projection theorem states that the one-dimensional diffusion

$$
dX_t = -\tfrac12 \sigma^2(X_t)\,dt + \sigma(X_t)\,dW_t,
\qquad \sigma^2(x) = \mathbb{E}[\,v_t \mid X_t = x\,]
$$

has the same marginals. The stochastic-volatility model is therefore priced exactly by a 1D
problem once the conditional expected variance $\sigma^2(x)$ is known. Instead of deriving it from
a particular SV model, we parameterise it directly, per expiry, with the SVI shape

$$
\sigma^2(x) = a + b\left(\rho\,(x-m) + \sqrt{(x-m)^2 + s^2}\right),
$$

positive whenever $b \ge 0$, $|\rho|<1$, $s>0$ and $a + b s\sqrt{1-\rho^2} > 0$. The calibration
works in the coordinates $(v_{\min}, b, \rho, m, s)$ with $a = v_{\min} - b s \sqrt{1-\rho^2}$, so plain
box bounds guarantee positivity.

The model is time-homogeneous within an expiry. Projected Heston variance is not, so this is an
approximation. For one short-dated slice it costs little.

## 2. Lamperti transform: unit diffusion

With $y = \int_0^x dx'/\sigma(x')$ (so $dx/dy = \sigma$), Itô's lemma gives

$$
dY_t = b(Y_t)\,dt + dW_t, \qquad b = -\tfrac12\left(\sigma + \sigma'\right),
$$

where $\sigma' = d\sigma/dx$. The solver obtains $x(y)$ by integrating $dx/dy = \sigma(x)$ outwards
from $y=0$ with RK4, so no root-finding is needed.

## 3. Gauge transform: Fokker-Planck becomes Schrödinger

The density $q(y,t)$ of $Y_t$ solves $q_t = -(b q)_y + \tfrac12 q_{yy}$. Substituting
$q = e^{B}\psi$ with $B' = b$ makes every first-derivative term cancel:

$$
\frac{\partial \psi}{\partial t} = \tfrac12 \psi_{yy} - V(y)\,\psi
\quad\Longleftrightarrow\quad
-\frac{\partial \psi}{\partial t} = H\psi, \qquad
H = -\tfrac12\frac{d^2}{dy^2} + V(y), \qquad
V = \tfrac12\left(b^2 + b_y\right).
$$

This is the time-dependent Schrödinger equation $i\partial_\tau\psi = H\psi$ continued to imaginary
time $\tau = -it$ (a Wick rotation), for a particle of unit mass with $\hbar = 1$. The gauge factor
has a closed form, which the code uses directly:

$$
B(y) = \int_0^y b\,dy' = -\tfrac12\int_0^x \left(1 + \frac{\sigma'}{\sigma}\right)dx'
= -\tfrac12\left[x + \ln\frac{\sigma(x)}{\sigma(0)}\right].
$$

**Sanity check.** For constant $\sigma$: $x = \sigma y$, $b = -\sigma/2$, $V \equiv \sigma^2/8$. The particle is
free, $\psi$ is a heat kernel, and the model *is* Black-Scholes. The test suite checks this
limit to below 0.01 index points.

## 4. Why the smile is a potential *well*

As $|x|\to\infty$ the SVI variance grows linearly, $\sigma \sim c\sqrt{|x|}$. Then
$y \sim 2\sqrt{|x|}/c$, so $\sigma \sim c^2|y|/2$, $b \sim -c^2|y|/4$, and

$$
V(y) \sim \frac{c^4}{32}\,y^2 .
$$

The potential is **asymptotically harmonic**, so $H$ has a purely discrete spectrum
$E_0 < E_1 < \dots$ and the transition density has the eigen-expansion

$$
q(y,t) = e^{B(y)}\sum_n e^{-E_n t}\,\phi_n(y)\,\phi_n(0).
$$

The linear-wing growth that Roger Lee's moment formula allows for implied variance is the same
condition that confines the quantum particle. The skew $\rho$ tilts the well. The curvature $s$ and
the shift $m$ shape its floor, and in short-dated SPX slices that floor often has a shallow dip
where $b_y < 0$. `cpp/src/spectral.cpp` computes the bound states with Sturm-sequence bisection and
inverse iteration. The tests check the harmonic-oscillator levels $E_n = \omega(n+\tfrac12)$ and that
the expansion reproduces the PDE prices.

## 5. Numerics

* **Grid.** Uniform in $y$, centred on the drift $b(0)T$, half-width $8\sqrt{T}$ (the $Y$-process
  has unit diffusion, so this is 8 standard deviations), Dirichlet $\psi=0$ at the ends.
* **Initial condition.** The short-time Gaussian kernel of $Y$ at $t_0 = 16h^2$, a smoothed Dirac
  delta at $y=0$. Starting from a literal delta would put all the error into the first steps.
* **Time stepping.** Crank-Nicolson, $(I + \tfrac{\Delta t}{2}H)\psi^{n+1} = (I - \tfrac{\Delta t}{2}H)\psi^n$,
  with Rannacher start-up: the first steps are replaced by implicit-Euler half-steps, which reuse
  the *same* matrix. $H$ is time-independent, so the tridiagonal LU factorisation is computed
  once and each step is one $O(N)$ sweep.
* **Pricing.** $C(K) = D\int (F e^{x(y)} - K)^+ q(y)\,dy$ by the trapezoid rule, with the payoff kink
  resolved inside its cell, after renormalising mass and matching $\mathbb{E}[e^X]=1$ (an $O(h^2)$
  martingale correction). Calls integrate the upper tail and puts the lower tail, which avoids
  cancellation for far out-of-the-money contracts. Prefix sums make $K$ strikes cost
  $O(N + K\log N)$.
* **Convergence.** The test suite shows second order: halving $h$ and $\Delta t$ divides the
  error by 3.9-4.0.
* **Independent check.** A separate backward Crank-Nicolson solver for the local-vol PDE in $x$
  (`cpp/tests/test_local_vol_crosscheck.cpp`) shares no code with the forward Schrödinger
  solver. The two agree to $5\times10^{-4}$ index points. When the backward grid was too
  coarse, refining it moved its answer toward the Schrödinger value.

## References

* I. Gyöngy (1986), *Mimicking the one-dimensional marginal distributions of processes having an
  Itô differential*, Probab. Theory Relat. Fields 71.
* H. Risken (1989), *The Fokker-Planck Equation*, ch. 5 (the Fokker-Planck/Schrödinger mapping).
* B. E. Baaquie (2004), *Quantum Finance*, Cambridge University Press.
* J. Gatheral (2006), *The Volatility Surface*, Wiley (SVI; local vs. stochastic volatility).
* R. Lee (2004), *The moment formula for implied volatility at extreme strikes*, Math. Finance 14.
* B. Dumas, J. Fleming, R. Whaley (1998), *Implied volatility functions: empirical tests*,
  J. Finance 53 (the ad-hoc Black-Scholes benchmark).
* S. Heston (1993), *A closed-form solution for options with stochastic volatility*, RFS 6.
