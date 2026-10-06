#pragma once

#include <vector>

#include "qp/effective_vol.hpp"

namespace qp {

/// Discretisation settings for the imaginary-time Schrödinger solver.
struct GridSpec {
    int n_space = 1200;      ///< number of spatial nodes in y
    int n_time = 200;        ///< Crank-Nicolson steps to the longest horizon
    double width_sd = 8.0;   ///< half-width of the y-domain in units of sqrt(T_max)
    int rannacher_steps = 2; ///< leading CN steps replaced by two implicit-Euler half-steps each
};

/// The quantum-mechanical picture of a 1D effective-volatility diffusion.
///
/// Log forward moneyness follows dX = -sigma(X)^2/2 dt + sigma(X) dW (a martingale in F e^X).
///
///   1. Lamperti transform  y = \int_0^x dx'/sigma(x')  gives unit diffusion
///          dY = b(Y) dt + dW,   b = -(sigma + sigma')/2.
///   2. Gauge transform     q(y,t) = e^{B(y)} psi(y,t),  B' = b,
///      which in closed form is B(y) = -1/2 [ x(y) + ln(sigma(x(y)) / sigma(0)) ].
///   3. The Fokker-Planck equation for q becomes the imaginary-time Schrödinger equation
///          d psi/dt = -H psi,   H = -1/2 d^2/dy^2 + V(y),   V = (b^2 + b_y) / 2,
///      i.e. a particle of unit mass (hbar = 1) in the potential well V.
///
/// With constant sigma the well is flat (V = sigma^2/8) and the model is exactly Black-Scholes.
class SchrodingerProblem {
public:
    /// Builds a uniform y-grid wide enough for horizons up to t_max.
    SchrodingerProblem(const EffectiveVolParams& params, double t_max, const GridSpec& grid = {});
    /// Builds the problem on an explicit y-range [y_lo, y_hi] with grid.n_space nodes.
    SchrodingerProblem(const EffectiveVolParams& params, double t_max, double y_lo, double y_hi,
                       const GridSpec& grid);

    const std::vector<double>& y() const { return y_; }
    const std::vector<double>& x() const { return x_; }          ///< x(y): log forward moneyness
    const std::vector<double>& sigma() const { return sigma_; }  ///< sigma(x(y))
    const std::vector<double>& drift() const { return b_; }      ///< b(y)
    const std::vector<double>& potential() const { return V_; }  ///< V(y)
    const std::vector<double>& gauge() const { return B_; }      ///< B(y)
    double h() const { return h_; }
    double t_max() const { return t_max_; }
    const GridSpec& grid() const { return grid_; }
    const EffectiveVol& vol() const { return vol_; }

private:
    EffectiveVol vol_;
    GridSpec grid_;
    double t_max_;
    double h_;
    std::vector<double> y_, x_, sigma_, b_, V_, B_;
};

/// Risk-neutral density of log forward moneyness at one horizon, sampled on the y-grid.
struct Density {
    double t = 0.0;
    std::vector<double> y;  ///< uniform grid
    std::vector<double> x;  ///< log forward moneyness at each node
    std::vector<double> q;  ///< density with respect to y (integrates to ~1 in dy)
    double h = 0.0;         ///< grid spacing in y

    /// Density with respect to x: p(x) = q(y) / sigma(x).
    std::vector<double> density_x(const std::vector<double>& sigma) const;
    /// Total probability mass (trapezoid rule).
    double mass() const;
    /// E[e^X]; equals 1 for an exact martingale.
    double forward_ratio() const;
};

/// Propagates the wavefunction in imaginary time with Crank-Nicolson finite differences.
///
/// The initial condition is the short-time Gaussian kernel of the Y-process at t0 = 16 h^2
/// (a smoothed Dirac delta at y = 0). The Hamiltonian is time-independent, so the system matrix
/// is factorised once per segment and each step costs O(N). Returns one density per requested
/// time; a single solve therefore prices every maturity <= t_max.
std::vector<Density> solve_densities(const SchrodingerProblem& problem, std::vector<double> times);

/// Convenience overload for a single horizon.
Density solve_density(const SchrodingerProblem& problem, double t);

/// Densities under a time-dependent well: the effective variance is piecewise constant in time,
/// params[j] applying on (ends[j-1], ends[j]] with ends[-1] = 0.
///
/// Each segment has its own Lamperti map and potential. At a breakpoint the density is carried
/// over in x (it is the same diffusion, so X is continuous) onto a y-grid of the next segment
/// that covers its support plus `width_sd` standard deviations of the segment's duration.
/// It is still one positive diffusion: densities are non-negative and the marginals increase in
/// convex order, so the prices it produces carry neither butterfly nor calendar arbitrage.
/// `grid.n_time` steps are spread over [0, max(times)].
std::vector<Density> solve_densities_piecewise(const std::vector<EffectiveVolParams>& params,
                                               const std::vector<double>& ends,
                                               std::vector<double> times,
                                               const GridSpec& grid = {});

}  // namespace qp
