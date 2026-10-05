#include "qp/schrodinger.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "qp/tridiagonal.hpp"

namespace qp {

namespace {

constexpr double kInvSqrt2Pi = 0.39894228040143267794;

/// One RK4 step of the Lamperti ODE dx/dy = sigma(x).
double rk4_step(const EffectiveVol& vol, double x, double dy) {
    const double k1 = vol.sigma(x);
    const double k2 = vol.sigma(x + 0.5 * dy * k1);
    const double k3 = vol.sigma(x + 0.5 * dy * k2);
    const double k4 = vol.sigma(x + dy * k3);
    return x + dy * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0;
}

/// Integrates dx/dy = sigma(x), x(0) = 0, from y_from to y_to using `substeps` RK4 steps.
double integrate_lamperti(const EffectiveVol& vol, double x, double y_from, double y_to,
                          int substeps) {
    const double dy = (y_to - y_from) / substeps;
    for (int i = 0; i < substeps; ++i) x = rk4_step(vol, x, dy);
    return x;
}

}  // namespace

SchrodingerProblem::SchrodingerProblem(const EffectiveVolParams& params, double t_max,
                                       const GridSpec& grid)
    : vol_(params), grid_(grid), t_max_(t_max) {
    if (!(t_max > 0.0)) throw std::invalid_argument("SchrodingerProblem: t_max must be > 0");
    if (grid.n_space < 16) throw std::invalid_argument("SchrodingerProblem: n_space too small");
    if (grid.n_time < 1) throw std::invalid_argument("SchrodingerProblem: n_time must be >= 1");

    const std::size_t n = static_cast<std::size_t>(grid.n_space);
    const double sigma0 = vol_.sigma(0.0);
    const double b0 = -0.5 * (sigma0 + vol_.jet(0.0).d1);
    const double half_width = grid.width_sd * std::sqrt(t_max);
    const double centre = b0 * t_max;  // follow the drift of Y so the density stays centred
    const double y_lo = centre - half_width;
    h_ = 2.0 * half_width / static_cast<double>(n - 1);

    y_.resize(n);
    for (std::size_t i = 0; i < n; ++i) y_[i] = y_lo + h_ * static_cast<double>(i);

    // Invert the Lamperti map by integrating dx/dy = sigma(x) outwards from y = 0.
    x_.resize(n);
    constexpr int kSubsteps = 4;
    const auto first_pos = static_cast<std::size_t>(
        std::lower_bound(y_.begin(), y_.end(), 0.0) - y_.begin());
    double xc = 0.0, yc = 0.0;
    for (std::size_t i = first_pos; i < n; ++i) {
        xc = integrate_lamperti(vol_, xc, yc, y_[i], kSubsteps);
        yc = y_[i];
        x_[i] = xc;
    }
    xc = 0.0, yc = 0.0;
    for (std::size_t i = first_pos; i-- > 0;) {
        xc = integrate_lamperti(vol_, xc, yc, y_[i], kSubsteps);
        yc = y_[i];
        x_[i] = xc;
    }

    sigma_.resize(n);
    b_.resize(n);
    V_.resize(n);
    B_.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
        const auto j = vol_.jet(x_[i]);
        const double b = -0.5 * (j.sigma + j.d1);
        const double db_dy = j.sigma * (-0.5 * (j.d1 + j.d2));  // chain rule: d/dy = sigma d/dx
        sigma_[i] = j.sigma;
        b_[i] = b;
        V_[i] = 0.5 * (b * b + db_dy);
        B_[i] = -0.5 * (x_[i] + std::log(j.sigma / sigma0));
    }
}

std::vector<double> Density::density_x(const std::vector<double>& sigma) const {
    std::vector<double> p(q.size());
    for (std::size_t i = 0; i < q.size(); ++i) p[i] = q[i] / sigma[i];
    return p;
}

double Density::mass() const {
    double m = 0.0;
    for (std::size_t i = 0; i < q.size(); ++i) {
        const double w = (i == 0 || i + 1 == q.size()) ? 0.5 * h : h;
        m += w * q[i];
    }
    return m;
}

double Density::forward_ratio() const {
    double m0 = 0.0, m1 = 0.0;
    for (std::size_t i = 0; i < q.size(); ++i) {
        const double w = (i == 0 || i + 1 == q.size()) ? 0.5 * h : h;
        m0 += w * q[i];
        m1 += w * q[i] * std::exp(x[i]);
    }
    return m1 / m0;
}

std::vector<Density> solve_densities(const SchrodingerProblem& problem, std::vector<double> times) {
    if (times.empty()) return {};
    std::sort(times.begin(), times.end());
    if (!(times.front() > 0.0)) throw std::invalid_argument("solve_densities: times must be > 0");
    if (times.back() > problem.t_max() * (1.0 + 1e-12)) {
        throw std::invalid_argument("solve_densities: time beyond the problem's t_max");
    }

    const auto& y = problem.y();
    const auto& V = problem.potential();
    const auto& B = problem.gauge();
    const double h = problem.h();
    const std::size_t n = y.size();
    const std::size_t m = n - 2;  // interior unknowns; Dirichlet psi = 0 on both ends
    const GridSpec& grid = problem.grid();

    // Smoothed Dirac delta: the exact short-time kernel of dY = b dt + dW to leading order.
    const double t0 = std::min(16.0 * h * h, 0.25 * times.front());
    const double b_at_0 = problem.vol().jet(0.0).sigma + problem.vol().jet(0.0).d1;
    const double mu = -0.5 * b_at_0 * t0;
    std::vector<double> psi(m);
    for (std::size_t i = 0; i < m; ++i) {
        const double z = (y[i + 1] - mu);
        const double q0 = kInvSqrt2Pi / std::sqrt(t0) * std::exp(-0.5 * z * z / t0);
        psi[i] = q0 * std::exp(-B[i + 1]);
    }

    // H = -1/2 d^2/dy^2 + V on the interior nodes.
    const double kinetic = 0.5 / (h * h);
    std::vector<double> H_off(m, -kinetic), H_diag(m);
    for (std::size_t i = 0; i < m; ++i) H_diag[i] = 2.0 * kinetic + V[i + 1];

    std::vector<Density> out;
    out.reserve(times.size());
    std::vector<double> rhs(m), lower(m), diag(m), upper(m), b_lower(m), b_diag(m), b_upper(m);
    const double span = times.back() - t0;
    double t = t0;
    int rannacher_left = grid.rannacher_steps;

    for (double target : times) {
        const double seg = target - t;
        if (seg > 0.0) {
            const int steps = std::max(
                {1, rannacher_left, static_cast<int>(std::ceil(grid.n_time * seg / span))});
            const double dt = seg / steps;

            // A = I + dt/2 H (shared by CN and by implicit-Euler half-steps), Bm = I - dt/2 H.
            for (std::size_t i = 0; i < m; ++i) {
                lower[i] = upper[i] = 0.5 * dt * H_off[i];
                diag[i] = 1.0 + 0.5 * dt * H_diag[i];
                b_lower[i] = b_upper[i] = -0.5 * dt * H_off[i];
                b_diag[i] = 1.0 - 0.5 * dt * H_diag[i];
            }
            const TridiagonalLU A(lower, diag, upper);

            for (int k = 0; k < steps; ++k) {
                if (rannacher_left > 0) {
                    // Two implicit-Euler half-steps damp the high-frequency modes of the start-up.
                    A.solve(psi);
                    A.solve(psi);
                    --rannacher_left;
                } else {
                    tridiagonal_multiply(b_lower, b_diag, b_upper, psi, rhs);
                    A.solve(rhs);
                    psi.swap(rhs);
                }
            }
            t = target;
        }

        Density d;
        d.t = target;
        d.h = h;
        d.y = y;
        d.x = problem.x();
        d.q.assign(n, 0.0);
        for (std::size_t i = 0; i < m; ++i) d.q[i + 1] = std::max(0.0, std::exp(B[i + 1]) * psi[i]);
        out.push_back(std::move(d));
    }
    return out;
}

Density solve_density(const SchrodingerProblem& problem, double t) {
    return solve_densities(problem, {t}).front();
}

}  // namespace qp
