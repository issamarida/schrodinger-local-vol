#include "qp/schrodinger.hpp"

#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

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

namespace {

/// Centred y-range of half-width width_sd * sqrt(t_max), following the drift of Y.
std::pair<double, double> default_range(const EffectiveVolParams& params, double t_max,
                                        const GridSpec& grid) {
    if (!(t_max > 0.0)) throw std::invalid_argument("SchrodingerProblem: t_max must be > 0");
    const EffectiveVol vol(params);
    const double b0 = -0.5 * (vol.sigma(0.0) + vol.jet(0.0).d1);
    const double half_width = grid.width_sd * std::sqrt(t_max);
    const double centre = b0 * t_max;
    return {centre - half_width, centre + half_width};
}

}  // namespace

SchrodingerProblem::SchrodingerProblem(const EffectiveVolParams& params, double t_max,
                                       const GridSpec& grid)
    : SchrodingerProblem(params, t_max, default_range(params, t_max, grid).first,
                         default_range(params, t_max, grid).second, grid) {}

SchrodingerProblem::SchrodingerProblem(const EffectiveVolParams& params, double t_max,
                                       double y_lo, double y_hi, const GridSpec& grid)
    : vol_(params), grid_(grid), t_max_(t_max) {
    if (!(t_max > 0.0)) throw std::invalid_argument("SchrodingerProblem: t_max must be > 0");
    if (grid.n_space < 16) throw std::invalid_argument("SchrodingerProblem: n_space too small");
    if (grid.n_time < 1) throw std::invalid_argument("SchrodingerProblem: n_time must be >= 1");
    if (!(y_hi > y_lo)) throw std::invalid_argument("SchrodingerProblem: empty y-range");

    const std::size_t n = static_cast<std::size_t>(grid.n_space);
    const double sigma0 = vol_.sigma(0.0);
    h_ = (y_hi - y_lo) / static_cast<double>(n - 1);

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

namespace {

/// Advances psi (the interior nodes of `problem`) from t to every target time (ascending), with
/// `rate` Crank-Nicolson steps per unit time, and returns the density at each target.
std::vector<Density> propagate(const SchrodingerProblem& problem, std::vector<double>& psi,
                               double& t, const std::vector<double>& targets, double rate,
                               int rannacher_left) {
    const auto& V = problem.potential();
    const auto& B = problem.gauge();
    const double h = problem.h();
    const std::size_t n = problem.y().size();
    const std::size_t m = n - 2;  // interior unknowns; Dirichlet psi = 0 on both ends

    // H = -1/2 d^2/dy^2 + V on the interior nodes.
    const double kinetic = 0.5 / (h * h);
    std::vector<double> H_off(m, -kinetic), H_diag(m);
    for (std::size_t i = 0; i < m; ++i) H_diag[i] = 2.0 * kinetic + V[i + 1];

    std::vector<Density> out;
    out.reserve(targets.size());
    std::vector<double> rhs(m), lower(m), diag(m), upper(m), b_lower(m), b_diag(m), b_upper(m);

    for (double target : targets) {
        const double seg = target - t;
        if (seg > 0.0) {
            const int steps =
                std::max({1, rannacher_left, static_cast<int>(std::ceil(rate * seg))});
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
        d.y = problem.y();
        d.x = problem.x();
        d.q.assign(n, 0.0);
        for (std::size_t i = 0; i < m; ++i) d.q[i + 1] = std::max(0.0, std::exp(B[i + 1]) * psi[i]);
        out.push_back(std::move(d));
    }
    return out;
}

/// Smoothed Dirac delta at y = 0: the short-time kernel of the Y-process at t0, as psi.
std::vector<double> initial_psi(const SchrodingerProblem& problem, double t0) {
    const auto& y = problem.y();
    const auto& B = problem.gauge();
    const std::size_t m = y.size() - 2;
    const double b_at_0 = problem.vol().jet(0.0).sigma + problem.vol().jet(0.0).d1;
    const double mu = -0.5 * b_at_0 * t0;
    std::vector<double> psi(m);
    for (std::size_t i = 0; i < m; ++i) {
        const double z = (y[i + 1] - mu);
        const double q0 = kInvSqrt2Pi / std::sqrt(t0) * std::exp(-0.5 * z * z / t0);
        psi[i] = q0 * std::exp(-B[i + 1]);
    }
    return psi;
}

/// y(x) = integral_0^x dx' / sigma(x') by composite Simpson.
double lamperti_y(const EffectiveVol& vol, double x) {
    constexpr int n = 4000;
    const double dx = x / n;
    double acc = 1.0 / vol.sigma(0.0) + 1.0 / vol.sigma(x);
    for (int i = 1; i < n; ++i) acc += (i % 2 ? 4.0 : 2.0) / vol.sigma(i * dx);
    return acc * dx / 3.0;
}

}  // namespace

std::vector<Density> solve_densities(const SchrodingerProblem& problem, std::vector<double> times) {
    if (times.empty()) return {};
    std::sort(times.begin(), times.end());
    if (!(times.front() > 0.0)) throw std::invalid_argument("solve_densities: times must be > 0");
    if (times.back() > problem.t_max() * (1.0 + 1e-12)) {
        throw std::invalid_argument("solve_densities: time beyond the problem's t_max");
    }
    const double h = problem.h();
    const double t0 = std::min(16.0 * h * h, 0.25 * times.front());
    auto psi = initial_psi(problem, t0);
    double t = t0;
    const double rate = problem.grid().n_time / (times.back() - t0);
    return propagate(problem, psi, t, times, rate, problem.grid().rannacher_steps);
}

std::vector<Density> solve_densities_piecewise(const std::vector<EffectiveVolParams>& params,
                                               const std::vector<double>& ends,
                                               std::vector<double> times, const GridSpec& grid) {
    if (params.empty() || params.size() != ends.size()) {
        throw std::invalid_argument("solve_densities_piecewise: need one segment end per well");
    }
    for (std::size_t j = 0; j < ends.size(); ++j) {
        if (!(ends[j] > (j ? ends[j - 1] : 0.0))) {
            throw std::invalid_argument("solve_densities_piecewise: ends must increase from 0");
        }
    }
    if (times.empty()) return {};
    std::sort(times.begin(), times.end());
    if (!(times.front() > 0.0) || times.back() > ends.back() * (1.0 + 1e-12)) {
        throw std::invalid_argument("solve_densities_piecewise: times must lie in (0, ends]");
    }
    const double horizon = times.back();
    const double rate = grid.n_time / horizon;

    std::vector<Density> out;
    std::size_t next_time = 0;
    std::vector<double> psi;
    double t = 0.0;
    std::unique_ptr<SchrodingerProblem> prev;
    Density carry;  // density at the end of the previous segment

    for (std::size_t j = 0; j < params.size() && next_time < times.size(); ++j) {
        const double seg_end = std::min(ends[j], horizon);
        std::unique_ptr<SchrodingerProblem> problem;
        if (j == 0) {
            problem = std::make_unique<SchrodingerProblem>(params[0], seg_end, grid);
            const double t0 = std::min(16.0 * problem->h() * problem->h(), 0.25 * times.front());
            psi = initial_psi(*problem, t0);
            t = t0;
        } else {
            // Support of the incoming density in x, mapped to this segment's Lamperti coordinate.
            const double q_max = *std::max_element(carry.q.begin(), carry.q.end());
            std::size_t lo = 0, hi = carry.q.size() - 1;
            while (lo < hi && carry.q[lo] < 1e-12 * q_max) ++lo;
            while (hi > lo && carry.q[hi] < 1e-12 * q_max) --hi;
            const EffectiveVol vol(params[j]);
            const double pad = grid.width_sd * std::sqrt(seg_end - t);
            const double y_lo = std::min(lamperti_y(vol, carry.x[lo]) - pad, -pad);
            const double y_hi = std::max(lamperti_y(vol, carry.x[hi]) + pad, pad);
            problem = std::make_unique<SchrodingerProblem>(params[j], seg_end, y_lo, y_hi, grid);

            // p(x) = q_old / sigma_old is continuous across the breakpoint; q_new = p sigma_new.
            const auto& xo = carry.x;
            const auto& so = prev->sigma();
            const auto& xn = problem->x();
            const auto& sn = problem->sigma();
            const auto& Bn = problem->gauge();
            psi.assign(xn.size() - 2, 0.0);
            for (std::size_t i = 1; i + 1 < xn.size(); ++i) {
                const auto it = std::upper_bound(xo.begin(), xo.end(), xn[i]);
                if (it == xo.begin() || it == xo.end()) continue;
                const std::size_t k = static_cast<std::size_t>(it - xo.begin());
                const double w = (xn[i] - xo[k - 1]) / (xo[k] - xo[k - 1]);
                const double p = (1.0 - w) * carry.q[k - 1] / so[k - 1] + w * carry.q[k] / so[k];
                psi[i - 1] = p * sn[i] * std::exp(-Bn[i]);
            }
        }

        std::vector<double> targets;
        while (next_time < times.size() && times[next_time] <= seg_end * (1.0 + 1e-12)) {
            targets.push_back(times[next_time++]);
        }
        const bool need_carry = next_time < times.size();
        if (need_carry && (targets.empty() || targets.back() < seg_end)) targets.push_back(seg_end);
        auto dens = propagate(*problem, psi, t, targets, rate, grid.rannacher_steps);
        for (auto& d : dens) {
            if (std::binary_search(times.begin(), times.end(), d.t)) out.push_back(d);
        }
        if (need_carry) carry = std::move(dens.back());
        prev = std::move(problem);
    }
    return out;
}

Density solve_density(const SchrodingerProblem& problem, double t) {
    return solve_densities(problem, {t}).front();
}

}  // namespace qp
