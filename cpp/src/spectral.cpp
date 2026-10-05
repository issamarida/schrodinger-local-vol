#include "qp/spectral.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "qp/tridiagonal.hpp"

namespace qp {

namespace {

/// Number of eigenvalues of the symmetric tridiagonal (d, e) strictly below lambda.
int sturm_count(const std::vector<double>& d, double e, double lambda) {
    int count = 0;
    double q = d[0] - lambda;
    if (q < 0.0) ++count;
    for (std::size_t i = 1; i < d.size(); ++i) {
        const double prev = (q == 0.0) ? 1e-300 : q;
        q = d[i] - lambda - e * e / prev;
        if (q < 0.0) ++count;
    }
    return count;
}

}  // namespace

Spectrum lowest_states(const std::vector<double>& y, const std::vector<double>& V, double h,
                       int count) {
    const std::size_t n = V.size();
    if (n < 3 || y.size() != n) throw std::invalid_argument("lowest_states: bad grid");
    const std::size_t m = n - 2;
    if (count < 1 || static_cast<std::size_t>(count) > m) {
        throw std::invalid_argument("lowest_states: count out of range");
    }

    const double kinetic = 0.5 / (h * h);
    const double e = -kinetic;
    std::vector<double> d(m);
    for (std::size_t i = 0; i < m; ++i) d[i] = 2.0 * kinetic + V[i + 1];

    // Gershgorin bounds.
    double lo = d[0] - std::abs(e), hi = d[0] + std::abs(e);
    for (double di : d) {
        lo = std::min(lo, di - 2.0 * std::abs(e));
        hi = std::max(hi, di + 2.0 * std::abs(e));
    }

    Spectrum spec;
    spec.y = y;
    spec.h = h;
    std::vector<double> lower(m), diag(m), upper(m);

    for (int k = 0; k < count; ++k) {
        // Bisection for the k-th eigenvalue (0-based): smallest lambda with count(lambda) > k.
        double a = lo, b = hi;
        for (int it = 0; it < 200 && (b - a) > 1e-13 * std::max(1.0, std::abs(b)); ++it) {
            const double mid = 0.5 * (a + b);
            if (sturm_count(d, e, mid) > k) b = mid; else a = mid;
        }
        const double E = 0.5 * (a + b);
        spec.energies.push_back(E);

        // Inverse iteration with a slightly perturbed shift to keep (H - shift) non-singular.
        const double shift = E - 1e-10 * std::max(1.0, std::abs(E));
        for (std::size_t i = 0; i < m; ++i) {
            lower[i] = upper[i] = e;
            diag[i] = d[i] - shift;
        }
        const TridiagonalLU lu(lower, diag, upper);
        std::vector<double> v(m);
        for (std::size_t i = 0; i < m; ++i) v[i] = 1.0 + 0.01 * std::sin(0.37 * static_cast<double>(i) + k);
        for (int it = 0; it < 3; ++it) {
            lu.solve(v);
            double norm = 0.0;
            for (double vi : v) norm += vi * vi;
            norm = std::sqrt(norm * h);
            for (double& vi : v) vi /= norm;
        }
        // Orthogonalise against previously found states (guards near-degenerate pairs).
        for (const auto& prev : spec.states) {
            double dot = 0.0;
            for (std::size_t i = 0; i < m; ++i) dot += prev[i + 1] * v[i] * h;
            for (std::size_t i = 0; i < m; ++i) v[i] -= dot * prev[i + 1];
        }
        double norm = 0.0;
        for (double vi : v) norm += vi * vi;
        norm = std::sqrt(norm * h);
        // Sign convention: positive where the state peaks.
        const auto peak = std::max_element(v.begin(), v.end(),
                                           [](double p, double q) { return std::abs(p) < std::abs(q); });
        const double sign = (*peak < 0.0) ? -1.0 : 1.0;

        std::vector<double> state(n, 0.0);
        for (std::size_t i = 0; i < m; ++i) state[i + 1] = sign * v[i] / norm;
        spec.states.push_back(std::move(state));
    }
    return spec;
}

Spectrum lowest_states(const SchrodingerProblem& problem, int count) {
    return lowest_states(problem.y(), problem.potential(), problem.h(), count);
}

Density spectral_density(const SchrodingerProblem& problem, const Spectrum& spectrum, double t) {
    const auto& y = problem.y();
    const auto& B = problem.gauge();
    const std::size_t n = y.size();

    // phi_n(0) by linear interpolation (y = 0 is generally not a node).
    const auto j = static_cast<std::size_t>(std::lower_bound(y.begin(), y.end(), 0.0) - y.begin());
    if (j == 0 || j >= n) throw std::runtime_error("spectral_density: y = 0 outside the grid");
    const double w = (0.0 - y[j - 1]) / (y[j] - y[j - 1]);

    Density d;
    d.t = t;
    d.h = problem.h();
    d.y = y;
    d.x = problem.x();
    d.q.assign(n, 0.0);
    for (std::size_t k = 0; k < spectrum.energies.size(); ++k) {
        const auto& phi = spectrum.states[k];
        const double phi0 = (1.0 - w) * phi[j - 1] + w * phi[j];
        const double amp = std::exp(-spectrum.energies[k] * t) * phi0;
        for (std::size_t i = 0; i < n; ++i) d.q[i] += amp * phi[i];
    }
    for (std::size_t i = 0; i < n; ++i) d.q[i] = std::max(0.0, std::exp(B[i]) * d.q[i]);
    return d;
}

}  // namespace qp
