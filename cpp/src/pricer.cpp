#include "qp/pricer.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace qp {

std::vector<double> price_from_density(const Density& density, double F, double df,
                                       const std::vector<double>& strikes,
                                       const std::vector<OptionType>& types) {
    if (strikes.size() != types.size()) {
        throw std::invalid_argument("price_from_density: strikes and types differ in length");
    }
    const auto& q = density.q;
    const auto& x = density.x;
    const double h = density.h;
    const std::size_t n = q.size();

    // Probability weights c_i (trapezoid rule) and the martingale-corrected terminal forward S_i.
    std::vector<double> c(n), S(n);
    double m0 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        c[i] = ((i == 0 || i + 1 == n) ? 0.5 * h : h) * q[i];
        m0 += c[i];
    }
    if (!(m0 > 0.0)) throw std::runtime_error("price_from_density: density has no mass");
    double m1 = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        c[i] /= m0;
        m1 += c[i] * std::exp(x[i]);
    }
    for (std::size_t i = 0; i < n; ++i) S[i] = F * std::exp(x[i]) / m1;

    // Prefix sums: lower tail [0, j) and upper tail [j, n).
    std::vector<double> low_p(n + 1, 0.0), low_s(n + 1, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
        low_p[i + 1] = low_p[i] + c[i];
        low_s[i + 1] = low_s[i] + c[i] * S[i];
    }
    std::vector<double> up_p(n + 1, 0.0), up_s(n + 1, 0.0);
    for (std::size_t i = n; i-- > 0;) {
        up_p[i] = up_p[i + 1] + c[i];
        up_s[i] = up_s[i + 1] + c[i] * S[i];
    }

    std::vector<double> prices(strikes.size());
    for (std::size_t k = 0; k < strikes.size(); ++k) {
        const double K = strikes[k];
        // j = first node with S_j > K.
        const auto j = static_cast<std::size_t>(std::upper_bound(S.begin(), S.end(), K) - S.begin());
        // Fraction of the bracketing cell lying above the strike, linear in x.
        double theta_up = 1.0;
        if (j > 0 && j < n) {
            const double xs = std::log(K / F * m1);
            theta_up = std::clamp((x[j] - xs) / (x[j] - x[j - 1]), 0.0, 1.0);
        }
        const double cell = 0.5 * h / m0;  // trapezoid half-weight, in normalised units
        if (types[k] == OptionType::Call) {
            double v = up_s[j] - K * up_p[j];
            if (j > 0 && j < n) v -= (1.0 - theta_up) * cell * q[j] * (S[j] - K);
            prices[k] = df * std::max(v, 0.0);
        } else {
            double v = K * low_p[j] - low_s[j];
            if (j > 0 && j < n) v -= theta_up * cell * q[j - 1] * (K - S[j - 1]);
            prices[k] = df * std::max(v, 0.0);
        }
    }
    return prices;
}

std::vector<double> schrodinger_price(const EffectiveVolParams& params, double F, double T,
                                      double df, const std::vector<double>& strikes,
                                      const std::vector<OptionType>& types, const GridSpec& grid) {
    const SchrodingerProblem problem(params, T, grid);
    return price_from_density(solve_density(problem, T), F, df, strikes, types);
}

std::vector<std::vector<double>> schrodinger_price_surface(
    const EffectiveVolParams& params, const std::vector<double>& expiries,
    const std::vector<double>& forwards, const std::vector<double>& dfs,
    const std::vector<std::vector<double>>& strikes,
    const std::vector<std::vector<OptionType>>& types, const GridSpec& grid) {
    const std::size_t n = expiries.size();
    if (forwards.size() != n || dfs.size() != n || strikes.size() != n || types.size() != n) {
        throw std::invalid_argument("schrodinger_price_surface: inconsistent slice counts");
    }
    if (n == 0) return {};
    const double t_max = *std::max_element(expiries.begin(), expiries.end());
    const SchrodingerProblem problem(params, t_max, grid);
    const auto densities = solve_densities(problem, expiries);  // ascending in time

    std::vector<std::vector<double>> out(n);
    for (std::size_t e = 0; e < n; ++e) {
        const auto it = std::find_if(densities.begin(), densities.end(),
                                     [&](const Density& d) { return d.t == expiries[e]; });
        out[e] = price_from_density(*it, forwards[e], dfs[e], strikes[e], types[e]);
    }
    return out;
}

std::vector<std::vector<double>> schrodinger_price_piecewise(
    const std::vector<EffectiveVolParams>& params, const std::vector<double>& ends,
    const std::vector<double>& expiries, const std::vector<double>& forwards,
    const std::vector<double>& dfs, const std::vector<std::vector<double>>& strikes,
    const std::vector<std::vector<OptionType>>& types, const GridSpec& grid) {
    const std::size_t n = expiries.size();
    if (forwards.size() != n || dfs.size() != n || strikes.size() != n || types.size() != n) {
        throw std::invalid_argument("schrodinger_price_piecewise: inconsistent slice counts");
    }
    if (n == 0) return {};
    const auto densities = solve_densities_piecewise(params, ends, expiries, grid);
    std::vector<std::vector<double>> out(n);
    for (std::size_t e = 0; e < n; ++e) {
        const auto it = std::find_if(densities.begin(), densities.end(),
                                     [&](const Density& d) { return d.t == expiries[e]; });
        out[e] = price_from_density(*it, forwards[e], dfs[e], strikes[e], types[e]);
    }
    return out;
}

}  // namespace qp
