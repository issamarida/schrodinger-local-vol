#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <vector>

#include "qp/black76.hpp"
#include "qp/pricer.hpp"
#include "qp/schrodinger.hpp"

using qp::OptionType;

namespace {

double max_abs_error_vs_black(double sigma, double T, const qp::GridSpec& grid) {
    const double F = 3600.0, df = 0.99;
    std::vector<double> strikes;
    std::vector<OptionType> types;
    for (double z = -4.0; z <= 4.0; z += 0.25) {
        const double K = F * std::exp(z * sigma * std::sqrt(T));
        strikes.push_back(K);
        types.push_back(K >= F ? OptionType::Call : OptionType::Put);
    }
    const auto prices =
        qp::schrodinger_price(qp::EffectiveVolParams::constant(sigma), F, T, df, strikes, types, grid);
    double err = 0.0;
    for (std::size_t i = 0; i < strikes.size(); ++i) {
        const double ref = qp::black76_price(types[i], F, strikes[i], T, sigma, df);
        err = std::max(err, std::abs(prices[i] - ref));
    }
    return err;
}

}  // namespace

TEST_CASE("Constant volatility gives a flat potential well V = sigma^2 / 8") {
    const double sigma = 0.3;
    const qp::SchrodingerProblem p(qp::EffectiveVolParams::constant(sigma), 0.1);
    for (std::size_t i = 0; i < p.y().size(); i += 97) {
        CHECK(p.x()[i] == doctest::Approx(sigma * p.y()[i]).epsilon(1e-12));
        CHECK(p.potential()[i] == doctest::Approx(sigma * sigma / 8.0).epsilon(1e-12));
        CHECK(p.gauge()[i] == doctest::Approx(-0.5 * p.x()[i]).epsilon(1e-12));
    }
}

TEST_CASE("Lamperti map inverts y = integral of dx / sigma(x)") {
    const qp::EffectiveVolParams params{0.01, 0.4, -0.6, 0.02, 0.1};
    const qp::EffectiveVol vol(params);
    const qp::SchrodingerProblem p(params, 0.1);
    for (std::size_t i = 0; i < p.y().size(); i += 131) {
        // Independent check: trapezoid integral of 1/sigma from 0 to x(y).
        const double x = p.x()[i];
        const int n = 20000;
        double y = 0.0;
        for (int k = 0; k < n; ++k) {
            const double a = x * k / n, b = x * (k + 1) / n;
            y += 0.5 * (b - a) * (1.0 / vol.sigma(a) + 1.0 / vol.sigma(b));
        }
        CHECK(y == doctest::Approx(p.y()[i]).epsilon(1e-7));
    }
}

TEST_CASE("Probability is conserved and the forward is a martingale before correction") {
    const qp::EffectiveVolParams params{0.015, 0.35, -0.75, 0.03, 0.06};
    for (double T : {3.0 / 365, 20.0 / 365, 0.25}) {
        const qp::SchrodingerProblem p(params, T);
        const auto d = qp::solve_density(p, T);
        CHECK(d.mass() == doctest::Approx(1.0).epsilon(2e-5));
        CHECK(d.forward_ratio() == doctest::Approx(1.0).epsilon(2e-5));
    }
}

TEST_CASE("Flat potential reproduces Black-Scholes across maturities") {
    for (double T : {2.0 / 365, 9.0 / 365, 30.0 / 365, 0.5}) {
        for (double sigma : {0.15, 0.4}) {
            // Default grid: error below 5e-6 F (0.018 index points), under half the 0.05 SPX tick.
            const double err = max_abs_error_vs_black(sigma, T, {});
            MESSAGE("T=" << T * 365 << "d sigma=" << sigma << " max abs error=" << err);
            CHECK(err < 5e-6 * 3600.0);
        }
    }
}

TEST_CASE("Crank-Nicolson converges at second order") {
    const double sigma = 0.25, T = 30.0 / 365;
    const double e1 = max_abs_error_vs_black(sigma, T, {200, 50, 8.0, 2});
    const double e2 = max_abs_error_vs_black(sigma, T, {400, 100, 8.0, 2});
    const double e3 = max_abs_error_vs_black(sigma, T, {800, 200, 8.0, 2});
    MESSAGE("errors: " << e1 << " " << e2 << " " << e3);
    CHECK(e1 / e2 > 3.0);
    CHECK(e2 / e3 > 3.0);
}

TEST_CASE("One forward solve prices a whole surface") {
    const qp::EffectiveVolParams params{0.02, 0.3, -0.7, 0.0, 0.1};
    const std::vector<double> expiries{7.0 / 365, 14.0 / 365, 28.0 / 365};
    const std::vector<double> forwards{3600, 3601, 3603}, dfs{0.999, 0.998, 0.996};
    const std::vector<std::vector<double>> strikes{{3400, 3700}, {3300, 3800}, {3200, 3900}};
    const std::vector<std::vector<OptionType>> types(3, {OptionType::Put, OptionType::Call});
    const auto surface = qp::schrodinger_price_surface(params, expiries, forwards, dfs, strikes, types);
    for (std::size_t e = 0; e < expiries.size(); ++e) {
        const auto single = qp::schrodinger_price(params, forwards[e], expiries[e], dfs[e], strikes[e],
                                                  types[e], {2400, 400, 8.0, 2});
        for (std::size_t k = 0; k < 2; ++k) {
            CHECK(surface[e][k] == doctest::Approx(single[k]).epsilon(2e-3));
        }
    }
}
