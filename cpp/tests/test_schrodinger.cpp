#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <vector>

#include "qp/schrodinger.hpp"

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
