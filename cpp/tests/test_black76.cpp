#include <doctest/doctest.h>

#include <cmath>
#include <initializer_list>

#include "qp/black76.hpp"

using qp::OptionType;

TEST_CASE("Black-76 matches a hand-computed reference") {
    // F = 100, K = 100, T = 1, sigma = 0.2, df = 1: price = F (2 N(0.1) - 1) = 7.965567455405804
    CHECK(qp::black76_price(OptionType::Call, 100, 100, 1, 0.2, 1) ==
          doctest::Approx(7.965567455405804).epsilon(1e-13));
}

TEST_CASE("Black-76 satisfies put-call parity") {
    for (double K : {50.0, 90.0, 100.0, 125.0, 300.0}) {
        for (double T : {0.01, 0.25, 2.0}) {
            const double F = 103.0, df = 0.97, sigma = 0.35;
            const double c = qp::black76_price(OptionType::Call, F, K, T, sigma, df);
            const double p = qp::black76_price(OptionType::Put, F, K, T, sigma, df);
            CHECK(c - p == doctest::Approx(df * (F - K)).epsilon(1e-12));
        }
    }
}

TEST_CASE("Black-76 vega agrees with a central difference") {
    const double F = 3600, K = 3300, T = 9.0 / 365, s = 0.38, df = 0.999;
    const double eps = 1e-6;
    const double fd = (qp::black76_price(OptionType::Put, F, K, T, s + eps, df) -
                       qp::black76_price(OptionType::Put, F, K, T, s - eps, df)) / (2 * eps);
    CHECK(qp::black76_vega(F, K, T, s, df) == doctest::Approx(fd).epsilon(1e-7));
}

TEST_CASE("Implied volatility round-trips across moneyness and maturity") {
    const double F = 3577.0, df = 0.995;
    for (double T : {2.0 / 365, 9.0 / 365, 30.0 / 365, 1.0}) {
        for (double k : {-0.5, -0.2, -0.05, 0.0, 0.04, 0.15}) {
            for (double sigma : {0.12, 0.35, 0.95}) {
                const double K = F * std::exp(k);
                for (auto type : {OptionType::Call, OptionType::Put}) {
                    const double price = qp::black76_price(type, F, K, T, sigma, df);
                    // Time value; for deep ITM contracts it vanishes into the rounding of the
                    // intrinsic value and volatility is no longer identifiable from the price.
                    const double otm = qp::black76_price(K >= F ? OptionType::Call : OptionType::Put,
                                                         F, K, T, sigma, df);
                    if (otm < 1e-6) continue;
                    const double iv = qp::black76_implied_vol(type, F, K, T, price, df);
                    CHECK(iv == doctest::Approx(sigma).epsilon(1e-7));
                }
            }
        }
    }
}

TEST_CASE("Implied volatility rejects arbitrageable prices") {
    CHECK(std::isnan(qp::black76_implied_vol(OptionType::Call, 100, 90, 1, 9.0, 1.0)));  // < intrinsic
    CHECK(std::isnan(qp::black76_implied_vol(OptionType::Call, 100, 90, 1, 101.0, 1.0)));  // > F
}
