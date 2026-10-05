#include <doctest/doctest.h>

#include <cmath>
#include <initializer_list>

#include "qp/effective_vol.hpp"

TEST_CASE("Effective-vol derivatives agree with finite differences") {
    const qp::EffectiveVol vol({0.02, 0.3, -0.7, 0.05, 0.08});
    const double eps = 1e-5;
    for (double x : {-0.6, -0.2, -0.03, 0.0, 0.05, 0.3}) {
        const auto j = vol.jet(x);
        CHECK(j.sigma == doctest::Approx(vol.sigma(x)).epsilon(1e-14));
        const double d1 = (vol.sigma(x + eps) - vol.sigma(x - eps)) / (2 * eps);
        const double d2 = (vol.sigma(x + eps) - 2 * vol.sigma(x) + vol.sigma(x - eps)) / (eps * eps);
        CHECK(j.d1 == doctest::Approx(d1).epsilon(1e-7));
        CHECK(j.d2 == doctest::Approx(d2).epsilon(1e-4));
    }
}

TEST_CASE("Flat parameters give a constant volatility") {
    const qp::EffectiveVol vol(qp::EffectiveVolParams::constant(0.25));
    for (double x : {-1.0, 0.0, 0.7}) {
        CHECK(vol.sigma(x) == doctest::Approx(0.25));
        CHECK(vol.jet(x).d1 == 0.0);
        CHECK(vol.jet(x).d2 == 0.0);
    }
}

TEST_CASE("Positivity constraints are enforced") {
    CHECK_THROWS(qp::EffectiveVol({-0.1, 0.1, 0.0, 0.0, 0.1}));  // negative minimum variance
    CHECK_THROWS(qp::EffectiveVol({0.04, -0.1, 0.0, 0.0, 0.1}));  // negative wing slope
    CHECK_THROWS(qp::EffectiveVol({0.04, 0.1, 1.0, 0.0, 0.1}));   // |rho| = 1
    CHECK_THROWS(qp::EffectiveVol({0.04, 0.1, 0.0, 0.0, 0.0}));   // s = 0
    CHECK_NOTHROW(qp::EffectiveVol({-0.01, 0.5, -0.5, 0.0, 0.1}));  // a < 0 but min variance > 0
}
