#include <doctest/doctest.h>

#include <cmath>
#include <initializer_list>
#include <vector>

#include "qp/tridiagonal.hpp"

TEST_CASE("Tridiagonal LU solves a diagonally dominant system") {
    const std::size_t n = 50;
    std::vector<double> lo(n), di(n), up(n), x(n);
    for (std::size_t i = 0; i < n; ++i) {
        lo[i] = -1.0 - 0.01 * static_cast<double>(i);
        up[i] = -0.5 + 0.02 * std::sin(static_cast<double>(i));
        di[i] = 4.0 + 0.1 * std::cos(static_cast<double>(i));
        x[i] = std::sin(0.3 * static_cast<double>(i)) + 0.1 * static_cast<double>(i);
    }
    std::vector<double> rhs;
    qp::tridiagonal_multiply(lo, di, up, x, rhs);

    const qp::TridiagonalLU lu(lo, di, up);
    lu.solve(rhs);
    for (std::size_t i = 0; i < n; ++i) CHECK(rhs[i] == doctest::Approx(x[i]).epsilon(1e-12));

    // The factorisation is reusable for a second right-hand side.
    std::vector<double> rhs2;
    qp::tridiagonal_multiply(lo, di, up, std::vector<double>(n, 1.0), rhs2);
    lu.solve(rhs2);
    for (double v : rhs2) CHECK(v == doctest::Approx(1.0).epsilon(1e-12));
}

TEST_CASE("Tridiagonal LU rejects mismatched bands") {
    CHECK_THROWS(qp::TridiagonalLU({0.0}, {1.0, 2.0}, {0.0, 0.0}));
}
