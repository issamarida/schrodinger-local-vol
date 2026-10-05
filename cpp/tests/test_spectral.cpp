#include <doctest/doctest.h>

#include <cmath>
#include <initializer_list>
#include <vector>

#include "qp/pricer.hpp"
#include "qp/spectral.hpp"

TEST_CASE("Harmonic oscillator levels are E_n = omega (n + 1/2)") {
    const double omega = 1.7;
    const int n = 2001;
    const double L = 8.0;
    const double h = 2 * L / (n - 1);
    std::vector<double> y(n), V(n);
    for (int i = 0; i < n; ++i) {
        y[i] = -L + h * i;
        V[i] = 0.5 * omega * omega * y[i] * y[i];
    }
    const auto spec = qp::lowest_states(y, V, h, 6);
    for (int k = 0; k < 6; ++k) {
        CHECK(spec.energies[k] == doctest::Approx(omega * (k + 0.5)).epsilon(1e-4));
    }
    // Ground state is the Gaussian (omega/pi)^{1/4} e^{-omega y^2 / 2}.
    const double peak = std::pow(omega / 3.14159265358979323846, 0.25);
    CHECK(spec.states[0][n / 2] == doctest::Approx(peak).epsilon(1e-4));
    // Orthonormality.
    double dot01 = 0.0, dot11 = 0.0;
    for (int i = 0; i < n; ++i) {
        dot01 += spec.states[0][i] * spec.states[1][i] * h;
        dot11 += spec.states[1][i] * spec.states[1][i] * h;
    }
    CHECK(std::abs(dot01) < 1e-8);
    CHECK(dot11 == doctest::Approx(1.0).epsilon(1e-10));
}

TEST_CASE("Eigen-expansion of the smile well reproduces the Crank-Nicolson density") {
    // Wide grid and a long horizon so the expansion converges with few states.
    const qp::EffectiveVolParams params{0.02, 0.6, -0.5, 0.0, 0.2};
    const double T = 1.0;
    const qp::SchrodingerProblem problem(params, T, {1500, 800, 8.0, 2});
    const auto cn = qp::solve_density(problem, T);
    const auto spec = qp::lowest_states(problem, 60);
    const auto sd = qp::spectral_density(problem, spec, T);

    const double F = 100.0, df = 1.0;
    const std::vector<double> strikes{70, 85, 100, 115, 140};
    const std::vector<qp::OptionType> types{qp::OptionType::Put, qp::OptionType::Put,
                                            qp::OptionType::Call, qp::OptionType::Call,
                                            qp::OptionType::Call};
    const auto p_cn = qp::price_from_density(cn, F, df, strikes, types);
    const auto p_sd = qp::price_from_density(sd, F, df, strikes, types);
    for (std::size_t k = 0; k < strikes.size(); ++k) {
        MESSAGE("K=" << strikes[k] << " CN=" << p_cn[k] << " spectral=" << p_sd[k]);
        CHECK(p_sd[k] == doctest::Approx(p_cn[k]).epsilon(5e-3));
    }
    // Discrete, increasing spectrum.
    for (std::size_t k = 1; k < spec.energies.size(); ++k) CHECK(spec.energies[k] > spec.energies[k - 1]);
}
