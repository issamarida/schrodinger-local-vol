// Independent validation of the Schrödinger pricer against a conventional backward PDE.
//
// The quantum solver works forward in time on the Lamperti/gauge-transformed wavefunction.
// Here we price the same contracts the textbook way: the backward Kolmogorov equation for
// u(tau, x) = C / (df F) under dX = -sigma^2/2 dt + sigma dW,
//     u_tau = 1/2 sigma(x)^2 (u_xx - u_x),   u(0, x) = (e^x - K/F)^+,
// solved with Crank-Nicolson on a uniform x-grid. The two share no discretisation code.

#include <doctest/doctest.h>

#include <cmath>
#include <initializer_list>
#include <vector>

#include "qp/effective_vol.hpp"
#include "qp/pricer.hpp"
#include "qp/tridiagonal.hpp"

namespace {

double backward_pde_call(const qp::EffectiveVolParams& params, double F, double K, double T, double df) {
    const qp::EffectiveVol vol(params);
    const int n = 12001, steps = 2400;
    const double x_lo = -2.5, x_hi = 1.5;
    const double h = (x_hi - x_lo) / (n - 1);
    const double dt = T / steps;
    const double k = K / F;

    std::vector<double> x(n), u(n);
    for (int i = 0; i < n; ++i) {
        x[i] = x_lo + h * i;
        u[i] = std::max(std::exp(x[i]) - k, 0.0);
    }
    // Interior operator L u = a (u_xx - u_x), a = sigma^2 / 2.
    const int m = n - 2;
    std::vector<double> lo(m), di(m), up(m), blo(m), bdi(m), bup(m), ilo(m), idi(m), iup(m);
    for (int i = 0; i < m; ++i) {
        const double a = 0.5 * vol.variance(x[i + 1]);
        const double l = a * (1.0 / (h * h) + 0.5 / h);
        const double d = -2.0 * a / (h * h);
        const double r = a * (1.0 / (h * h) - 0.5 / h);
        lo[i] = -0.5 * dt * l; di[i] = 1.0 - 0.5 * dt * d; up[i] = -0.5 * dt * r;
        blo[i] = 0.5 * dt * l; bdi[i] = 1.0 + 0.5 * dt * d; bup[i] = 0.5 * dt * r;
        ilo[i] = -0.5 * dt * l; idi[i] = 1.0 - 0.5 * dt * d; iup[i] = -0.5 * dt * r;  // half-step Euler
    }
    const qp::TridiagonalLU cn(lo, di, up);
    const qp::TridiagonalLU euler(ilo, idi, iup);
    const double u_hi = std::exp(x_hi) - k;  // deep ITM boundary (martingale: e^x - k)

    std::vector<double> rhs(m), interior(m);
    for (int s = 0; s < steps; ++s) {
        for (int i = 0; i < m; ++i) interior[i] = u[i + 1];
        if (s < 2) {  // Rannacher start-up
            for (int half = 0; half < 2; ++half) {
                interior[m - 1] -= iup[m - 1] * u_hi;
                euler.solve(interior);
            }
        } else {
            qp::tridiagonal_multiply(blo, bdi, bup, interior, rhs);
            rhs[m - 1] += bup[m - 1] * u_hi - up[m - 1] * u_hi;
            cn.solve(rhs);
            interior.swap(rhs);
        }
        for (int i = 0; i < m; ++i) u[i + 1] = interior[i];
        u[0] = 0.0;
        u[n - 1] = u_hi;
    }
    // Interpolate at x = 0.
    const int j = static_cast<int>(std::floor((0.0 - x_lo) / h));
    const double w = (0.0 - x[j]) / h;
    return df * F * ((1.0 - w) * u[j] + w * u[j + 1]);
}

}  // namespace

TEST_CASE("Schrödinger pricer agrees with an independent backward local-vol PDE") {
    const double F = 3600.0, df = 0.997;
    // A short-dated SPX-like smile: steep put skew, upturn in the call wing.
    const qp::EffectiveVolParams params{0.012, 0.45, -0.8, 0.02, 0.05};
    for (double T : {5.0 / 365, 21.0 / 365}) {
        for (double K : {3000.0, 3300.0, 3500.0, 3600.0, 3700.0, 3850.0}) {
            const double ref = backward_pde_call(params, F, K, T, df);
            const double qm = qp::schrodinger_price(params, F, T, df, {K}, {qp::OptionType::Call},
                                                    {2000, 400, 9.0, 2})[0];
            MESSAGE("T=" << T * 365 << "d K=" << K << " backward=" << ref << " schrodinger=" << qm);
            CHECK(std::abs(qm - ref) < 5e-3);  // index points
        }
    }
}
