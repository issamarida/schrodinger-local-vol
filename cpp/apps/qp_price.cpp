// qp_price: price one expiry slice with the Schrödinger solver from the command line.
//
//   qp_price --forward 3577 --days 9 --rate 0.03 \
//            --a 0.012 --b 0.45 --rho -0.8 --m 0.02 --s 0.05 \
//            --strikes 3000,3300,3500,3577,3700,3850 [--states 5]

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "qp/black76.hpp"
#include "qp/pricer.hpp"
#include "qp/spectral.hpp"

namespace {

void usage() {
    std::cerr << "usage: qp_price --forward F --days D [--rate r] [--a a --b b --rho rho --m m --s s]\n"
                 "                --strikes K1,K2,... [--states N] [--nx N] [--nt N]\n";
}

std::vector<double> parse_list(const std::string& s) {
    std::vector<double> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) out.push_back(std::stod(item));
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    double F = 0, days = 0, rate = 0;
    qp::EffectiveVolParams p{0.012, 0.45, -0.8, 0.02, 0.05};
    qp::GridSpec grid;
    std::vector<double> strikes;
    int states = 0;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (i + 1 >= argc) { usage(); return 2; }
        const std::string val = argv[++i];
        if (arg == "--forward") F = std::stod(val);
        else if (arg == "--days") days = std::stod(val);
        else if (arg == "--rate") rate = std::stod(val);
        else if (arg == "--a") p.a = std::stod(val);
        else if (arg == "--b") p.b = std::stod(val);
        else if (arg == "--rho") p.rho = std::stod(val);
        else if (arg == "--m") p.m = std::stod(val);
        else if (arg == "--s") p.s = std::stod(val);
        else if (arg == "--strikes") strikes = parse_list(val);
        else if (arg == "--states") states = std::stoi(val);
        else if (arg == "--nx") grid.n_space = std::stoi(val);
        else if (arg == "--nt") grid.n_time = std::stoi(val);
        else { usage(); return 2; }
    }
    if (F <= 0 || days <= 0 || strikes.empty()) { usage(); return 2; }

    try {
        const double T = days / 365.0;
        const double df = std::exp(-rate * T);
        std::vector<qp::OptionType> types;
        for (double K : strikes) types.push_back(K >= F ? qp::OptionType::Call : qp::OptionType::Put);

        const qp::SchrodingerProblem problem(p, T, grid);
        const auto density = qp::solve_density(problem, T);
        const auto prices = qp::price_from_density(density, F, df, strikes, types);

        std::printf("Schrodinger pricer  F=%.2f  T=%.4f (%.1f d)  df=%.6f\n", F, T, days, df);
        std::printf("effective vol: a=%.5g b=%.5g rho=%.4g m=%.4g s=%.4g   sigma(0)=%.4f\n", p.a, p.b,
                    p.rho, p.m, p.s, problem.vol().sigma(0.0));
        std::printf("grid: %d x %d   mass=%.8f   E[e^X]=%.8f\n\n", grid.n_space, grid.n_time,
                    density.mass(), density.forward_ratio());
        std::printf("%10s %5s %14s %10s\n", "strike", "type", "price", "impl.vol");
        for (std::size_t k = 0; k < strikes.size(); ++k) {
            const double iv = qp::black76_implied_vol(types[k], F, strikes[k], T, prices[k], df);
            std::printf("%10.2f %5s %14.6f %9.2f%%\n", strikes[k],
                        types[k] == qp::OptionType::Call ? "C" : "P", prices[k], 100.0 * iv);
        }

        if (states > 0) {
            // The pricing grid spans +-8 sd of the horizon. On it the lowest eigenstates of a
            // short-dated well are modes of the Dirichlet box, not of the well, so the spectrum
            // is computed on a domain 8x wider (same spacing) and checked against one 16x wider.
            auto wide_spectrum = [&](double factor) {
                qp::GridSpec g = grid;
                g.width_sd *= factor;
                g.n_space = static_cast<int>(grid.n_space * factor);
                return qp::lowest_states(qp::SchrodingerProblem(p, T, g), states);
            };
            const auto spec = wide_spectrum(8.0);
            const auto check = wide_spectrum(16.0);
            std::printf("\nlowest bound states of H = -1/2 d^2/dy^2 + V(y), on +-%.0f sd:\n",
                        8.0 * grid.width_sd);
            for (std::size_t k = 0; k < spec.energies.size(); ++k) {
                const double drift = std::abs(check.energies[k] - spec.energies[k]) /
                                     std::max(std::abs(check.energies[k]), 1e-12);
                std::printf("  E_%zu = %.6f%s\n", k, spec.energies[k],
                            drift > 0.01 ? "   (not converged: still moves with the domain)" : "");
            }
            if (spec.energies.size() > 1) {
                const double gap_t = (spec.energies[1] - spec.energies[0]) * T;
                std::printf("T (E_1 - E_0) = %.4g: the eigen-expansion %s at this maturity\n", gap_t,
                            gap_t > 1.0 ? "converges in a few states"
                                        : "needs many states; the PDE is what prices");
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
