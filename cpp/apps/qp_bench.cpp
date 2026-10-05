// qp_bench: wall-clock cost of one slice (potential build + propagation + 100 strikes).

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

#include "qp/pricer.hpp"

int main() {
    const qp::EffectiveVolParams params{0.012, 0.45, -0.8, 0.02, 0.05};
    const double F = 3600.0, df = 0.999;
    std::vector<double> strikes;
    std::vector<qp::OptionType> types;
    for (int k = 0; k < 100; ++k) {
        const double K = 3000.0 + 8.0 * k;
        strikes.push_back(K);
        types.push_back(K >= F ? qp::OptionType::Call : qp::OptionType::Put);
    }

    std::printf("%8s %8s %8s %14s %14s\n", "n_space", "n_time", "days", "median [us]", "slices/s");
    volatile double sink = 0.0;
    for (int n_space : {300, 600, 1200, 2400}) {
        for (double days : {7.0, 30.0}) {
            const qp::GridSpec grid{n_space, n_space / 6, 8.0, 2};
            std::vector<double> times;
            for (int rep = 0; rep < 41; ++rep) {
                const auto t0 = std::chrono::steady_clock::now();
                const auto prices = qp::schrodinger_price(params, F, days / 365.0, df, strikes, types, grid);
                const auto t1 = std::chrono::steady_clock::now();
                sink = sink + prices[0];
                times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            std::nth_element(times.begin(), times.begin() + 20, times.end());
            const double med = times[20];
            std::printf("%8d %8d %8.0f %14.1f %14.0f\n", n_space, grid.n_time, days, med, 1e6 / med);
        }
    }
    return 0;
}
