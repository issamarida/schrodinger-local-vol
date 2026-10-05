#include "qp/black76.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace qp {

namespace {
constexpr double kInvSqrt2 = 0.70710678118654752440;
constexpr double kInvSqrt2Pi = 0.39894228040143267794;
constexpr double kSqrt2Pi = 2.50662827463100050242;
}  // namespace

double norm_cdf(double x) { return 0.5 * std::erfc(-x * kInvSqrt2); }

double norm_pdf(double x) { return kInvSqrt2Pi * std::exp(-0.5 * x * x); }

double black76_price(OptionType type, double F, double K, double T, double sigma, double df) {
    const double intrinsic_call = std::max(F - K, 0.0);
    const double intrinsic_put = std::max(K - F, 0.0);
    if (T <= 0.0 || sigma <= 0.0) {
        return df * (type == OptionType::Call ? intrinsic_call : intrinsic_put);
    }
    const double s = sigma * std::sqrt(T);
    const double d1 = (std::log(F / K) + 0.5 * s * s) / s;
    const double d2 = d1 - s;
    if (type == OptionType::Call) {
        return df * (F * norm_cdf(d1) - K * norm_cdf(d2));
    }
    return df * (K * norm_cdf(-d2) - F * norm_cdf(-d1));
}

double black76_vega(double F, double K, double T, double sigma, double df) {
    if (T <= 0.0 || sigma <= 0.0) return 0.0;
    const double sqrtT = std::sqrt(T);
    const double s = sigma * sqrtT;
    const double d1 = (std::log(F / K) + 0.5 * s * s) / s;
    return df * F * norm_pdf(d1) * sqrtT;
}

double black76_implied_vol(OptionType type, double F, double K, double T, double price, double df,
                           double tol, int max_iter) {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    if (T <= 0.0 || F <= 0.0 || K <= 0.0 || df <= 0.0) return nan;

    // Work with the undiscounted out-of-the-money price: it is the best conditioned.
    double undiscounted = price / df;
    if (type == OptionType::Call && K < F) undiscounted -= (F - K);  // call -> put via parity
    if (type == OptionType::Put && K > F) undiscounted += (F - K);   // put -> call via parity
    const OptionType otm = (K >= F) ? OptionType::Call : OptionType::Put;
    const double upper_bound = (otm == OptionType::Call) ? F : K;
    if (!(undiscounted > 0.0) || undiscounted >= upper_bound) return nan;

    // Bracket in total volatility s = sigma * sqrt(T).
    double lo = 1e-8, hi = 1.0;
    auto f = [&](double s) { return black76_price(otm, F, K, 1.0, s, 1.0) - undiscounted; };
    while (f(hi) < 0.0) {
        hi *= 2.0;
        if (hi > 50.0) return nan;
    }

    // Initial guess from the Brenner-Subrahmanyam ATM approximation, clamped to the bracket.
    double s = std::clamp(kSqrt2Pi * undiscounted / F, lo, hi);
    for (int i = 0; i < max_iter; ++i) {
        const double diff = f(s);
        if (std::abs(diff) < tol * std::max(1.0, undiscounted)) break;
        if (diff > 0.0) hi = s; else lo = s;
        const double vega = black76_vega(F, K, 1.0, s, 1.0);
        double next = (vega > 1e-300) ? s - diff / vega : 0.5 * (lo + hi);
        if (!(next > lo && next < hi)) next = 0.5 * (lo + hi);  // keep Newton inside the bracket
        if (std::abs(next - s) < 1e-15) { s = next; break; }
        s = next;
    }
    return s / std::sqrt(T);
}

}  // namespace qp
