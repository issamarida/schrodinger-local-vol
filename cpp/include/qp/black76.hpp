#pragma once

namespace qp {

enum class OptionType { Call, Put };

/// Standard normal cumulative distribution function.
double norm_cdf(double x);

/// Standard normal probability density function.
double norm_pdf(double x);

/// Black-76 price of a European option on a forward.
///
/// @param F      forward price for the option's expiry
/// @param K      strike
/// @param T      time to expiry in years
/// @param sigma  annualised Black volatility
/// @param df     discount factor to expiry, e^{-rT}
double black76_price(OptionType type, double F, double K, double T, double sigma, double df);

/// dPrice/dSigma of a Black-76 option (identical for calls and puts).
double black76_vega(double F, double K, double T, double sigma, double df);

/// Black-76 implied volatility.
///
/// Safeguarded Newton iteration on total volatility with a bisection fallback.
/// Returns NaN when the price lies outside the no-arbitrage bounds.
double black76_implied_vol(OptionType type, double F, double K, double T, double price, double df,
                           double tol = 1e-10, int max_iter = 100);

}  // namespace qp
