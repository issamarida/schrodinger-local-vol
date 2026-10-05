#pragma once

namespace qp {

/// Effective local volatility of a 1D projection of a stochastic-volatility model.
///
/// By Gyöngy's Markovian projection theorem, the marginal distributions of a stochastic-volatility
/// model dX = -v/2 dt + sqrt(v) dW are reproduced by a 1D diffusion whose local variance is
/// sigma^2(x) = E[v | X = x]. We parameterise that conditional expectation with the SVI shape
///
///     sigma^2(x) = a + b * ( rho * (x - m) + sqrt((x - m)^2 + s^2) ),
///
/// where x = ln(F_t / F_0) is log forward moneyness. The shape is chosen deliberately: its wings
/// grow linearly in |x|, which (after the Lamperti transform) turns the quantum potential into an
/// asymptotically harmonic well, so the Hamiltonian has a purely discrete spectrum.
///
/// Positivity requires b >= 0, |rho| < 1, s > 0 and a + b * s * sqrt(1 - rho^2) > 0.
struct EffectiveVolParams {
    double a = 0.04;    ///< variance level
    double b = 0.0;     ///< wing slope
    double rho = 0.0;   ///< skew, in (-1, 1)
    double m = 0.0;     ///< horizontal shift of the variance minimum
    double s = 0.1;     ///< ATM curvature (smoothing) scale, > 0

    /// A flat volatility: reduces the model exactly to Black-Scholes.
    static EffectiveVolParams constant(double sigma) { return {sigma * sigma, 0.0, 0.0, 0.0, 0.1}; }

    /// Minimum of sigma^2(x) over the real line.
    double min_variance() const;

    /// Throws std::invalid_argument when the parameters violate the positivity constraints.
    void validate() const;
};

class EffectiveVol {
public:
    explicit EffectiveVol(const EffectiveVolParams& p);

    double variance(double x) const;
    double sigma(double x) const;

    /// sigma(x) together with its first and second x-derivatives.
    struct Jet {
        double sigma, d1, d2;
    };
    Jet jet(double x) const;

    const EffectiveVolParams& params() const { return p_; }

private:
    EffectiveVolParams p_;
};

}  // namespace qp
