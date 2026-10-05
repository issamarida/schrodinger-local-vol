#include "qp/effective_vol.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

namespace qp {

double EffectiveVolParams::min_variance() const {
    return a + b * s * std::sqrt(std::max(0.0, 1.0 - rho * rho));
}

void EffectiveVolParams::validate() const {
    if (!(b >= 0.0)) throw std::invalid_argument("EffectiveVol: b must be >= 0");
    if (!(std::abs(rho) < 1.0)) throw std::invalid_argument("EffectiveVol: |rho| must be < 1");
    if (!(s > 0.0)) throw std::invalid_argument("EffectiveVol: s must be > 0");
    if (!(min_variance() > 0.0)) {
        throw std::invalid_argument("EffectiveVol: variance not positive everywhere (min = " +
                                    std::to_string(min_variance()) + ")");
    }
}

EffectiveVol::EffectiveVol(const EffectiveVolParams& p) : p_(p) { p_.validate(); }

double EffectiveVol::variance(double x) const {
    const double z = x - p_.m;
    return p_.a + p_.b * (p_.rho * z + std::sqrt(z * z + p_.s * p_.s));
}

double EffectiveVol::sigma(double x) const { return std::sqrt(variance(x)); }

EffectiveVol::Jet EffectiveVol::jet(double x) const {
    // w = sigma^2, sigma = sqrt(w):
    //   sigma'  = w' / (2 sigma)
    //   sigma'' = w'' / (2 sigma) - w'^2 / (4 sigma^3)
    const double z = x - p_.m;
    const double r = std::sqrt(z * z + p_.s * p_.s);
    const double w = p_.a + p_.b * (p_.rho * z + r);
    const double w1 = p_.b * (p_.rho + z / r);
    const double w2 = p_.b * p_.s * p_.s / (r * r * r);
    const double sig = std::sqrt(w);
    return {sig, w1 / (2.0 * sig), w2 / (2.0 * sig) - w1 * w1 / (4.0 * sig * w)};
}

}  // namespace qp
