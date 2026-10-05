#pragma once

#include <vector>

#include "qp/black76.hpp"
#include "qp/schrodinger.hpp"

namespace qp {

/// Prices European options by integrating payoffs against a Schrödinger-evolved density.
///
/// The discrete density is renormalised to unit mass and its first moment is matched to the
/// forward (a martingale correction of O(h^2), see `Density::forward_ratio`), so put-call parity
/// holds up to the O(h^2) quadrature error. Calls are integrated over the upper tail and puts
/// over the lower tail (no cancellation for far out-of-the-money contracts), with the payoff
/// kink resolved inside the cell that contains the strike.
/// Strikes do not need to be sorted; the cost is O(N + K log N).
std::vector<double> price_from_density(const Density& density, double F, double df,
                                       const std::vector<double>& strikes,
                                       const std::vector<OptionType>& types);

/// End-to-end pricing of one expiry slice: build the potential, propagate to T, integrate payoffs.
std::vector<double> schrodinger_price(const EffectiveVolParams& params, double F, double T,
                                      double df, const std::vector<double>& strikes,
                                      const std::vector<OptionType>& types,
                                      const GridSpec& grid = {});

/// Prices several expiries from a single forward solve (all slices share one effective vol).
/// `forwards`, `dfs` and `strikes`/`types` are indexed by expiry.
std::vector<std::vector<double>> schrodinger_price_surface(
    const EffectiveVolParams& params, const std::vector<double>& expiries,
    const std::vector<double>& forwards, const std::vector<double>& dfs,
    const std::vector<std::vector<double>>& strikes,
    const std::vector<std::vector<OptionType>>& types, const GridSpec& grid = {});

}  // namespace qp
