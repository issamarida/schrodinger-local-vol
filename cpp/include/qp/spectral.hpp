#pragma once

#include <vector>

#include "qp/schrodinger.hpp"

namespace qp {

/// Lowest eigenpairs of the discrete Hamiltonian H = -1/2 d^2/dy^2 + V (Dirichlet boundaries).
struct Spectrum {
    std::vector<double> energies;                   ///< ascending
    std::vector<std::vector<double>> states;        ///< on the full y-grid, unit L2 norm
    std::vector<double> y;
    double h = 0.0;
};

/// Computes the `count` lowest bound states of a tridiagonal Hamiltonian by Sturm-sequence
/// bisection (eigenvalues) and inverse iteration (eigenvectors). `V` is sampled on a uniform grid
/// of spacing `h`; the end nodes carry the Dirichlet condition.
Spectrum lowest_states(const std::vector<double>& y, const std::vector<double>& V, double h,
                       int count);

/// Spectrum of a calibrated smile potential.
Spectrum lowest_states(const SchrodingerProblem& problem, int count);

/// Density at time t from the truncated eigen-expansion
///     q(y, t) = e^{B(y)} sum_n e^{-E_n t} phi_n(y) phi_n(0),
/// i.e. the Euclidean propagator of the well started from a Dirac delta at y = 0.
/// For t much larger than 1/(E_1 - E_0) a handful of states reproduces the PDE solution.
Density spectral_density(const SchrodingerProblem& problem, const Spectrum& spectrum, double t);

}  // namespace qp
