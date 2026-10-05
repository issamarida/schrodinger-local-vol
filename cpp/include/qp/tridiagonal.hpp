#pragma once

#include <cstddef>
#include <vector>

namespace qp {

/// LU factorisation of a tridiagonal matrix, reusable across many right-hand sides.
///
/// The Schrödinger Hamiltonian is time-independent, so the Crank-Nicolson system matrix is
/// factorised once per solve and every time step costs a single O(N) forward/back sweep.
/// No pivoting: callers must supply a diagonally dominant matrix. For I + dt/2 H with
/// H = -1/2 d^2/dy^2 + V this holds whenever 1 + dt/2 V > 0 on every node.
class TridiagonalLU {
public:
    TridiagonalLU() = default;

    /// @param lower  sub-diagonal, lower[i] multiplies x[i-1] in row i (lower[0] ignored)
    /// @param diag   main diagonal
    /// @param upper  super-diagonal, upper[i] multiplies x[i+1] in row i (upper[n-1] ignored)
    TridiagonalLU(const std::vector<double>& lower, const std::vector<double>& diag,
                  const std::vector<double>& upper);

    /// Solves A x = rhs in place.
    void solve(std::vector<double>& rhs) const;

    std::size_t size() const { return inv_diag_.size(); }

private:
    std::vector<double> lower_;  // multipliers l_i
    std::vector<double> inv_diag_;  // reciprocal pivots 1/u_i (division-free sweeps)
    std::vector<double> upper_;  // unchanged super-diagonal
};

/// Computes y = A x for a tridiagonal A given by its three bands.
void tridiagonal_multiply(const std::vector<double>& lower, const std::vector<double>& diag,
                          const std::vector<double>& upper, const std::vector<double>& x,
                          std::vector<double>& y);

}  // namespace qp
