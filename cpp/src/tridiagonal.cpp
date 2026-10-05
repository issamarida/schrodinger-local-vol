#include "qp/tridiagonal.hpp"

#include <cmath>
#include <stdexcept>

namespace qp {

TridiagonalLU::TridiagonalLU(const std::vector<double>& lower, const std::vector<double>& diag,
                             const std::vector<double>& upper)
    : lower_(diag.size()), inv_diag_(diag.size()), upper_(upper) {
    const std::size_t n = diag.size();
    if (n == 0 || lower.size() != n || upper.size() != n) {
        throw std::invalid_argument("TridiagonalLU: bands must be non-empty and equally sized");
    }
    double pivot = diag[0];
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) {
            lower_[i] = lower[i] * inv_diag_[i - 1];
            pivot = diag[i] - lower_[i] * upper[i - 1];
        }
        if (pivot == 0.0) throw std::runtime_error("TridiagonalLU: zero pivot");
        inv_diag_[i] = 1.0 / pivot;
    }
}

void TridiagonalLU::solve(std::vector<double>& rhs) const {
    const std::size_t n = inv_diag_.size();
    for (std::size_t i = 1; i < n; ++i) rhs[i] -= lower_[i] * rhs[i - 1];
    rhs[n - 1] *= inv_diag_[n - 1];
    for (std::size_t i = n - 1; i-- > 0;) rhs[i] = (rhs[i] - upper_[i] * rhs[i + 1]) * inv_diag_[i];
}

void tridiagonal_multiply(const std::vector<double>& lower, const std::vector<double>& diag,
                          const std::vector<double>& upper, const std::vector<double>& x,
                          std::vector<double>& y) {
    const std::size_t n = diag.size();
    y.resize(n);
    if (n == 1) {
        y[0] = diag[0] * x[0];
        return;
    }
    y[0] = diag[0] * x[0] + upper[0] * x[1];
    for (std::size_t i = 1; i + 1 < n; ++i) {
        y[i] = lower[i] * x[i - 1] + diag[i] * x[i] + upper[i] * x[i + 1];
    }
    y[n - 1] = lower[n - 1] * x[n - 2] + diag[n - 1] * x[n - 1];
}

}  // namespace qp
