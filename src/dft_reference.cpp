#include "fft/dft_reference.hpp"

#include <cmath>
#include <numbers>

namespace fft {

std::vector<std::complex<double>> dft_reference(const std::vector<std::complex<double>>& x) {
    const std::size_t n = x.size();
    std::vector<std::complex<double>> X(n, std::complex<double>(0.0, 0.0));

    for (std::size_t k = 0; k < n; ++k) {
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t n_idx = 0; n_idx < n; ++n_idx) {
            const double angle = -2.0 * std::numbers::pi_v<double> * static_cast<double>(k) *
                                 static_cast<double>(n_idx) / static_cast<double>(n);
            const std::complex<double> twiddle(std::cos(angle), std::sin(angle));
            sum += x[n_idx] * twiddle;
        }
        X[k] = sum;
    }

    return X;
}

std::vector<std::complex<double>> idft_reference(const std::vector<std::complex<double>>& X) {
    const std::size_t n = X.size();
    std::vector<std::complex<double>> x(n, std::complex<double>(0.0, 0.0));

    for (std::size_t n_idx = 0; n_idx < n; ++n_idx) {
        std::complex<double> sum(0.0, 0.0);
        for (std::size_t k = 0; k < n; ++k) {
            const double angle = 2.0 * std::numbers::pi_v<double> * static_cast<double>(k) *
                                 static_cast<double>(n_idx) / static_cast<double>(n);
            const std::complex<double> twiddle(std::cos(angle), std::sin(angle));
            sum += X[k] * twiddle;
        }
        x[n_idx] = sum / static_cast<double>(n);
    }

    return x;
}

}  // namespace fft
