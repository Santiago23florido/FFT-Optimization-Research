#include "fft/fft_split_radix.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace fft::split_radix {
namespace {

void split_radix_recursive(std::vector<std::complex<double>>& x, bool inverse) {
    const std::size_t n = x.size();
    if (n <= 1) {
        return;
    }

    if (n == 2) {
        const std::complex<double> a = x[0];
        const std::complex<double> b = x[1];
        x[0] = a + b;
        x[1] = a - b;
        return;
    }

    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd_1(n / 4);
    std::vector<std::complex<double>> odd_3(n / 4);

    for (std::size_t i = 0; i < n / 4; ++i) {
        even[2 * i] = x[4 * i];
        even[2 * i + 1] = x[4 * i + 2];
        odd_1[i] = x[4 * i + 1];
        odd_3[i] = x[4 * i + 3];
    }

    split_radix_recursive(even, inverse);
    split_radix_recursive(odd_1, inverse);
    split_radix_recursive(odd_3, inverse);

    const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(n);
    const std::complex<double> w_step(std::cos(angle), std::sin(angle));
    const std::complex<double> w_step_3 = w_step * w_step * w_step;

    std::complex<double> w_1(1.0, 0.0);
    std::complex<double> w_3(1.0, 0.0);
    const std::complex<double> j_sign(0.0, inverse ? 1.0 : -1.0);

    for (std::size_t k = 0; k < n / 4; ++k) {
        const std::complex<double> t_1 = w_1 * odd_1[k];
        const std::complex<double> t_3 = w_3 * odd_3[k];

        const std::complex<double> e_0 = even[k];
        const std::complex<double> e_1 = even[k + n / 4];

        x[k] = e_0 + t_1 + t_3;
        x[k + n / 2] = e_0 - t_1 - t_3;

        // Split-radix odd branches are combined through j-rotated differences.
        const std::complex<double> odd_diff = t_1 - t_3;
        x[k + n / 4] = e_1 + j_sign * odd_diff;
        x[k + (3 * n) / 4] = e_1 - j_sign * odd_diff;

        w_1 *= w_step;
        w_3 *= w_step_3;
    }
}

}  // namespace

bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1U)) == 0;
}

void fft_inplace(std::vector<std::complex<double>>& x) {
    if (!is_power_of_two(x.size())) {
        throw std::invalid_argument("Split-radix FFT input size must be a non-zero power of two.");
    }

    split_radix_recursive(x, false);
}

void ifft_inplace(std::vector<std::complex<double>>& x) {
    if (!is_power_of_two(x.size())) {
        throw std::invalid_argument("Split-radix FFT input size must be a non-zero power of two.");
    }

    split_radix_recursive(x, true);

    const double scale = 1.0 / static_cast<double>(x.size());
    for (std::complex<double>& value : x) {
        value *= scale;
    }
}

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x) {
    fft_inplace(x);
    return x;
}

std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x) {
    ifft_inplace(x);
    return x;
}

}  // namespace fft::split_radix
