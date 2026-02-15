#include "fft/fft_radix2_recursive.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <vector>

namespace fft::radix2_recursive {
namespace {

void fft_recursive(std::vector<std::complex<double>>& x, bool inverse) {
    const std::size_t n = x.size();
    if (n <= 1) {
        return;
    }

    std::vector<std::complex<double>> even(n / 2);
    std::vector<std::complex<double>> odd(n / 2);

    for (std::size_t i = 0; i < n / 2; ++i) {
        even[i] = x[2 * i];
        odd[i] = x[2 * i + 1];
    }

    fft_recursive(even, inverse);
    fft_recursive(odd, inverse);

    const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(n);
    const std::complex<double> w_n(std::cos(angle), std::sin(angle));
    std::complex<double> w(1.0, 0.0);

    for (std::size_t k = 0; k < n / 2; ++k) {
        const std::complex<double> t = w * odd[k];
        x[k] = even[k] + t;
        x[k + n / 2] = even[k] - t;
        w *= w_n;
    }
}

}  // namespace

bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1U)) == 0;
}

void fft_inplace(std::vector<std::complex<double>>& x) {
    if (!is_power_of_two(x.size())) {
        throw std::invalid_argument("Radix-2 recursive FFT input size must be a non-zero power of two.");
    }
    fft_recursive(x, false);
}

void ifft_inplace(std::vector<std::complex<double>>& x) {
    if (!is_power_of_two(x.size())) {
        throw std::invalid_argument("Radix-2 recursive FFT input size must be a non-zero power of two.");
    }
    fft_recursive(x, true);

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

}  // namespace fft::radix2_recursive
