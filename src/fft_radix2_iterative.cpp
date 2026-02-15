#include "fft/fft_radix2_iterative.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace fft::radix2_iterative {
namespace {

std::size_t reverse_bits(std::size_t value, std::size_t bit_count) {
    std::size_t reversed = 0;
    for (std::size_t bit = 0; bit < bit_count; ++bit) {
        reversed = (reversed << 1U) | (value & 1U);
        value >>= 1U;
    }
    return reversed;
}

void bit_reversal_permute(std::vector<std::complex<double>>& x) {
    const std::size_t n = x.size();
    std::size_t bit_count = 0;
    for (std::size_t t = n; t > 1; t >>= 1U) {
        ++bit_count;
    }

    for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = reverse_bits(i, bit_count);
        if (j > i) {
            std::swap(x[i], x[j]);
        }
    }
}

void fft_core(std::vector<std::complex<double>>& x, bool inverse) {
    const std::size_t n = x.size();
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("Radix-2 iterative FFT input size must be a non-zero power of two.");
    }

    bit_reversal_permute(x);

    for (std::size_t len = 2; len <= n; len <<= 1U) {
        const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len);
        const std::complex<double> w_len(std::cos(angle), std::sin(angle));

        for (std::size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (std::size_t j = 0; j < (len / 2); ++j) {
                const std::complex<double> even = x[i + j];
                const std::complex<double> odd = x[i + j + (len / 2)] * w;

                // Butterfly: (E, O) -> (E + W*O, E - W*O).
                x[i + j] = even + odd;
                x[i + j + (len / 2)] = even - odd;
                w *= w_len;
            }
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(n);
        for (std::complex<double>& value : x) {
            value *= scale;
        }
    }
}

}  // namespace

bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1U)) == 0;
}

void fft_inplace(std::vector<std::complex<double>>& x) {
    fft_core(x, false);
}

void ifft_inplace(std::vector<std::complex<double>>& x) {
    fft_core(x, true);
}

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x) {
    fft_inplace(x);
    return x;
}

std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x) {
    ifft_inplace(x);
    return x;
}

}  // namespace fft::radix2_iterative
