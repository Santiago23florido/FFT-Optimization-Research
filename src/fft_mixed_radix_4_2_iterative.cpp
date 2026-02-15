#include "fft/fft_mixed_radix_4_2_iterative.hpp"

#include <bit>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

namespace fft::mixed_radix_4_2_iterative {
namespace {

void bit_reversal_permute(std::vector<std::complex<double>>& x) {
    const std::size_t n = x.size();
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1U;
        while ((j & bit) != 0U) {
            j ^= bit;
            bit >>= 1U;
        }
        j ^= bit;
        if (i < j) {
            std::swap(x[i], x[j]);
        }
    }
}

void apply_len2_stage(std::vector<std::complex<double>>& x) {
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; i += 2U) {
        const std::complex<double> a = x[i];
        const std::complex<double> b = x[i + 1];
        x[i] = a + b;
        x[i + 1] = a - b;
    }
}

void apply_fused_radix4_stage(std::vector<std::complex<double>>& x, std::size_t len, bool inverse) {
    const std::size_t n = x.size();
    const std::size_t len2 = len * 2U;
    const std::size_t half = len / 2U;

    const double angle_len = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len);
    const std::complex<double> w_len_step(std::cos(angle_len), std::sin(angle_len));

    const double angle_len2 = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len2);
    const std::complex<double> w_len2_step(std::cos(angle_len2), std::sin(angle_len2));
    const std::complex<double> w_quarter(0.0, inverse ? 1.0 : -1.0);

    for (std::size_t i = 0; i < n; i += len2) {
        std::complex<double> w_len(1.0, 0.0);
        std::complex<double> w_len2(1.0, 0.0);
        std::complex<double> w_len2_quarter = w_quarter;

        for (std::size_t j = 0; j < half; ++j) {
            const std::size_t i0 = i + j;
            const std::size_t i1 = i0 + half;
            const std::size_t i2 = i0 + len;
            const std::size_t i3 = i2 + half;

            const std::complex<double> a0 = x[i0];
            const std::complex<double> a1 = x[i1];
            const std::complex<double> b0 = x[i2];
            const std::complex<double> b1 = x[i3];

            const std::complex<double> a1_tw = a1 * w_len;
            const std::complex<double> b1_tw = b1 * w_len;

            const std::complex<double> p0 = a0 + a1_tw;
            const std::complex<double> p1 = a0 - a1_tw;
            const std::complex<double> q0 = b0 + b1_tw;
            const std::complex<double> q1 = b0 - b1_tw;

            const std::complex<double> t0 = q0 * w_len2;
            const std::complex<double> t1 = q1 * w_len2_quarter;

            x[i0] = p0 + t0;
            x[i2] = p0 - t0;
            x[i1] = p1 + t1;
            x[i3] = p1 - t1;

            w_len *= w_len_step;
            w_len2 *= w_len2_step;
            w_len2_quarter *= w_len2_step;
        }
    }
}

void fft_core(std::vector<std::complex<double>>& x, bool inverse) {
    const std::size_t n = x.size();
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("Mixed radix 4/2 iterative FFT input size must be a non-zero power of two.");
    }
    if (n == 1) {
        return;
    }

    const std::size_t log2_n = std::countr_zero(n);

    bit_reversal_permute(x);

    if ((log2_n & 1U) != 0U) {
        apply_len2_stage(x);
    }

    for (std::size_t len = ((log2_n & 1U) != 0U) ? 4U : 2U; len < n; len <<= 2U) {
        apply_fused_radix4_stage(x, len, inverse);
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

}  // namespace fft::mixed_radix_4_2_iterative
