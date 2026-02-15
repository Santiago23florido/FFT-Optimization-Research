#include "fft/fft_radix2_soa.hpp"

#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define FFT_HAS_X86_SIMD 1
#include <immintrin.h>
#else
#define FFT_HAS_X86_SIMD 0
#endif

namespace fft::radix2_soa {
namespace {

void bit_reversal_permute(ComplexSoA& x) {
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
            std::swap(x.re[i], x.re[j]);
            std::swap(x.im[i], x.im[j]);
        }
    }
}

void fft_core_scalar(ComplexSoA& x, bool inverse) {
    const std::size_t n = x.size();
    bit_reversal_permute(x);

    for (std::size_t len = 2; len <= n; len <<= 1U) {
        const std::size_t half = len / 2U;
        const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len);
        const double step_re = std::cos(angle);
        const double step_im = std::sin(angle);

        for (std::size_t i = 0; i < n; i += len) {
            double w_re = 1.0;
            double w_im = 0.0;

            for (std::size_t j = 0; j < half; ++j) {
                const std::size_t idx0 = i + j;
                const std::size_t idx1 = idx0 + half;

                const double even_re = x.re[idx0];
                const double even_im = x.im[idx0];

                const double odd_src_re = x.re[idx1];
                const double odd_src_im = x.im[idx1];
                const double odd_re = odd_src_re * w_re - odd_src_im * w_im;
                const double odd_im = odd_src_re * w_im + odd_src_im * w_re;

                x.re[idx0] = even_re + odd_re;
                x.im[idx0] = even_im + odd_im;
                x.re[idx1] = even_re - odd_re;
                x.im[idx1] = even_im - odd_im;

                const double next_w_re = w_re * step_re - w_im * step_im;
                const double next_w_im = w_re * step_im + w_im * step_re;
                w_re = next_w_re;
                w_im = next_w_im;
            }
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(n);
        for (std::size_t i = 0; i < n; ++i) {
            x.re[i] *= scale;
            x.im[i] *= scale;
        }
    }
}

#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) void fft_core_avx2(ComplexSoA& x, bool inverse) {
    const std::size_t n = x.size();
    bit_reversal_permute(x);

    std::vector<double> tw_re(n / 2U);
    std::vector<double> tw_im(n / 2U);

    for (std::size_t len = 2; len <= n; len <<= 1U) {
        const std::size_t half = len / 2U;
        const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len);
        const double step_re = std::cos(angle);
        const double step_im = std::sin(angle);

        double w_re = 1.0;
        double w_im = 0.0;
        for (std::size_t j = 0; j < half; ++j) {
            tw_re[j] = w_re;
            tw_im[j] = w_im;
            const double next_w_re = w_re * step_re - w_im * step_im;
            const double next_w_im = w_re * step_im + w_im * step_re;
            w_re = next_w_re;
            w_im = next_w_im;
        }

        for (std::size_t i = 0; i < n; i += len) {
            std::size_t j = 0;
            for (; j + 4U <= half; j += 4U) {
                const std::size_t idx0 = i + j;
                const std::size_t idx1 = idx0 + half;

                const __m256d twr = _mm256_loadu_pd(tw_re.data() + j);
                const __m256d twi = _mm256_loadu_pd(tw_im.data() + j);
                const __m256d even_re = _mm256_loadu_pd(x.re.data() + idx0);
                const __m256d even_im = _mm256_loadu_pd(x.im.data() + idx0);
                const __m256d odd_re = _mm256_loadu_pd(x.re.data() + idx1);
                const __m256d odd_im = _mm256_loadu_pd(x.im.data() + idx1);

                const __m256d odd_tw_re = _mm256_sub_pd(_mm256_mul_pd(odd_re, twr), _mm256_mul_pd(odd_im, twi));
                const __m256d odd_tw_im = _mm256_add_pd(_mm256_mul_pd(odd_re, twi), _mm256_mul_pd(odd_im, twr));

                _mm256_storeu_pd(x.re.data() + idx0, _mm256_add_pd(even_re, odd_tw_re));
                _mm256_storeu_pd(x.im.data() + idx0, _mm256_add_pd(even_im, odd_tw_im));
                _mm256_storeu_pd(x.re.data() + idx1, _mm256_sub_pd(even_re, odd_tw_re));
                _mm256_storeu_pd(x.im.data() + idx1, _mm256_sub_pd(even_im, odd_tw_im));
            }

            for (; j < half; ++j) {
                const std::size_t idx0 = i + j;
                const std::size_t idx1 = idx0 + half;

                const double even_re = x.re[idx0];
                const double even_im = x.im[idx0];

                const double odd_src_re = x.re[idx1];
                const double odd_src_im = x.im[idx1];
                const double odd_val_re = odd_src_re * tw_re[j] - odd_src_im * tw_im[j];
                const double odd_val_im = odd_src_re * tw_im[j] + odd_src_im * tw_re[j];

                x.re[idx0] = even_re + odd_val_re;
                x.im[idx0] = even_im + odd_val_im;
                x.re[idx1] = even_re - odd_val_re;
                x.im[idx1] = even_im - odd_val_im;
            }
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(n);
        const __m256d scale_v = _mm256_set1_pd(scale);
        std::size_t i = 0;
        for (; i + 4U <= n; i += 4U) {
            _mm256_storeu_pd(x.re.data() + i, _mm256_mul_pd(_mm256_loadu_pd(x.re.data() + i), scale_v));
            _mm256_storeu_pd(x.im.data() + i, _mm256_mul_pd(_mm256_loadu_pd(x.im.data() + i), scale_v));
        }
        for (; i < n; ++i) {
            x.re[i] *= scale;
            x.im[i] *= scale;
        }
    }
}

__attribute__((target("avx512f"))) void fft_core_avx512(ComplexSoA& x, bool inverse) {
    const std::size_t n = x.size();
    bit_reversal_permute(x);

    std::vector<double> tw_re(n / 2U);
    std::vector<double> tw_im(n / 2U);

    for (std::size_t len = 2; len <= n; len <<= 1U) {
        const std::size_t half = len / 2U;
        const double angle = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len);
        const double step_re = std::cos(angle);
        const double step_im = std::sin(angle);

        double w_re = 1.0;
        double w_im = 0.0;
        for (std::size_t j = 0; j < half; ++j) {
            tw_re[j] = w_re;
            tw_im[j] = w_im;
            const double next_w_re = w_re * step_re - w_im * step_im;
            const double next_w_im = w_re * step_im + w_im * step_re;
            w_re = next_w_re;
            w_im = next_w_im;
        }

        for (std::size_t i = 0; i < n; i += len) {
            std::size_t j = 0;
            for (; j + 8U <= half; j += 8U) {
                const std::size_t idx0 = i + j;
                const std::size_t idx1 = idx0 + half;

                const __m512d twr = _mm512_loadu_pd(tw_re.data() + j);
                const __m512d twi = _mm512_loadu_pd(tw_im.data() + j);
                const __m512d even_re = _mm512_loadu_pd(x.re.data() + idx0);
                const __m512d even_im = _mm512_loadu_pd(x.im.data() + idx0);
                const __m512d odd_re = _mm512_loadu_pd(x.re.data() + idx1);
                const __m512d odd_im = _mm512_loadu_pd(x.im.data() + idx1);

                const __m512d odd_tw_re = _mm512_sub_pd(_mm512_mul_pd(odd_re, twr), _mm512_mul_pd(odd_im, twi));
                const __m512d odd_tw_im = _mm512_add_pd(_mm512_mul_pd(odd_re, twi), _mm512_mul_pd(odd_im, twr));

                _mm512_storeu_pd(x.re.data() + idx0, _mm512_add_pd(even_re, odd_tw_re));
                _mm512_storeu_pd(x.im.data() + idx0, _mm512_add_pd(even_im, odd_tw_im));
                _mm512_storeu_pd(x.re.data() + idx1, _mm512_sub_pd(even_re, odd_tw_re));
                _mm512_storeu_pd(x.im.data() + idx1, _mm512_sub_pd(even_im, odd_tw_im));
            }

            for (; j < half; ++j) {
                const std::size_t idx0 = i + j;
                const std::size_t idx1 = idx0 + half;

                const double even_re = x.re[idx0];
                const double even_im = x.im[idx0];

                const double odd_src_re = x.re[idx1];
                const double odd_src_im = x.im[idx1];
                const double odd_val_re = odd_src_re * tw_re[j] - odd_src_im * tw_im[j];
                const double odd_val_im = odd_src_re * tw_im[j] + odd_src_im * tw_re[j];

                x.re[idx0] = even_re + odd_val_re;
                x.im[idx0] = even_im + odd_val_im;
                x.re[idx1] = even_re - odd_val_re;
                x.im[idx1] = even_im - odd_val_im;
            }
        }
    }

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(n);
        const __m512d scale_v = _mm512_set1_pd(scale);
        std::size_t i = 0;
        for (; i + 8U <= n; i += 8U) {
            _mm512_storeu_pd(x.re.data() + i, _mm512_mul_pd(_mm512_loadu_pd(x.re.data() + i), scale_v));
            _mm512_storeu_pd(x.im.data() + i, _mm512_mul_pd(_mm512_loadu_pd(x.im.data() + i), scale_v));
        }
        for (; i < n; ++i) {
            x.re[i] *= scale;
            x.im[i] *= scale;
        }
    }
}

bool cpu_supports_avx2() {
    return __builtin_cpu_supports("avx2");
}

bool cpu_supports_avx512() {
    return __builtin_cpu_supports("avx512f");
}
#endif

void fft_core(ComplexSoA& x, bool inverse) {
    const std::size_t n = x.size();
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("Radix-2 SoA FFT input size must be a non-zero power of two.");
    }

#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx512()) {
        fft_core_avx512(x, inverse);
        return;
    }
    if (cpu_supports_avx2()) {
        fft_core_avx2(x, inverse);
        return;
    }
#endif

    fft_core_scalar(x, inverse);
}

}  // namespace

bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1U)) == 0;
}

void fft_inplace(ComplexSoA& x) {
    fft_core(x, false);
}

void ifft_inplace(ComplexSoA& x) {
    fft_core(x, true);
}

void fft_inplace(std::vector<std::complex<double>>& x) {
    ComplexSoA soa = ComplexSoA::from_aos(x);
    fft_inplace(soa);
    x = soa.to_aos();
}

void ifft_inplace(std::vector<std::complex<double>>& x) {
    ComplexSoA soa = ComplexSoA::from_aos(x);
    ifft_inplace(soa);
    x = soa.to_aos();
}

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x) {
    fft_inplace(x);
    return x;
}

std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x) {
    ifft_inplace(x);
    return x;
}

}  // namespace fft::radix2_soa
