#include "fft/fft_mixed_radix_4_2_soa.hpp"

#include <bit>
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

namespace fft::mixed_radix_4_2_soa {
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

void apply_len2_stage(ComplexSoA& x) {
    const std::size_t n = x.size();
    for (std::size_t i = 0; i < n; i += 2U) {
        const double a_re = x.re[i];
        const double a_im = x.im[i];
        const double b_re = x.re[i + 1];
        const double b_im = x.im[i + 1];

        x.re[i] = a_re + b_re;
        x.im[i] = a_im + b_im;
        x.re[i + 1] = a_re - b_re;
        x.im[i + 1] = a_im - b_im;
    }
}

void apply_fused_radix4_stage_scalar(ComplexSoA& x, std::size_t len, bool inverse) {
    const std::size_t n = x.size();
    const std::size_t len2 = len * 2U;
    const std::size_t half = len / 2U;

    const double angle_len = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len);
    const double len_step_re = std::cos(angle_len);
    const double len_step_im = std::sin(angle_len);

    const double angle_len2 = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len2);
    const double len2_step_re = std::cos(angle_len2);
    const double len2_step_im = std::sin(angle_len2);

    const double quarter_re = 0.0;
    const double quarter_im = inverse ? 1.0 : -1.0;

    for (std::size_t i = 0; i < n; i += len2) {
        double w_len_re = 1.0;
        double w_len_im = 0.0;
        double w_len2_re = 1.0;
        double w_len2_im = 0.0;
        double w_len2_quarter_re = quarter_re;
        double w_len2_quarter_im = quarter_im;

        for (std::size_t j = 0; j < half; ++j) {
            const std::size_t i0 = i + j;
            const std::size_t i1 = i0 + half;
            const std::size_t i2 = i0 + len;
            const std::size_t i3 = i2 + half;

            const double a0_re = x.re[i0];
            const double a0_im = x.im[i0];
            const double a1_re = x.re[i1];
            const double a1_im = x.im[i1];
            const double b0_re = x.re[i2];
            const double b0_im = x.im[i2];
            const double b1_re = x.re[i3];
            const double b1_im = x.im[i3];

            const double a1_tw_re = a1_re * w_len_re - a1_im * w_len_im;
            const double a1_tw_im = a1_re * w_len_im + a1_im * w_len_re;
            const double b1_tw_re = b1_re * w_len_re - b1_im * w_len_im;
            const double b1_tw_im = b1_re * w_len_im + b1_im * w_len_re;

            const double p0_re = a0_re + a1_tw_re;
            const double p0_im = a0_im + a1_tw_im;
            const double p1_re = a0_re - a1_tw_re;
            const double p1_im = a0_im - a1_tw_im;
            const double q0_re = b0_re + b1_tw_re;
            const double q0_im = b0_im + b1_tw_im;
            const double q1_re = b0_re - b1_tw_re;
            const double q1_im = b0_im - b1_tw_im;

            const double t0_re = q0_re * w_len2_re - q0_im * w_len2_im;
            const double t0_im = q0_re * w_len2_im + q0_im * w_len2_re;
            const double t1_re = q1_re * w_len2_quarter_re - q1_im * w_len2_quarter_im;
            const double t1_im = q1_re * w_len2_quarter_im + q1_im * w_len2_quarter_re;

            x.re[i0] = p0_re + t0_re;
            x.im[i0] = p0_im + t0_im;
            x.re[i2] = p0_re - t0_re;
            x.im[i2] = p0_im - t0_im;
            x.re[i1] = p1_re + t1_re;
            x.im[i1] = p1_im + t1_im;
            x.re[i3] = p1_re - t1_re;
            x.im[i3] = p1_im - t1_im;

            const double next_w_len_re = w_len_re * len_step_re - w_len_im * len_step_im;
            const double next_w_len_im = w_len_re * len_step_im + w_len_im * len_step_re;
            w_len_re = next_w_len_re;
            w_len_im = next_w_len_im;

            const double next_w_len2_re = w_len2_re * len2_step_re - w_len2_im * len2_step_im;
            const double next_w_len2_im = w_len2_re * len2_step_im + w_len2_im * len2_step_re;
            w_len2_re = next_w_len2_re;
            w_len2_im = next_w_len2_im;

            const double next_w_len2_quarter_re =
                w_len2_quarter_re * len2_step_re - w_len2_quarter_im * len2_step_im;
            const double next_w_len2_quarter_im =
                w_len2_quarter_re * len2_step_im + w_len2_quarter_im * len2_step_re;
            w_len2_quarter_re = next_w_len2_quarter_re;
            w_len2_quarter_im = next_w_len2_quarter_im;
        }
    }
}

#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) void apply_fused_radix4_stage_avx2(ComplexSoA& x, std::size_t len, bool inverse) {
    const std::size_t n = x.size();
    const std::size_t len2 = len * 2U;
    const std::size_t half = len / 2U;

    const double angle_len = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len);
    const double len_step_re = std::cos(angle_len);
    const double len_step_im = std::sin(angle_len);

    const double angle_len2 = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len2);
    const double len2_step_re = std::cos(angle_len2);
    const double len2_step_im = std::sin(angle_len2);

    const double quarter_re = 0.0;
    const double quarter_im = inverse ? 1.0 : -1.0;

    std::vector<double> w_len_re(half);
    std::vector<double> w_len_im(half);
    std::vector<double> w_len2_re(half);
    std::vector<double> w_len2_im(half);
    std::vector<double> w_len2_quarter_re(half);
    std::vector<double> w_len2_quarter_im(half);

    double tw_len_re = 1.0;
    double tw_len_im = 0.0;
    double tw_len2_re = 1.0;
    double tw_len2_im = 0.0;
    double tw_len2_quarter_re = quarter_re;
    double tw_len2_quarter_im = quarter_im;

    for (std::size_t j = 0; j < half; ++j) {
        w_len_re[j] = tw_len_re;
        w_len_im[j] = tw_len_im;
        w_len2_re[j] = tw_len2_re;
        w_len2_im[j] = tw_len2_im;
        w_len2_quarter_re[j] = tw_len2_quarter_re;
        w_len2_quarter_im[j] = tw_len2_quarter_im;

        const double next_w_len_re = tw_len_re * len_step_re - tw_len_im * len_step_im;
        const double next_w_len_im = tw_len_re * len_step_im + tw_len_im * len_step_re;
        tw_len_re = next_w_len_re;
        tw_len_im = next_w_len_im;

        const double next_w_len2_re = tw_len2_re * len2_step_re - tw_len2_im * len2_step_im;
        const double next_w_len2_im = tw_len2_re * len2_step_im + tw_len2_im * len2_step_re;
        tw_len2_re = next_w_len2_re;
        tw_len2_im = next_w_len2_im;

        const double next_w_len2_quarter_re = tw_len2_quarter_re * len2_step_re - tw_len2_quarter_im * len2_step_im;
        const double next_w_len2_quarter_im = tw_len2_quarter_re * len2_step_im + tw_len2_quarter_im * len2_step_re;
        tw_len2_quarter_re = next_w_len2_quarter_re;
        tw_len2_quarter_im = next_w_len2_quarter_im;
    }

    for (std::size_t i = 0; i < n; i += len2) {
        std::size_t j = 0;
        for (; j + 4U <= half; j += 4U) {
            const std::size_t i0 = i + j;
            const std::size_t i1 = i0 + half;
            const std::size_t i2 = i0 + len;
            const std::size_t i3 = i2 + half;

            const __m256d a0_re = _mm256_loadu_pd(x.re.data() + i0);
            const __m256d a0_im = _mm256_loadu_pd(x.im.data() + i0);
            const __m256d a1_re = _mm256_loadu_pd(x.re.data() + i1);
            const __m256d a1_im = _mm256_loadu_pd(x.im.data() + i1);
            const __m256d b0_re = _mm256_loadu_pd(x.re.data() + i2);
            const __m256d b0_im = _mm256_loadu_pd(x.im.data() + i2);
            const __m256d b1_re = _mm256_loadu_pd(x.re.data() + i3);
            const __m256d b1_im = _mm256_loadu_pd(x.im.data() + i3);

            const __m256d wl_re = _mm256_loadu_pd(w_len_re.data() + j);
            const __m256d wl_im = _mm256_loadu_pd(w_len_im.data() + j);
            const __m256d wl2_re = _mm256_loadu_pd(w_len2_re.data() + j);
            const __m256d wl2_im = _mm256_loadu_pd(w_len2_im.data() + j);
            const __m256d wl2q_re = _mm256_loadu_pd(w_len2_quarter_re.data() + j);
            const __m256d wl2q_im = _mm256_loadu_pd(w_len2_quarter_im.data() + j);

            const __m256d a1_tw_re = _mm256_sub_pd(_mm256_mul_pd(a1_re, wl_re), _mm256_mul_pd(a1_im, wl_im));
            const __m256d a1_tw_im = _mm256_add_pd(_mm256_mul_pd(a1_re, wl_im), _mm256_mul_pd(a1_im, wl_re));
            const __m256d b1_tw_re = _mm256_sub_pd(_mm256_mul_pd(b1_re, wl_re), _mm256_mul_pd(b1_im, wl_im));
            const __m256d b1_tw_im = _mm256_add_pd(_mm256_mul_pd(b1_re, wl_im), _mm256_mul_pd(b1_im, wl_re));

            const __m256d p0_re = _mm256_add_pd(a0_re, a1_tw_re);
            const __m256d p0_im = _mm256_add_pd(a0_im, a1_tw_im);
            const __m256d p1_re = _mm256_sub_pd(a0_re, a1_tw_re);
            const __m256d p1_im = _mm256_sub_pd(a0_im, a1_tw_im);
            const __m256d q0_re = _mm256_add_pd(b0_re, b1_tw_re);
            const __m256d q0_im = _mm256_add_pd(b0_im, b1_tw_im);
            const __m256d q1_re = _mm256_sub_pd(b0_re, b1_tw_re);
            const __m256d q1_im = _mm256_sub_pd(b0_im, b1_tw_im);

            const __m256d t0_re = _mm256_sub_pd(_mm256_mul_pd(q0_re, wl2_re), _mm256_mul_pd(q0_im, wl2_im));
            const __m256d t0_im = _mm256_add_pd(_mm256_mul_pd(q0_re, wl2_im), _mm256_mul_pd(q0_im, wl2_re));
            const __m256d t1_re = _mm256_sub_pd(_mm256_mul_pd(q1_re, wl2q_re), _mm256_mul_pd(q1_im, wl2q_im));
            const __m256d t1_im = _mm256_add_pd(_mm256_mul_pd(q1_re, wl2q_im), _mm256_mul_pd(q1_im, wl2q_re));

            _mm256_storeu_pd(x.re.data() + i0, _mm256_add_pd(p0_re, t0_re));
            _mm256_storeu_pd(x.im.data() + i0, _mm256_add_pd(p0_im, t0_im));
            _mm256_storeu_pd(x.re.data() + i2, _mm256_sub_pd(p0_re, t0_re));
            _mm256_storeu_pd(x.im.data() + i2, _mm256_sub_pd(p0_im, t0_im));
            _mm256_storeu_pd(x.re.data() + i1, _mm256_add_pd(p1_re, t1_re));
            _mm256_storeu_pd(x.im.data() + i1, _mm256_add_pd(p1_im, t1_im));
            _mm256_storeu_pd(x.re.data() + i3, _mm256_sub_pd(p1_re, t1_re));
            _mm256_storeu_pd(x.im.data() + i3, _mm256_sub_pd(p1_im, t1_im));
        }

        for (; j < half; ++j) {
            const std::size_t i0 = i + j;
            const std::size_t i1 = i0 + half;
            const std::size_t i2 = i0 + len;
            const std::size_t i3 = i2 + half;

            const double a0_re = x.re[i0];
            const double a0_im = x.im[i0];
            const double a1_re = x.re[i1];
            const double a1_im = x.im[i1];
            const double b0_re = x.re[i2];
            const double b0_im = x.im[i2];
            const double b1_re = x.re[i3];
            const double b1_im = x.im[i3];

            const double a1_tw_re = a1_re * w_len_re[j] - a1_im * w_len_im[j];
            const double a1_tw_im = a1_re * w_len_im[j] + a1_im * w_len_re[j];
            const double b1_tw_re = b1_re * w_len_re[j] - b1_im * w_len_im[j];
            const double b1_tw_im = b1_re * w_len_im[j] + b1_im * w_len_re[j];

            const double p0_re = a0_re + a1_tw_re;
            const double p0_im = a0_im + a1_tw_im;
            const double p1_re = a0_re - a1_tw_re;
            const double p1_im = a0_im - a1_tw_im;
            const double q0_re = b0_re + b1_tw_re;
            const double q0_im = b0_im + b1_tw_im;
            const double q1_re = b0_re - b1_tw_re;
            const double q1_im = b0_im - b1_tw_im;

            const double t0_re = q0_re * w_len2_re[j] - q0_im * w_len2_im[j];
            const double t0_im = q0_re * w_len2_im[j] + q0_im * w_len2_re[j];
            const double t1_re = q1_re * w_len2_quarter_re[j] - q1_im * w_len2_quarter_im[j];
            const double t1_im = q1_re * w_len2_quarter_im[j] + q1_im * w_len2_quarter_re[j];

            x.re[i0] = p0_re + t0_re;
            x.im[i0] = p0_im + t0_im;
            x.re[i2] = p0_re - t0_re;
            x.im[i2] = p0_im - t0_im;
            x.re[i1] = p1_re + t1_re;
            x.im[i1] = p1_im + t1_im;
            x.re[i3] = p1_re - t1_re;
            x.im[i3] = p1_im - t1_im;
        }
    }
}

__attribute__((target("avx512f"))) void apply_fused_radix4_stage_avx512(ComplexSoA& x, std::size_t len,
                                                                         bool inverse) {
    const std::size_t n = x.size();
    const std::size_t len2 = len * 2U;
    const std::size_t half = len / 2U;

    const double angle_len = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len);
    const double len_step_re = std::cos(angle_len);
    const double len_step_im = std::sin(angle_len);

    const double angle_len2 = (inverse ? 2.0 : -2.0) * std::numbers::pi_v<double> / static_cast<double>(len2);
    const double len2_step_re = std::cos(angle_len2);
    const double len2_step_im = std::sin(angle_len2);

    const double quarter_re = 0.0;
    const double quarter_im = inverse ? 1.0 : -1.0;

    std::vector<double> w_len_re(half);
    std::vector<double> w_len_im(half);
    std::vector<double> w_len2_re(half);
    std::vector<double> w_len2_im(half);
    std::vector<double> w_len2_quarter_re(half);
    std::vector<double> w_len2_quarter_im(half);

    double tw_len_re = 1.0;
    double tw_len_im = 0.0;
    double tw_len2_re = 1.0;
    double tw_len2_im = 0.0;
    double tw_len2_quarter_re = quarter_re;
    double tw_len2_quarter_im = quarter_im;

    for (std::size_t j = 0; j < half; ++j) {
        w_len_re[j] = tw_len_re;
        w_len_im[j] = tw_len_im;
        w_len2_re[j] = tw_len2_re;
        w_len2_im[j] = tw_len2_im;
        w_len2_quarter_re[j] = tw_len2_quarter_re;
        w_len2_quarter_im[j] = tw_len2_quarter_im;

        const double next_w_len_re = tw_len_re * len_step_re - tw_len_im * len_step_im;
        const double next_w_len_im = tw_len_re * len_step_im + tw_len_im * len_step_re;
        tw_len_re = next_w_len_re;
        tw_len_im = next_w_len_im;

        const double next_w_len2_re = tw_len2_re * len2_step_re - tw_len2_im * len2_step_im;
        const double next_w_len2_im = tw_len2_re * len2_step_im + tw_len2_im * len2_step_re;
        tw_len2_re = next_w_len2_re;
        tw_len2_im = next_w_len2_im;

        const double next_w_len2_quarter_re = tw_len2_quarter_re * len2_step_re - tw_len2_quarter_im * len2_step_im;
        const double next_w_len2_quarter_im = tw_len2_quarter_re * len2_step_im + tw_len2_quarter_im * len2_step_re;
        tw_len2_quarter_re = next_w_len2_quarter_re;
        tw_len2_quarter_im = next_w_len2_quarter_im;
    }

    for (std::size_t i = 0; i < n; i += len2) {
        std::size_t j = 0;
        for (; j + 8U <= half; j += 8U) {
            const std::size_t i0 = i + j;
            const std::size_t i1 = i0 + half;
            const std::size_t i2 = i0 + len;
            const std::size_t i3 = i2 + half;

            const __m512d a0_re = _mm512_loadu_pd(x.re.data() + i0);
            const __m512d a0_im = _mm512_loadu_pd(x.im.data() + i0);
            const __m512d a1_re = _mm512_loadu_pd(x.re.data() + i1);
            const __m512d a1_im = _mm512_loadu_pd(x.im.data() + i1);
            const __m512d b0_re = _mm512_loadu_pd(x.re.data() + i2);
            const __m512d b0_im = _mm512_loadu_pd(x.im.data() + i2);
            const __m512d b1_re = _mm512_loadu_pd(x.re.data() + i3);
            const __m512d b1_im = _mm512_loadu_pd(x.im.data() + i3);

            const __m512d wl_re = _mm512_loadu_pd(w_len_re.data() + j);
            const __m512d wl_im = _mm512_loadu_pd(w_len_im.data() + j);
            const __m512d wl2_re = _mm512_loadu_pd(w_len2_re.data() + j);
            const __m512d wl2_im = _mm512_loadu_pd(w_len2_im.data() + j);
            const __m512d wl2q_re = _mm512_loadu_pd(w_len2_quarter_re.data() + j);
            const __m512d wl2q_im = _mm512_loadu_pd(w_len2_quarter_im.data() + j);

            const __m512d a1_tw_re = _mm512_sub_pd(_mm512_mul_pd(a1_re, wl_re), _mm512_mul_pd(a1_im, wl_im));
            const __m512d a1_tw_im = _mm512_add_pd(_mm512_mul_pd(a1_re, wl_im), _mm512_mul_pd(a1_im, wl_re));
            const __m512d b1_tw_re = _mm512_sub_pd(_mm512_mul_pd(b1_re, wl_re), _mm512_mul_pd(b1_im, wl_im));
            const __m512d b1_tw_im = _mm512_add_pd(_mm512_mul_pd(b1_re, wl_im), _mm512_mul_pd(b1_im, wl_re));

            const __m512d p0_re = _mm512_add_pd(a0_re, a1_tw_re);
            const __m512d p0_im = _mm512_add_pd(a0_im, a1_tw_im);
            const __m512d p1_re = _mm512_sub_pd(a0_re, a1_tw_re);
            const __m512d p1_im = _mm512_sub_pd(a0_im, a1_tw_im);
            const __m512d q0_re = _mm512_add_pd(b0_re, b1_tw_re);
            const __m512d q0_im = _mm512_add_pd(b0_im, b1_tw_im);
            const __m512d q1_re = _mm512_sub_pd(b0_re, b1_tw_re);
            const __m512d q1_im = _mm512_sub_pd(b0_im, b1_tw_im);

            const __m512d t0_re = _mm512_sub_pd(_mm512_mul_pd(q0_re, wl2_re), _mm512_mul_pd(q0_im, wl2_im));
            const __m512d t0_im = _mm512_add_pd(_mm512_mul_pd(q0_re, wl2_im), _mm512_mul_pd(q0_im, wl2_re));
            const __m512d t1_re = _mm512_sub_pd(_mm512_mul_pd(q1_re, wl2q_re), _mm512_mul_pd(q1_im, wl2q_im));
            const __m512d t1_im = _mm512_add_pd(_mm512_mul_pd(q1_re, wl2q_im), _mm512_mul_pd(q1_im, wl2q_re));

            _mm512_storeu_pd(x.re.data() + i0, _mm512_add_pd(p0_re, t0_re));
            _mm512_storeu_pd(x.im.data() + i0, _mm512_add_pd(p0_im, t0_im));
            _mm512_storeu_pd(x.re.data() + i2, _mm512_sub_pd(p0_re, t0_re));
            _mm512_storeu_pd(x.im.data() + i2, _mm512_sub_pd(p0_im, t0_im));
            _mm512_storeu_pd(x.re.data() + i1, _mm512_add_pd(p1_re, t1_re));
            _mm512_storeu_pd(x.im.data() + i1, _mm512_add_pd(p1_im, t1_im));
            _mm512_storeu_pd(x.re.data() + i3, _mm512_sub_pd(p1_re, t1_re));
            _mm512_storeu_pd(x.im.data() + i3, _mm512_sub_pd(p1_im, t1_im));
        }

        for (; j < half; ++j) {
            const std::size_t i0 = i + j;
            const std::size_t i1 = i0 + half;
            const std::size_t i2 = i0 + len;
            const std::size_t i3 = i2 + half;

            const double a0_re = x.re[i0];
            const double a0_im = x.im[i0];
            const double a1_re = x.re[i1];
            const double a1_im = x.im[i1];
            const double b0_re = x.re[i2];
            const double b0_im = x.im[i2];
            const double b1_re = x.re[i3];
            const double b1_im = x.im[i3];

            const double a1_tw_re = a1_re * w_len_re[j] - a1_im * w_len_im[j];
            const double a1_tw_im = a1_re * w_len_im[j] + a1_im * w_len_re[j];
            const double b1_tw_re = b1_re * w_len_re[j] - b1_im * w_len_im[j];
            const double b1_tw_im = b1_re * w_len_im[j] + b1_im * w_len_re[j];

            const double p0_re = a0_re + a1_tw_re;
            const double p0_im = a0_im + a1_tw_im;
            const double p1_re = a0_re - a1_tw_re;
            const double p1_im = a0_im - a1_tw_im;
            const double q0_re = b0_re + b1_tw_re;
            const double q0_im = b0_im + b1_tw_im;
            const double q1_re = b0_re - b1_tw_re;
            const double q1_im = b0_im - b1_tw_im;

            const double t0_re = q0_re * w_len2_re[j] - q0_im * w_len2_im[j];
            const double t0_im = q0_re * w_len2_im[j] + q0_im * w_len2_re[j];
            const double t1_re = q1_re * w_len2_quarter_re[j] - q1_im * w_len2_quarter_im[j];
            const double t1_im = q1_re * w_len2_quarter_im[j] + q1_im * w_len2_quarter_re[j];

            x.re[i0] = p0_re + t0_re;
            x.im[i0] = p0_im + t0_im;
            x.re[i2] = p0_re - t0_re;
            x.im[i2] = p0_im - t0_im;
            x.re[i1] = p1_re + t1_re;
            x.im[i1] = p1_im + t1_im;
            x.re[i3] = p1_re - t1_re;
            x.im[i3] = p1_im - t1_im;
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

void fft_core_scalar(ComplexSoA& x, bool inverse) {
    const std::size_t n = x.size();
    const std::size_t log2_n = std::countr_zero(n);

    bit_reversal_permute(x);

    if ((log2_n & 1U) != 0U) {
        apply_len2_stage(x);
    }

    for (std::size_t len = ((log2_n & 1U) != 0U) ? 4U : 2U; len < n; len <<= 2U) {
        apply_fused_radix4_stage_scalar(x, len, inverse);
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
__attribute__((target("avx2")))
#endif
void fft_core_avx2_dispatch(ComplexSoA& x, bool inverse) {
    const std::size_t n = x.size();
    const std::size_t log2_n = std::countr_zero(n);

    bit_reversal_permute(x);

    if ((log2_n & 1U) != 0U) {
        apply_len2_stage(x);
    }

#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
    for (std::size_t len = ((log2_n & 1U) != 0U) ? 4U : 2U; len < n; len <<= 2U) {
        apply_fused_radix4_stage_avx2(x, len, inverse);
    }
#endif

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(n);
#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
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
#else
        for (std::size_t i = 0; i < n; ++i) {
            x.re[i] *= scale;
            x.im[i] *= scale;
        }
#endif
    }
}

#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx512f")))
#endif
void fft_core_avx512_dispatch(ComplexSoA& x, bool inverse) {
    const std::size_t n = x.size();
    const std::size_t log2_n = std::countr_zero(n);

    bit_reversal_permute(x);

    if ((log2_n & 1U) != 0U) {
        apply_len2_stage(x);
    }

#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
    for (std::size_t len = ((log2_n & 1U) != 0U) ? 4U : 2U; len < n; len <<= 2U) {
        apply_fused_radix4_stage_avx512(x, len, inverse);
    }
#endif

    if (inverse) {
        const double scale = 1.0 / static_cast<double>(n);
#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
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
#else
        for (std::size_t i = 0; i < n; ++i) {
            x.re[i] *= scale;
            x.im[i] *= scale;
        }
#endif
    }
}

void fft_core(ComplexSoA& x, bool inverse) {
    const std::size_t n = x.size();
    if (!is_power_of_two(n)) {
        throw std::invalid_argument("Mixed radix 4/2 SoA FFT input size must be a non-zero power of two.");
    }
    if (n == 1) {
        return;
    }

#if FFT_HAS_X86_SIMD && (defined(__GNUC__) || defined(__clang__))
    if (cpu_supports_avx512()) {
        fft_core_avx512_dispatch(x, inverse);
        return;
    }
    if (cpu_supports_avx2()) {
        fft_core_avx2_dispatch(x, inverse);
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

}  // namespace fft::mixed_radix_4_2_soa
