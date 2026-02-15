#include "fft/dft_reference.hpp"
#include "fft/fft_dispatch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <numbers>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr double kRoundTripTolerance = 1e-10;
constexpr double kBinTolerance = 1e-10;
constexpr double kParsevalTolerance = 1e-10;

std::vector<fft::Algorithm> power_of_two_algorithms() {
    return {fft::Algorithm::Radix2Iterative, fft::Algorithm::Radix2Recursive};
}

std::vector<fft::Algorithm> all_algorithms() {
    return {fft::Algorithm::Radix2Iterative, fft::Algorithm::Radix2Recursive, fft::Algorithm::DirectDft};
}

std::vector<std::complex<double>> random_complex_vector(std::size_t n, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<std::complex<double>> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = std::complex<double>(dist(rng), dist(rng));
    }
    return x;
}

double l2_norm(const std::vector<std::complex<double>>& x) {
    double sum = 0.0;
    for (const auto& value : x) {
        sum += std::norm(value);
    }
    return std::sqrt(sum);
}

double relative_l2_error(const std::vector<std::complex<double>>& actual,
                         const std::vector<std::complex<double>>& expected) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error("Vector size mismatch in relative_l2_error.");
    }

    std::vector<std::complex<double>> diff(actual.size());
    for (std::size_t i = 0; i < actual.size(); ++i) {
        diff[i] = actual[i] - expected[i];
    }

    const double denom = std::max(l2_norm(expected), 1e-30);
    return l2_norm(diff) / denom;
}

double abs_error(const std::complex<double>& a, const std::complex<double>& b) {
    return std::abs(a - b);
}

void expect(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_round_trip_power_of_two_algorithms() {
    std::mt19937_64 rng(0xD1CEB00CULL);

    for (const fft::Algorithm algorithm : power_of_two_algorithms()) {
        for (std::size_t n = 2; n <= 4096; n <<= 1U) {
            const auto x = random_complex_vector(n, rng);
            const auto y = fft::ifft(fft::fft(x, algorithm), algorithm);

            const double rel_error = relative_l2_error(y, x);
            if (rel_error >= kRoundTripTolerance) {
                std::ostringstream oss;
                oss << "Round-trip relative L2 error too high for algorithm=" << fft::algorithm_name(algorithm)
                    << ", N=" << n << ": " << std::setprecision(16) << rel_error << " (tolerance "
                    << kRoundTripTolerance << ')';
                throw std::runtime_error(oss.str());
            }
        }
    }
}

void test_round_trip_direct_dft() {
    std::mt19937_64 rng(0x12345678ULL);
    constexpr std::array<std::size_t, 9> sizes = {2, 3, 4, 5, 8, 16, 24, 32, 64};

    for (const std::size_t n : sizes) {
        const auto x = random_complex_vector(n, rng);
        const auto y = fft::ifft(fft::fft(x, fft::Algorithm::DirectDft), fft::Algorithm::DirectDft);

        const double rel_error = relative_l2_error(y, x);
        if (rel_error >= kRoundTripTolerance) {
            std::ostringstream oss;
            oss << "Round-trip relative L2 error too high for direct_dft, N=" << n << ": "
                << std::setprecision(16) << rel_error << " (tolerance " << kRoundTripTolerance << ')';
            throw std::runtime_error(oss.str());
        }
    }
}

void test_fft_matches_dft_reference() {
    std::mt19937_64 rng(0xA5A5A5A5ULL);

    for (const fft::Algorithm algorithm : power_of_two_algorithms()) {
        for (std::size_t n = 2; n <= 256; n <<= 1U) {
            for (int sample = 0; sample < 4; ++sample) {
                const auto x = random_complex_vector(n, rng);
                const auto fast = fft::fft(x, algorithm);
                const auto ref = fft::dft_reference(x);

                for (std::size_t k = 0; k < n; ++k) {
                    const double err = abs_error(fast[k], ref[k]);
                    if (err >= kBinTolerance) {
                        std::ostringstream oss;
                        oss << "FFT vs DFT mismatch for algorithm=" << fft::algorithm_name(algorithm) << ", N="
                            << n << ", sample=" << sample << ", bin=" << k << ": error="
                            << std::setprecision(16) << err << " (tolerance " << kBinTolerance << ')';
                        throw std::runtime_error(oss.str());
                    }
                }
            }
        }
    }
}

void test_pure_complex_tone() {
    const std::size_t n = 256;
    const std::size_t tone = 37;

    std::vector<std::complex<double>> x(n);
    for (std::size_t i = 0; i < n; ++i) {
        const double angle = 2.0 * std::numbers::pi_v<double> * static_cast<double>(tone) *
                             static_cast<double>(i) / static_cast<double>(n);
        x[i] = std::complex<double>(std::cos(angle), std::sin(angle));
    }

    for (const fft::Algorithm algorithm : all_algorithms()) {
        const auto X = fft::fft(x, algorithm);
        const double expected_peak = static_cast<double>(n);
        const double peak_error = std::abs(X[tone] - std::complex<double>(expected_peak, 0.0));

        if (peak_error >= 1e-8) {
            std::ostringstream oss;
            oss << "Complex tone peak mismatch for algorithm=" << fft::algorithm_name(algorithm)
                << ": peak error=" << std::setprecision(16) << peak_error;
            throw std::runtime_error(oss.str());
        }

        const double leakage_limit = 1e-8;
        for (std::size_t k = 0; k < n; ++k) {
            if (k == tone) {
                continue;
            }
            if (std::abs(X[k]) >= leakage_limit) {
                std::ostringstream oss;
                oss << "Unexpected leakage for algorithm=" << fft::algorithm_name(algorithm) << " at bin " << k
                    << ": " << std::setprecision(16) << std::abs(X[k]);
                throw std::runtime_error(oss.str());
            }
        }
    }
}

void test_parseval_identity() {
    std::mt19937_64 rng(0x0F0F0F0FULL);

    for (const fft::Algorithm algorithm : power_of_two_algorithms()) {
        for (std::size_t n = 2; n <= 2048; n <<= 1U) {
            const auto x = random_complex_vector(n, rng);
            const auto X = fft::fft(x, algorithm);

            double time_energy = 0.0;
            for (const auto& value : x) {
                time_energy += std::norm(value);
            }

            double freq_energy = 0.0;
            for (const auto& value : X) {
                freq_energy += std::norm(value);
            }
            freq_energy /= static_cast<double>(n);

            const double relative_error = std::abs(time_energy - freq_energy) / std::max(time_energy, 1.0);
            if (relative_error >= kParsevalTolerance) {
                std::ostringstream oss;
                oss << "Parseval mismatch for algorithm=" << fft::algorithm_name(algorithm) << ", N=" << n
                    << ": relative error=" << std::setprecision(16) << relative_error << " (tolerance "
                    << kParsevalTolerance << ')';
                throw std::runtime_error(oss.str());
            }
        }
    }
}

void test_invalid_size_rejection() {
    std::vector<std::complex<double>> x(3, std::complex<double>(0.0, 0.0));

    for (const fft::Algorithm algorithm : power_of_two_algorithms()) {
        bool threw = false;
        try {
            auto data = x;
            fft::fft_inplace(data, algorithm);
        } catch (const std::invalid_argument&) {
            threw = true;
        }

        if (!threw) {
            throw std::runtime_error(std::string("Expected std::invalid_argument for algorithm=") +
                                     fft::algorithm_name(algorithm) + " and N=3.");
        }
    }

    auto direct_data = x;
    fft::fft_inplace(direct_data, fft::Algorithm::DirectDft);
}

void test_algorithm_parser() {
    expect(fft::parse_algorithm_name("radix2_iterative").has_value(), "Failed to parse radix2_iterative.");
    expect(fft::parse_algorithm_name("radix2_recursive").has_value(), "Failed to parse radix2_recursive.");
    expect(fft::parse_algorithm_name("direct_dft").has_value(), "Failed to parse direct_dft.");
    expect(!fft::parse_algorithm_name("unknown").has_value(), "Unknown algorithm should not parse.");
}

class TestRunner {
  public:
    template <typename Fn>
    void run(const std::string& name, Fn&& fn) {
        try {
            fn();
            ++passed_;
            std::cout << "[PASS] " << name << '\n';
        } catch (const std::exception& e) {
            ++failed_;
            std::cout << "[FAIL] " << name << " :: " << e.what() << '\n';
        } catch (...) {
            ++failed_;
            std::cout << "[FAIL] " << name << " :: unknown exception" << '\n';
        }
    }

    int failed() const {
        return failed_;
    }

    void print_summary() const {
        std::cout << "Summary: " << passed_ << " passed, " << failed_ << " failed" << '\n';
    }

  private:
    int passed_ = 0;
    int failed_ = 0;
};

}  // namespace

int main() {
    TestRunner runner;

    runner.run("Round-trip ifft(fft(x)) for radix-2 models", test_round_trip_power_of_two_algorithms);
    runner.run("Round-trip ifft(fft(x)) for direct DFT model", test_round_trip_direct_dft);
    runner.run("Radix-2 FFT models match O(N^2) DFT", test_fft_matches_dft_reference);
    runner.run("Pure complex tone concentration", test_pure_complex_tone);
    runner.run("Parseval identity", test_parseval_identity);
    runner.run("Invalid size rejection", test_invalid_size_rejection);
    runner.run("Algorithm parser", test_algorithm_parser);

    runner.print_summary();
    return runner.failed() == 0 ? 0 : 1;
}
