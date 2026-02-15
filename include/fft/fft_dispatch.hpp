#ifndef FFT_FFT_DISPATCH_HPP
#define FFT_FFT_DISPATCH_HPP

#include "fft/fft_soa.hpp"

#include <complex>
#include <optional>
#include <string_view>
#include <vector>

namespace fft {

enum class Algorithm {
    Radix2Iterative,
    MixedRadix42Iterative,
    Radix2SoA,
    MixedRadix42SoA,
    Radix2Recursive,
    SplitRadix,
    DirectDft
};

const char* algorithm_name(Algorithm algorithm);
std::optional<Algorithm> parse_algorithm_name(std::string_view name);
std::vector<Algorithm> supported_algorithms();

void fft_inplace(std::vector<std::complex<double>>& x, Algorithm algorithm);
void ifft_inplace(std::vector<std::complex<double>>& x, Algorithm algorithm);
void fft_inplace_soa(ComplexSoA& x, Algorithm algorithm);
void ifft_inplace_soa(ComplexSoA& x, Algorithm algorithm);

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x, Algorithm algorithm);
std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x, Algorithm algorithm);

void fft_split_radix_inplace(std::vector<std::complex<double>>& x);
void ifft_split_radix_inplace(std::vector<std::complex<double>>& x);
void fft_mixed_radix_4_2_inplace(std::vector<std::complex<double>>& x);
void ifft_mixed_radix_4_2_inplace(std::vector<std::complex<double>>& x);
void fft_radix2_soa_inplace(ComplexSoA& x);
void ifft_radix2_soa_inplace(ComplexSoA& x);
void fft_mixed42_soa_inplace(ComplexSoA& x);
void ifft_mixed42_soa_inplace(ComplexSoA& x);

}  // namespace fft

#endif
