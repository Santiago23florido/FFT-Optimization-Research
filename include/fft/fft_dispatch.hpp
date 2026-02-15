#ifndef FFT_FFT_DISPATCH_HPP
#define FFT_FFT_DISPATCH_HPP

#include <complex>
#include <optional>
#include <string_view>
#include <vector>

namespace fft {

enum class Algorithm {
    Radix2Iterative,
    Radix2Recursive,
    SplitRadix,
    DirectDft
};

const char* algorithm_name(Algorithm algorithm);
std::optional<Algorithm> parse_algorithm_name(std::string_view name);
std::vector<Algorithm> supported_algorithms();

void fft_inplace(std::vector<std::complex<double>>& x, Algorithm algorithm);
void ifft_inplace(std::vector<std::complex<double>>& x, Algorithm algorithm);

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x, Algorithm algorithm);
std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x, Algorithm algorithm);

void fft_split_radix_inplace(std::vector<std::complex<double>>& x);
void ifft_split_radix_inplace(std::vector<std::complex<double>>& x);

}  // namespace fft

#endif
