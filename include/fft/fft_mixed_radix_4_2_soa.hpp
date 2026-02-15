#ifndef FFT_FFT_MIXED_RADIX_4_2_SOA_HPP
#define FFT_FFT_MIXED_RADIX_4_2_SOA_HPP

#include "fft/fft_soa.hpp"

#include <complex>
#include <cstddef>
#include <vector>

namespace fft::mixed_radix_4_2_soa {

bool is_power_of_two(std::size_t n);

void fft_inplace(ComplexSoA& x);
void ifft_inplace(ComplexSoA& x);

void fft_inplace(std::vector<std::complex<double>>& x);
void ifft_inplace(std::vector<std::complex<double>>& x);

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x);

}  // namespace fft::mixed_radix_4_2_soa

#endif
