#ifndef FFT_FFT_RADIX2_RECURSIVE_HPP
#define FFT_FFT_RADIX2_RECURSIVE_HPP

#include <complex>
#include <cstddef>
#include <vector>

namespace fft::radix2_recursive {

bool is_power_of_two(std::size_t n);

void fft_inplace(std::vector<std::complex<double>>& x);
void ifft_inplace(std::vector<std::complex<double>>& x);

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x);

}  // namespace fft::radix2_recursive

#endif
