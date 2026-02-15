#ifndef FFT_FFT_RADIX2_ITERATIVE_HPP
#define FFT_FFT_RADIX2_ITERATIVE_HPP

#include <complex>
#include <cstddef>
#include <vector>

namespace fft::radix2_iterative {

bool is_power_of_two(std::size_t n);

void fft_inplace(std::vector<std::complex<double>>& x);
void ifft_inplace(std::vector<std::complex<double>>& x);

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x);
std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x);

}  // namespace fft::radix2_iterative

#endif
