#ifndef FFT_DFT_REFERENCE_HPP
#define FFT_DFT_REFERENCE_HPP

#include <complex>
#include <vector>

namespace fft {

std::vector<std::complex<double>> dft_reference(const std::vector<std::complex<double>>& x);
std::vector<std::complex<double>> idft_reference(const std::vector<std::complex<double>>& X);

}  // namespace fft

#endif
