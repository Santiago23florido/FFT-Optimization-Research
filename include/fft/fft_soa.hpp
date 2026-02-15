#ifndef FFT_FFT_SOA_HPP
#define FFT_FFT_SOA_HPP

#include <complex>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace fft {

struct ComplexSoA {
    std::vector<double> re;
    std::vector<double> im;

    ComplexSoA() = default;

    explicit ComplexSoA(std::size_t n) : re(n, 0.0), im(n, 0.0) {}

    std::size_t size() const {
        return re.size();
    }

    bool empty() const {
        return re.empty();
    }

    void resize(std::size_t n) {
        re.resize(n);
        im.resize(n);
    }

    static ComplexSoA from_aos(const std::vector<std::complex<double>>& x) {
        ComplexSoA soa(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) {
            soa.re[i] = x[i].real();
            soa.im[i] = x[i].imag();
        }
        return soa;
    }

    std::vector<std::complex<double>> to_aos() const {
        if (re.size() != im.size()) {
            throw std::runtime_error("ComplexSoA real and imaginary arrays must have the same size.");
        }
        std::vector<std::complex<double>> aos(re.size());
        for (std::size_t i = 0; i < re.size(); ++i) {
            aos[i] = std::complex<double>(re[i], im[i]);
        }
        return aos;
    }
};

}  // namespace fft

#endif
