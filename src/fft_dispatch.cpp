#include "fft/fft_dispatch.hpp"

#include "fft/dft_reference.hpp"
#include "fft/fft_radix2_iterative.hpp"
#include "fft/fft_radix2_recursive.hpp"

#include <string>

namespace fft {

const char* algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Radix2Iterative:
            return "radix2_iterative";
        case Algorithm::Radix2Recursive:
            return "radix2_recursive";
        case Algorithm::DirectDft:
            return "direct_dft";
    }
    return "unknown";
}

std::optional<Algorithm> parse_algorithm_name(std::string_view name) {
    if (name == "radix2_iterative" || name == "iterative" || name == "radix2") {
        return Algorithm::Radix2Iterative;
    }
    if (name == "radix2_recursive" || name == "recursive") {
        return Algorithm::Radix2Recursive;
    }
    if (name == "direct_dft" || name == "dft" || name == "reference") {
        return Algorithm::DirectDft;
    }
    return std::nullopt;
}

std::vector<Algorithm> supported_algorithms() {
    return {Algorithm::Radix2Iterative, Algorithm::Radix2Recursive, Algorithm::DirectDft};
}

void fft_inplace(std::vector<std::complex<double>>& x, Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Radix2Iterative:
            radix2_iterative::fft_inplace(x);
            return;
        case Algorithm::Radix2Recursive:
            radix2_recursive::fft_inplace(x);
            return;
        case Algorithm::DirectDft:
            x = dft_reference(x);
            return;
    }
}

void ifft_inplace(std::vector<std::complex<double>>& x, Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Radix2Iterative:
            radix2_iterative::ifft_inplace(x);
            return;
        case Algorithm::Radix2Recursive:
            radix2_recursive::ifft_inplace(x);
            return;
        case Algorithm::DirectDft:
            x = idft_reference(x);
            return;
    }
}

std::vector<std::complex<double>> fft(std::vector<std::complex<double>> x, Algorithm algorithm) {
    fft_inplace(x, algorithm);
    return x;
}

std::vector<std::complex<double>> ifft(std::vector<std::complex<double>> x, Algorithm algorithm) {
    ifft_inplace(x, algorithm);
    return x;
}

}  // namespace fft
