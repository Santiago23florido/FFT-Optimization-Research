#include "fft/fft_dispatch.hpp"

#include "fft/dft_reference.hpp"
#include "fft/fft_mixed_radix_4_2_iterative.hpp"
#include "fft/fft_mixed_radix_4_2_soa.hpp"
#include "fft/fft_radix2_iterative.hpp"
#include "fft/fft_radix2_soa.hpp"
#include "fft/fft_radix2_recursive.hpp"
#include "fft/fft_split_radix.hpp"

#include <stdexcept>
#include <string>

namespace fft {

const char* algorithm_name(Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Radix2Iterative:
            return "radix2_aos";
        case Algorithm::MixedRadix42Iterative:
            return "mixed42_aos";
        case Algorithm::Radix2SoA:
            return "radix2_soa";
        case Algorithm::MixedRadix42SoA:
            return "mixed42_soa";
        case Algorithm::Radix2Recursive:
            return "radix2_recursive";
        case Algorithm::SplitRadix:
            return "split_radix";
        case Algorithm::DirectDft:
            return "direct_dft";
    }
    return "unknown";
}

std::optional<Algorithm> parse_algorithm_name(std::string_view name) {
    if (name == "radix2_aos" || name == "radix2_iterative" || name == "iterative" || name == "radix2") {
        return Algorithm::Radix2Iterative;
    }
    if (name == "mixed42_aos" || name == "mixed_radix_4_2_iterative" || name == "mixed_radix_4_2" ||
        name == "mixed42") {
        return Algorithm::MixedRadix42Iterative;
    }
    if (name == "radix2_soa" || name == "radix2_soa_iterative") {
        return Algorithm::Radix2SoA;
    }
    if (name == "mixed42_soa" || name == "mixed_radix_4_2_soa") {
        return Algorithm::MixedRadix42SoA;
    }
    if (name == "radix2_recursive" || name == "recursive") {
        return Algorithm::Radix2Recursive;
    }
    if (name == "split_radix" || name == "split-radix" || name == "split") {
        return Algorithm::SplitRadix;
    }
    if (name == "direct_dft" || name == "dft" || name == "reference") {
        return Algorithm::DirectDft;
    }
    return std::nullopt;
}

std::vector<Algorithm> supported_algorithms() {
    return {
        Algorithm::Radix2Iterative,
        Algorithm::MixedRadix42Iterative,
        Algorithm::Radix2SoA,
        Algorithm::MixedRadix42SoA,
        Algorithm::Radix2Recursive,
        Algorithm::SplitRadix,
        Algorithm::DirectDft,
    };
}

void fft_inplace(std::vector<std::complex<double>>& x, Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Radix2Iterative:
            radix2_iterative::fft_inplace(x);
            return;
        case Algorithm::MixedRadix42Iterative:
            mixed_radix_4_2_iterative::fft_inplace(x);
            return;
        case Algorithm::Radix2SoA:
            radix2_soa::fft_inplace(x);
            return;
        case Algorithm::MixedRadix42SoA:
            mixed_radix_4_2_soa::fft_inplace(x);
            return;
        case Algorithm::Radix2Recursive:
            radix2_recursive::fft_inplace(x);
            return;
        case Algorithm::SplitRadix:
            split_radix::fft_inplace(x);
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
        case Algorithm::MixedRadix42Iterative:
            mixed_radix_4_2_iterative::ifft_inplace(x);
            return;
        case Algorithm::Radix2SoA:
            radix2_soa::ifft_inplace(x);
            return;
        case Algorithm::MixedRadix42SoA:
            mixed_radix_4_2_soa::ifft_inplace(x);
            return;
        case Algorithm::Radix2Recursive:
            radix2_recursive::ifft_inplace(x);
            return;
        case Algorithm::SplitRadix:
            split_radix::ifft_inplace(x);
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

void fft_inplace_soa(ComplexSoA& x, Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Radix2SoA:
            radix2_soa::fft_inplace(x);
            return;
        case Algorithm::MixedRadix42SoA:
            mixed_radix_4_2_soa::fft_inplace(x);
            return;
        default:
            throw std::invalid_argument("SoA dispatch supports only radix2_soa and mixed42_soa.");
    }
}

void ifft_inplace_soa(ComplexSoA& x, Algorithm algorithm) {
    switch (algorithm) {
        case Algorithm::Radix2SoA:
            radix2_soa::ifft_inplace(x);
            return;
        case Algorithm::MixedRadix42SoA:
            mixed_radix_4_2_soa::ifft_inplace(x);
            return;
        default:
            throw std::invalid_argument("SoA dispatch supports only radix2_soa and mixed42_soa.");
    }
}

void fft_split_radix_inplace(std::vector<std::complex<double>>& x) {
    split_radix::fft_inplace(x);
}

void ifft_split_radix_inplace(std::vector<std::complex<double>>& x) {
    split_radix::ifft_inplace(x);
}

void fft_mixed_radix_4_2_inplace(std::vector<std::complex<double>>& x) {
    mixed_radix_4_2_iterative::fft_inplace(x);
}

void ifft_mixed_radix_4_2_inplace(std::vector<std::complex<double>>& x) {
    mixed_radix_4_2_iterative::ifft_inplace(x);
}

void fft_radix2_soa_inplace(ComplexSoA& x) {
    radix2_soa::fft_inplace(x);
}

void ifft_radix2_soa_inplace(ComplexSoA& x) {
    radix2_soa::ifft_inplace(x);
}

void fft_mixed42_soa_inplace(ComplexSoA& x) {
    mixed_radix_4_2_soa::fft_inplace(x);
}

void ifft_mixed42_soa_inplace(ComplexSoA& x) {
    mixed_radix_4_2_soa::ifft_inplace(x);
}

}  // namespace fft
