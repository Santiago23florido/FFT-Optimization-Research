#include "fft/fft_dispatch.hpp"
#include "fft/fft_radix2_iterative.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Options {
    std::size_t n = 16;
    std::size_t tone = 1;
    fft::Algorithm algorithm = fft::Algorithm::Radix2Iterative;
    bool real_sine = false;
    bool complex_tone = false;
    std::string csv_path;
};

void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name
              << " --N <length> --tone <k> [--algorithm <name>] [--real_sine | --complex_tone] "
                 "[--csv <file>]\n"
              << "\n"
              << "Options:\n"
              << "  --N <length>         Signal length (required).\n"
              << "  --tone <k>           Tone index k (required, wrapped modulo N).\n"
              << "  --algorithm <name>   FFT model: radix2_iterative, radix2_recursive, direct_dft.\n"
              << "  --real_sine          Generate x[n] = sin(2*pi*k*n/N).\n"
              << "  --complex_tone       Generate x[n] = exp(j*2*pi*k*n/N).\n"
              << "  --csv <file>         Write spectrum magnitudes as CSV (k,magnitude).\n"
              << "  --help               Show this help message.\n";
}

bool parse_size_t(const std::string& text, std::size_t& out) {
    try {
        std::size_t pos = 0;
        const unsigned long long value = std::stoull(text, &pos, 10);
        if (pos != text.size()) {
            return false;
        }
        if (value > std::numeric_limits<std::size_t>::max()) {
            return false;
        }
        out = static_cast<std::size_t>(value);
        return true;
    } catch (...) {
        return false;
    }
}

int parse_args(int argc, char** argv, Options& options, std::string& error) {
    bool has_n = false;
    bool has_tone = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            return 1;
        }
        if (arg == "--N") {
            if (i + 1 >= argc) {
                error = "Missing value after --N.";
                return -1;
            }
            if (!parse_size_t(argv[++i], options.n)) {
                error = "Invalid value for --N; expected a non-negative integer.";
                return -1;
            }
            has_n = true;
            continue;
        }
        if (arg == "--tone") {
            if (i + 1 >= argc) {
                error = "Missing value after --tone.";
                return -1;
            }
            if (!parse_size_t(argv[++i], options.tone)) {
                error = "Invalid value for --tone; expected a non-negative integer.";
                return -1;
            }
            has_tone = true;
            continue;
        }
        if (arg == "--algorithm") {
            if (i + 1 >= argc) {
                error = "Missing value after --algorithm.";
                return -1;
            }
            const std::string name = argv[++i];
            const std::optional<fft::Algorithm> parsed = fft::parse_algorithm_name(name);
            if (!parsed.has_value()) {
                error = "Invalid value for --algorithm. Use radix2_iterative, radix2_recursive, or direct_dft.";
                return -1;
            }
            options.algorithm = *parsed;
            continue;
        }
        if (arg == "--real_sine") {
            options.real_sine = true;
            continue;
        }
        if (arg == "--complex_tone") {
            options.complex_tone = true;
            continue;
        }
        if (arg == "--csv") {
            if (i + 1 >= argc) {
                error = "Missing value after --csv.";
                return -1;
            }
            options.csv_path = argv[++i];
            continue;
        }

        error = "Unknown argument: " + arg;
        return -1;
    }

    if (!has_n) {
        error = "Missing required argument --N.";
        return -1;
    }
    if (!has_tone) {
        error = "Missing required argument --tone.";
        return -1;
    }
    if (options.n == 0) {
        error = "--N must be greater than zero.";
        return -1;
    }
    if (options.algorithm != fft::Algorithm::DirectDft && !fft::radix2_iterative::is_power_of_two(options.n)) {
        error = "Selected radix-2 algorithm requires --N to be a non-zero power of two.";
        return -1;
    }
    if (options.real_sine && options.complex_tone) {
        error = "Choose only one signal type: --real_sine or --complex_tone.";
        return -1;
    }
    if (!options.real_sine && !options.complex_tone) {
        options.complex_tone = true;
    }

    return 0;
}

std::vector<std::complex<double>> generate_signal(const Options& options, std::size_t tone_mod_n) {
    std::vector<std::complex<double>> x(options.n);
    for (std::size_t n = 0; n < options.n; ++n) {
        const double angle = 2.0 * std::numbers::pi_v<double> * static_cast<double>(tone_mod_n) *
                             static_cast<double>(n) / static_cast<double>(options.n);
        if (options.real_sine) {
            x[n] = std::complex<double>(std::sin(angle), 0.0);
        } else {
            x[n] = std::complex<double>(std::cos(angle), std::sin(angle));
        }
    }
    return x;
}

std::vector<std::pair<std::size_t, double>> top_bins(const std::vector<std::complex<double>>& spectrum,
                                                      std::size_t top_count) {
    std::vector<std::pair<std::size_t, double>> bins;
    bins.reserve(spectrum.size());
    for (std::size_t k = 0; k < spectrum.size(); ++k) {
        bins.emplace_back(k, std::abs(spectrum[k]));
    }

    const std::size_t count = std::min(top_count, bins.size());
    std::partial_sort(bins.begin(), bins.begin() + static_cast<std::ptrdiff_t>(count), bins.end(),
                      [](const auto& lhs, const auto& rhs) {
                          if (lhs.second == rhs.second) {
                              return lhs.first < rhs.first;
                          }
                          return lhs.second > rhs.second;
                      });
    bins.resize(count);
    return bins;
}

void write_csv(const std::string& path, const std::vector<std::complex<double>>& spectrum) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Unable to open CSV output file: " + path);
    }

    out << "k,magnitude\n";
    out << std::setprecision(17);
    for (std::size_t k = 0; k < spectrum.size(); ++k) {
        out << k << ',' << std::abs(spectrum[k]) << '\n';
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options;
        std::string parse_error;
        const int parse_result = parse_args(argc, argv, options, parse_error);

        if (parse_result == 1) {
            print_help(argv[0]);
            return 0;
        }
        if (parse_result != 0) {
            std::cerr << "Error: " << parse_error << "\n\n";
            print_help(argv[0]);
            return 1;
        }

        const std::size_t tone_mod_n = options.tone % options.n;
        const std::vector<std::complex<double>> signal = generate_signal(options, tone_mod_n);
        const std::vector<std::complex<double>> spectrum = fft::fft(signal, options.algorithm);

        if (!options.csv_path.empty()) {
            write_csv(options.csv_path, spectrum);
        }

        const auto strongest = top_bins(spectrum, 5);

        std::cout << std::fixed << std::setprecision(6);
        std::cout << "N: " << options.n << '\n';
        std::cout << "Tone index k: " << tone_mod_n << '\n';
        std::cout << "Algorithm: " << fft::algorithm_name(options.algorithm) << '\n';
        std::cout << "Signal type: " << (options.real_sine ? "real_sine" : "complex_tone") << '\n';
        if (options.tone != tone_mod_n) {
            std::cout << "Input tone wrapped modulo N: " << options.tone << " -> " << tone_mod_n << '\n';
        }
        std::cout << "Top 5 bins by magnitude:\n";
        for (const auto& [k, magnitude] : strongest) {
            std::cout << "  bin " << k << ": " << magnitude << '\n';
        }

        if (!options.csv_path.empty()) {
            std::cout << "CSV written to: " << options.csv_path << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
