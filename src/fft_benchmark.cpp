#include "fft/fft_dispatch.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

bool is_power_of_two(std::size_t n) {
    return n != 0 && (n & (n - 1U)) == 0;
}

struct Options {
    std::vector<std::size_t> sizes = {64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<fft::Algorithm> algorithms = fft::supported_algorithms();
    std::size_t iterations = 40;
    std::size_t warmup = 5;
    std::uint64_t seed = 1337;
    std::string csv_path = "fft_benchmark_summary.csv";
    std::string raw_csv_path;
};

struct BenchmarkStats {
    double mean_us = 0.0;
    double median_us = 0.0;
    double min_us = 0.0;
    double max_us = 0.0;
    double stddev_us = 0.0;
    double p95_us = 0.0;
    double time_per_sample_ns = 0.0;
    double time_per_nlog2n_ns = 0.0;
    double throughput_samples_per_s = 0.0;
};

std::string trim(std::string value) {
    const auto begin = value.find_first_not_of(" \t\n\r");
    if (begin == std::string::npos) {
        return "";
    }
    const auto end = value.find_last_not_of(" \t\n\r");
    return value.substr(begin, end - begin + 1);
}

std::vector<std::string> split_csv(const std::string& text) {
    std::vector<std::string> tokens;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        token = trim(token);
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

bool parse_size_t(const std::string& text, std::size_t& value) {
    try {
        std::size_t index = 0;
        const unsigned long long parsed = std::stoull(text, &index, 10);
        if (index != text.size()) {
            return false;
        }
        value = static_cast<std::size_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_uint64(const std::string& text, std::uint64_t& value) {
    try {
        std::size_t index = 0;
        const unsigned long long parsed = std::stoull(text, &index, 10);
        if (index != text.size()) {
            return false;
        }
        value = static_cast<std::uint64_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

bool parse_size_list(const std::string& text, std::vector<std::size_t>& sizes, std::string& error) {
    sizes.clear();
    const std::vector<std::string> tokens = split_csv(text);
    if (tokens.empty()) {
        error = "--sizes cannot be empty.";
        return false;
    }

    for (const std::string& token : tokens) {
        std::size_t value = 0;
        if (!parse_size_t(token, value) || value == 0) {
            error = "Invalid size in --sizes: " + token;
            return false;
        }
        sizes.push_back(value);
    }
    return true;
}

bool parse_algorithm_list(const std::string& text, std::vector<fft::Algorithm>& algorithms, std::string& error) {
    algorithms.clear();
    const std::vector<std::string> tokens = split_csv(text);
    if (tokens.empty()) {
        error = "--algorithms cannot be empty.";
        return false;
    }

    for (const std::string& token : tokens) {
        const std::optional<fft::Algorithm> parsed = fft::parse_algorithm_name(token);
        if (!parsed.has_value()) {
            error = "Unknown algorithm in --algorithms: " + token;
            return false;
        }
        algorithms.push_back(*parsed);
    }
    return true;
}

void print_help(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "\n"
              << "Options:\n"
              << "  --sizes <csv>        Signal lengths, e.g. 64,128,256,512\n"
              << "  --algorithms <csv>   radix2_iterative,mixed_radix_4_2_iterative,radix2_recursive,split_radix,direct_dft\n"
              << "  --iterations <n>     Number of measured iterations per size/algorithm\n"
              << "  --warmup <n>         Number of warmup iterations per size/algorithm\n"
              << "  --seed <n>           Base random seed for generated input vectors\n"
              << "  --csv <file>         Summary CSV output path\n"
              << "  --raw_csv <file>     Optional raw per-iteration timings CSV path\n"
              << "  --help               Show this help message\n";
}

int parse_args(int argc, char** argv, Options& options, std::string& error) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            return 1;
        }
        if (arg == "--sizes") {
            if (i + 1 >= argc) {
                error = "Missing value after --sizes.";
                return -1;
            }
            if (!parse_size_list(argv[++i], options.sizes, error)) {
                return -1;
            }
            continue;
        }
        if (arg == "--algorithms") {
            if (i + 1 >= argc) {
                error = "Missing value after --algorithms.";
                return -1;
            }
            if (!parse_algorithm_list(argv[++i], options.algorithms, error)) {
                return -1;
            }
            continue;
        }
        if (arg == "--iterations") {
            if (i + 1 >= argc) {
                error = "Missing value after --iterations.";
                return -1;
            }
            if (!parse_size_t(argv[++i], options.iterations) || options.iterations == 0) {
                error = "--iterations must be a positive integer.";
                return -1;
            }
            continue;
        }
        if (arg == "--warmup") {
            if (i + 1 >= argc) {
                error = "Missing value after --warmup.";
                return -1;
            }
            if (!parse_size_t(argv[++i], options.warmup)) {
                error = "--warmup must be a non-negative integer.";
                return -1;
            }
            continue;
        }
        if (arg == "--seed") {
            if (i + 1 >= argc) {
                error = "Missing value after --seed.";
                return -1;
            }
            if (!parse_uint64(argv[++i], options.seed)) {
                error = "--seed must be a non-negative integer.";
                return -1;
            }
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
        if (arg == "--raw_csv") {
            if (i + 1 >= argc) {
                error = "Missing value after --raw_csv.";
                return -1;
            }
            options.raw_csv_path = argv[++i];
            continue;
        }

        error = "Unknown argument: " + arg;
        return -1;
    }

    if (options.sizes.empty()) {
        error = "At least one size is required.";
        return -1;
    }
    if (options.algorithms.empty()) {
        error = "At least one algorithm is required.";
        return -1;
    }

    return 0;
}

std::vector<std::complex<double>> generate_signal(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    std::vector<std::complex<double>> signal(n);
    for (std::complex<double>& value : signal) {
        value = std::complex<double>(dist(rng), dist(rng));
    }
    return signal;
}

BenchmarkStats compute_stats(const std::vector<double>& durations_ns, std::size_t n) {
    BenchmarkStats stats;
    if (durations_ns.empty()) {
        return stats;
    }

    std::vector<double> sorted = durations_ns;
    std::sort(sorted.begin(), sorted.end());

    const double mean_ns = std::accumulate(sorted.begin(), sorted.end(), 0.0) / static_cast<double>(sorted.size());
    const double min_ns = sorted.front();
    const double max_ns = sorted.back();

    const std::size_t middle = sorted.size() / 2;
    const double median_ns =
        (sorted.size() % 2 == 0) ? 0.5 * (sorted[middle - 1] + sorted[middle]) : sorted[middle];

    const std::size_t p95_index =
        static_cast<std::size_t>(std::ceil(0.95 * static_cast<double>(sorted.size()))) - 1;
    const double p95_ns = sorted[std::min(p95_index, sorted.size() - 1)];

    double variance = 0.0;
    for (const double value : sorted) {
        const double delta = value - mean_ns;
        variance += delta * delta;
    }
    variance /= static_cast<double>(sorted.size());

    stats.mean_us = mean_ns / 1000.0;
    stats.median_us = median_ns / 1000.0;
    stats.min_us = min_ns / 1000.0;
    stats.max_us = max_ns / 1000.0;
    stats.stddev_us = std::sqrt(variance) / 1000.0;
    stats.p95_us = p95_ns / 1000.0;
    stats.time_per_sample_ns = mean_ns / static_cast<double>(n);

    const double nlog2n = static_cast<double>(n) * std::log2(std::max<std::size_t>(n, 2));
    stats.time_per_nlog2n_ns = mean_ns / nlog2n;

    const double mean_seconds = mean_ns * 1e-9;
    stats.throughput_samples_per_s = static_cast<double>(n) / mean_seconds;

    return stats;
}

void validate_size_for_algorithm(std::size_t n, fft::Algorithm algorithm) {
    if (algorithm != fft::Algorithm::DirectDft && !is_power_of_two(n)) {
        throw std::invalid_argument(std::string("Algorithm ") + fft::algorithm_name(algorithm) +
                                    " requires power-of-two sizes. Invalid N=" + std::to_string(n));
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

        std::ofstream summary_csv(options.csv_path);
        if (!summary_csv) {
            throw std::runtime_error("Unable to open summary CSV file: " + options.csv_path);
        }

        std::ofstream raw_csv;
        if (!options.raw_csv_path.empty()) {
            raw_csv.open(options.raw_csv_path);
            if (!raw_csv) {
                throw std::runtime_error("Unable to open raw CSV file: " + options.raw_csv_path);
            }
            raw_csv << "algorithm,size,iteration,elapsed_us\n";
            raw_csv << std::setprecision(17);
        }

        summary_csv << "algorithm,size,iterations,warmup,mean_us,median_us,min_us,max_us,stddev_us,p95_us,"
                       "time_per_sample_ns,time_per_nlog2n_ns,throughput_samples_per_s,checksum\n";
        summary_csv << std::setprecision(17);

        std::cout << "Running FFT benchmark with " << options.algorithms.size() << " algorithm(s) and "
                  << options.sizes.size() << " size(s).\n";

        for (const fft::Algorithm algorithm : options.algorithms) {
            for (const std::size_t n : options.sizes) {
                validate_size_for_algorithm(n, algorithm);

                const std::uint64_t signal_seed = options.seed ^ (static_cast<std::uint64_t>(n) * 0x9E3779B97F4A7C15ULL);
                const std::vector<std::complex<double>> base_signal = generate_signal(n, signal_seed);

                for (std::size_t i = 0; i < options.warmup; ++i) {
                    auto warm = base_signal;
                    fft::fft_inplace(warm, algorithm);
                }

                std::vector<double> durations_ns;
                durations_ns.reserve(options.iterations);
                double checksum = 0.0;

                for (std::size_t iter = 0; iter < options.iterations; ++iter) {
                    auto work = base_signal;

                    const auto start = std::chrono::steady_clock::now();
                    fft::fft_inplace(work, algorithm);
                    const auto end = std::chrono::steady_clock::now();

                    const double elapsed_ns =
                        static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
                    durations_ns.push_back(elapsed_ns);

                    checksum += std::abs(work[0]);

                    if (raw_csv) {
                        raw_csv << fft::algorithm_name(algorithm) << ',' << n << ',' << iter << ','
                                << (elapsed_ns / 1000.0) << '\n';
                    }
                }

                const BenchmarkStats stats = compute_stats(durations_ns, n);

                summary_csv << fft::algorithm_name(algorithm) << ',' << n << ',' << options.iterations << ','
                            << options.warmup << ',' << stats.mean_us << ',' << stats.median_us << ','
                            << stats.min_us << ',' << stats.max_us << ',' << stats.stddev_us << ',' << stats.p95_us
                            << ',' << stats.time_per_sample_ns << ',' << stats.time_per_nlog2n_ns << ','
                            << stats.throughput_samples_per_s << ',' << checksum << '\n';

                std::cout << "algorithm=" << fft::algorithm_name(algorithm) << " N=" << n
                          << " mean_us=" << std::fixed << std::setprecision(3) << stats.mean_us
                          << " p95_us=" << stats.p95_us
                          << " throughput(samples/s)=" << std::setprecision(2) << stats.throughput_samples_per_s
                          << '\n';
            }
        }

        std::cout << "Summary CSV: " << options.csv_path << '\n';
        if (!options.raw_csv_path.empty()) {
            std::cout << "Raw CSV: " << options.raw_csv_path << '\n';
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
