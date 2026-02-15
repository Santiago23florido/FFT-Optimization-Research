# FFT Optimization Research

C++20 framework for developing and comparing FFT algorithms with a correctness-first baseline.

## Included algorithms

- `radix2_iterative`: iterative in-place Cooley-Tukey with bit-reversal permutation
- `radix2_recursive`: recursive radix-2 Cooley-Tukey
- `split_radix`: split-radix FFT with recursive decomposition
- `direct_dft`: exact \(O(N^2)\) DFT reference

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
```

## Run tests

```bash
ctest --test-dir build -C Release --output-on-failure
```

## Demo

```bash
./build/fft_demo --N 1024 --tone 37 --algorithm radix2_iterative --complex_tone
./build/fft_demo --N 1024 --tone 37 --algorithm radix2_recursive --real_sine
./build/fft_demo --N 1024 --tone 37 --algorithm split_radix --complex_tone
./build/fft_demo --N 1000 --tone 37 --algorithm direct_dft --complex_tone
```

## C++ benchmark (CSV output)

```bash
./build/fft_benchmark \
  --sizes 64,128,256,512,1024,2048,4096 \
  --algorithms radix2_iterative,radix2_recursive,split_radix,direct_dft \
  --iterations 40 \
  --warmup 5 \
  --csv fft_benchmark_summary.csv \
  --raw_csv fft_benchmark_raw.csv
```

The benchmark exports multiple comparable timing metrics per algorithm and size:

- `mean_us`, `median_us`, `min_us`, `max_us`, `stddev_us`, `p95_us`
- `time_per_sample_ns`, `time_per_nlog2n_ns`, `throughput_samples_per_s`

## Automated benchmarking + plotting (Python)

```bash
python3 scripts/fft_benchmark_plot.py \
  --build-dir build \
  --config Release \
  --sizes 64,128,256,512,1024,2048,4096 \
  --algorithms radix2_iterative,radix2_recursive,split_radix,direct_dft \
  --iterations 40 \
  --warmup 5 \
  --output-dir benchmark_results
```

By default, the script synchronizes the same generated files to both `benchmark_results` and `docs/figures` so report and local benchmark artifacts stay identical.

Generated artifacts:

- `benchmark_results/fft_benchmark_summary.csv`
- `benchmark_results/fft_benchmark_raw.csv`
- `benchmark_results/mean_runtime.png`
- `benchmark_results/mean_runtime_us.png`
- `benchmark_results/p95_runtime_us.png`
- `benchmark_results/time_per_sample_ns.png`
- `benchmark_results/throughput.png`
- `benchmark_results/throughput_samples_per_s.png`

`matplotlib` is required for plots.

## Documentation

- `docs/fft_ieee_paper.tex`: IEEE-format paper root file
- `docs/dsections/`: section files used by the IEEE paper
