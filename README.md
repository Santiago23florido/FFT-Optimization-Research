# Radix-2 FFT in C++20

This project implements a dependency-free radix-2 Cooley-Tukey FFT with:

- In-place iterative FFT and IFFT (`1/N` normalization in IFFT)
- Reference \(O(N^2)\) DFT/IDFT for validation
- CLI demo (`fft_demo`) for tone generation and spectrum inspection
- Automated tests (`test_fft`) integrated with CTest

## Build

```bash
cmake -S . -B build
cmake --build build --config Release
```

## Run Tests

```bash
ctest --test-dir build -C Release
```

## Run Demo

Complex tone:

```bash
./build/fft_demo --N 1024 --tone 37 --complex_tone
```

Real sine with CSV output:

```bash
./build/fft_demo --N 1024 --tone 37 --real_sine --csv spectrum.csv
```

On Windows with multi-config generators, use `build/Release/fft_demo.exe`.

## Documentation

- `docs/fft_ieee_paper.tex`: IEEE-format paper root file
- `docs/dsections/`: section files included by the IEEE paper (`abstract`, `introduction`, `mathematical_foundations`, `algorithm_and_implementation`, `verification`, `applications`, `conclusion`)
# FFT-Optimization-Research
