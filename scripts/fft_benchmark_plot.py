#!/usr/bin/env python3
"""Run C++ FFT benchmarks and generate comparison plots."""

from __future__ import annotations

import argparse
import csv
import shlex
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ALGORITHM_LABELS: dict[str, str] = {
    "radix2_aos": "Radix-2 AoS",
    "mixed42_aos": "Mixed Radix (4/2) AoS",
    "radix2_soa": "Radix-2 SoA",
    "mixed42_soa": "Mixed Radix (4/2) SoA",
    "radix2_recursive": "Radix-2 Recursive",
    "split_radix": "Split-Radix",
    "direct_dft": "Direct DFT",
}

ALGORITHM_ORDER: dict[str, int] = {
    "radix2_aos": 0,
    "mixed42_aos": 1,
    "radix2_recursive": 2,
    "split_radix": 3,
    "direct_dft": 4,
    "radix2_soa": 5,
    "mixed42_soa": 6,
}

MAIN_DEFAULT_ALGORITHMS = "radix2_aos,mixed42_aos,radix2_recursive,split_radix,direct_dft"
LAYOUT_DEFAULT_ALGORITHMS = "radix2_aos,radix2_soa,mixed42_aos,mixed42_soa"


def run_command(command: list[str], cwd: Path | None = None) -> None:
    printable = " ".join(shlex.quote(part) for part in command)
    print(f"$ {printable}", flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def parse_csv_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def find_benchmark_binary(build_dir: Path, config: str) -> Path:
    candidates = [
        build_dir / "fft_benchmark",
        build_dir / "fft_benchmark.exe",
        build_dir / config / "fft_benchmark",
        build_dir / config / "fft_benchmark.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate fft_benchmark executable. Build the project first or verify build/config paths."
    )


def load_summary_rows(csv_path: Path) -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            parsed: dict[str, float | str] = {"algorithm": row["algorithm"]}
            for key in (
                "size",
                "iterations",
                "warmup",
                "mean_us",
                "median_us",
                "min_us",
                "max_us",
                "stddev_us",
                "p95_us",
                "time_per_sample_ns",
                "time_per_nlog2n_ns",
                "throughput_samples_per_s",
                "checksum",
            ):
                parsed[key] = float(row[key])
            rows.append(parsed)
    return rows


def filter_rows(rows: list[dict[str, float | str]], algorithms: set[str]) -> list[dict[str, float | str]]:
    return [row for row in rows if str(row["algorithm"]) in algorithms]


def plot_metric(
    rows: list[dict[str, float | str]],
    metric_key: str,
    ylabel: str,
    title: str,
    output_path: Path,
    y_log: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate plots. Install it with 'pip install matplotlib'."
        ) from exc

    grouped: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        algorithm = str(row["algorithm"])
        size = int(row["size"])
        value = float(row[metric_key])
        grouped[algorithm].append((size, value))

    plt.figure(figsize=(10, 6))
    for algorithm, points in sorted(grouped.items(), key=lambda item: ALGORITHM_ORDER.get(item[0], 99)):
        points.sort(key=lambda item: item[0])
        xs = [size for size, _ in points]
        ys = [value for _, value in points]
        plt.plot(xs, ys, marker="o", linewidth=2, label=ALGORITHM_LABELS.get(algorithm, algorithm))

    plt.xlabel("Signal length N")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xscale("log", base=2)
    if y_log:
        plt.yscale("log")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_speedup(
    rows: list[dict[str, float | str]],
    metric_key: str,
    baseline_algorithm: str,
    title: str,
    output_path: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to generate plots. Install it with 'pip install matplotlib'."
        ) from exc

    grouped: dict[str, dict[int, float]] = defaultdict(dict)
    for row in rows:
        algorithm = str(row["algorithm"])
        size = int(row["size"])
        value = float(row[metric_key])
        grouped[algorithm][size] = value

    if baseline_algorithm not in grouped:
        raise RuntimeError(f"Speedup baseline '{baseline_algorithm}' not present in benchmark data.")

    baseline_map = grouped[baseline_algorithm]
    if not baseline_map:
        raise RuntimeError(f"Speedup baseline '{baseline_algorithm}' has no data points.")

    plt.figure(figsize=(10, 6))
    for algorithm, values_by_size in sorted(grouped.items(), key=lambda item: ALGORITHM_ORDER.get(item[0], 99)):
        common_sizes = sorted(set(values_by_size.keys()) & set(baseline_map.keys()))
        if not common_sizes:
            continue
        speedups = [baseline_map[size] / values_by_size[size] for size in common_sizes]
        plt.plot(
            common_sizes,
            speedups,
            marker="o",
            linewidth=2,
            label=ALGORITHM_LABELS.get(algorithm, algorithm),
        )

    baseline_label = ALGORITHM_LABELS.get(baseline_algorithm, baseline_algorithm)
    plt.axhline(1.0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    plt.xlabel("Signal length N")
    plt.ylabel(f"Speedup vs {baseline_label}")
    plt.title(title)
    plt.xscale("log", base=2)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_benchmark(
    benchmark_bin: Path,
    sizes: str,
    algorithms: str,
    iterations: int,
    warmup: int,
    seed: int,
    summary_csv: Path,
    raw_csv: Path,
) -> list[dict[str, float | str]]:
    run_command(
        [
            str(benchmark_bin),
            "--sizes",
            sizes,
            "--algorithms",
            algorithms,
            "--iterations",
            str(iterations),
            "--warmup",
            str(warmup),
            "--seed",
            str(seed),
            "--csv",
            str(summary_csv),
            "--raw_csv",
            str(raw_csv),
        ]
    )

    rows = load_summary_rows(summary_csv)
    if not rows:
        raise RuntimeError(f"No rows found in summary CSV: {summary_csv}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build, run FFT benchmark executable, and generate timing comparison plots."
    )
    parser.add_argument("--project-root", default=".", help="Project root directory.")
    parser.add_argument("--build-dir", default="build", help="CMake build directory.")
    parser.add_argument("--config", default="Release", help="Build configuration (Release/Debug).")
    parser.add_argument(
        "--sizes",
        default="64,128,256,512,1024,2048,4096",
        help="Comma-separated signal sizes.",
    )
    parser.add_argument(
        "--algorithms",
        default=MAIN_DEFAULT_ALGORITHMS,
        help="Comma-separated algorithm list for the main benchmark study.",
    )
    parser.add_argument(
        "--layout-algorithms",
        default=LAYOUT_DEFAULT_ALGORITHMS,
        help="Comma-separated algorithm list for the AoS/SoA layout study.",
    )
    parser.add_argument("--iterations", type=int, default=40, help="Measured iterations per case.")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup iterations per case.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for CSV files and figures.",
    )
    parser.add_argument(
        "--layout-subdir",
        default="layout_optimization",
        help="Subdirectory used for AoS/SoA layout-study artifacts.",
    )
    parser.add_argument(
        "--sync-dirs",
        default="benchmark_results,docs/figures",
        help="Comma-separated directories that will receive identical copies of generated outputs.",
    )
    parser.add_argument("--skip-build", action="store_true", help="Skip CMake configure/build step.")
    parser.add_argument(
        "--skip-layout-study",
        action="store_true",
        help="Skip the secondary AoS/SoA layout-study benchmark and plots.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    build_dir = (project_root / args.build_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        run_command(
            [
                "cmake",
                "-S",
                str(project_root),
                "-B",
                str(build_dir),
                f"-DCMAKE_BUILD_TYPE={args.config}",
            ]
        )
        run_command(["cmake", "--build", str(build_dir), "--config", args.config, "--target", "fft_benchmark"])

    benchmark_bin = find_benchmark_binary(build_dir, args.config)

    main_summary_csv = output_dir / "fft_benchmark_summary.csv"
    main_raw_csv = output_dir / "fft_benchmark_raw.csv"
    main_rows = run_benchmark(
        benchmark_bin,
        sizes=args.sizes,
        algorithms=args.algorithms,
        iterations=args.iterations,
        warmup=args.warmup,
        seed=args.seed,
        summary_csv=main_summary_csv,
        raw_csv=main_raw_csv,
    )

    plot_metric(
        main_rows,
        metric_key="mean_us",
        ylabel="Mean execution time (microseconds)",
        title="FFT Benchmark: Mean Runtime",
        output_path=output_dir / "mean_runtime.png",
        y_log=True,
    )
    plot_metric(
        main_rows,
        metric_key="mean_us",
        ylabel="Mean execution time (microseconds)",
        title="FFT Benchmark: Mean Runtime",
        output_path=output_dir / "mean_runtime_us.png",
        y_log=True,
    )
    plot_metric(
        main_rows,
        metric_key="p95_us",
        ylabel="P95 execution time (microseconds)",
        title="FFT Benchmark: P95 Runtime",
        output_path=output_dir / "p95_runtime_us.png",
        y_log=True,
    )
    plot_metric(
        main_rows,
        metric_key="time_per_sample_ns",
        ylabel="Time per sample (nanoseconds)",
        title="FFT Benchmark: Time per Sample",
        output_path=output_dir / "time_per_sample_ns.png",
        y_log=True,
    )
    plot_metric(
        main_rows,
        metric_key="throughput_samples_per_s",
        ylabel="Throughput (samples per second)",
        title="FFT Benchmark: Throughput",
        output_path=output_dir / "throughput.png",
        y_log=True,
    )
    plot_metric(
        main_rows,
        metric_key="throughput_samples_per_s",
        ylabel="Throughput (samples per second)",
        title="FFT Benchmark: Throughput",
        output_path=output_dir / "throughput_samples_per_s.png",
        y_log=True,
    )

    main_generated_files = [
        main_summary_csv,
        main_raw_csv,
        output_dir / "mean_runtime.png",
        output_dir / "mean_runtime_us.png",
        output_dir / "p95_runtime_us.png",
        output_dir / "time_per_sample_ns.png",
        output_dir / "throughput.png",
        output_dir / "throughput_samples_per_s.png",
    ]

    layout_generated_files: list[Path] = []
    layout_output_dir = output_dir / args.layout_subdir
    if not args.skip_layout_study:
        layout_output_dir.mkdir(parents=True, exist_ok=True)

        layout_summary_csv = layout_output_dir / "fft_layout_summary.csv"
        layout_raw_csv = layout_output_dir / "fft_layout_raw.csv"
        layout_rows = run_benchmark(
            benchmark_bin,
            sizes=args.sizes,
            algorithms=args.layout_algorithms,
            iterations=args.iterations,
            warmup=args.warmup,
            seed=args.seed,
            summary_csv=layout_summary_csv,
            raw_csv=layout_raw_csv,
        )

        plot_metric(
            filter_rows(layout_rows, {"radix2_aos", "radix2_soa"}),
            metric_key="mean_us",
            ylabel="Mean execution time (microseconds)",
            title="Layout Study: Radix-2 AoS vs SoA",
            output_path=layout_output_dir / "radix2_aos_vs_soa_runtime.png",
            y_log=True,
        )
        plot_metric(
            filter_rows(layout_rows, {"mixed42_aos", "mixed42_soa"}),
            metric_key="mean_us",
            ylabel="Mean execution time (microseconds)",
            title="Layout Study: Mixed Radix (4/2) AoS vs SoA",
            output_path=layout_output_dir / "mixed42_aos_vs_soa_runtime.png",
            y_log=True,
        )
        plot_speedup(
            layout_rows,
            metric_key="mean_us",
            baseline_algorithm="radix2_aos",
            title="Layout Study: Speedup vs Radix-2 AoS",
            output_path=layout_output_dir / "speedup_vs_radix2_aos.png",
        )

        layout_generated_files = [
            layout_summary_csv,
            layout_raw_csv,
            layout_output_dir / "radix2_aos_vs_soa_runtime.png",
            layout_output_dir / "mixed42_aos_vs_soa_runtime.png",
            layout_output_dir / "speedup_vs_radix2_aos.png",
        ]

    synced_dirs: list[Path] = []
    for sync_dir in parse_csv_list(args.sync_dirs):
        destination_dir = (project_root / sync_dir).resolve()
        if destination_dir == output_dir:
            continue

        destination_dir.mkdir(parents=True, exist_ok=True)
        for source_file in main_generated_files:
            shutil.copy2(source_file, destination_dir / source_file.name)

        if layout_generated_files:
            destination_layout_dir = destination_dir / args.layout_subdir
            destination_layout_dir.mkdir(parents=True, exist_ok=True)
            for source_file in layout_generated_files:
                shutil.copy2(source_file, destination_layout_dir / source_file.name)

        synced_dirs.append(destination_dir)

    print("Generated main-study files:")
    for file_path in main_generated_files:
        print(f"- {file_path}")

    if layout_generated_files:
        print("Generated layout-study files:")
        for file_path in layout_generated_files:
            print(f"- {file_path}")

    if synced_dirs:
        print("Synchronized directories:")
        for directory in synced_dirs:
            print(f"- {directory}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pylint: disable=broad-except
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1)
