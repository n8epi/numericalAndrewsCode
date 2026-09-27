#!/usr/bin/env python3
"""Benchmark the numerical core used to construct the MMSSQV plots.

Dataset loading, file output, raster rendering, and Feature Congestion are
excluded.  Each phase is timed independently, so the independently measured
phase medians need not sum exactly to the end-to-end median.
"""

from __future__ import annotations

from numerical_andrews.paths import RESULTS
import argparse
import csv
import gc
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

# Request a single numerical-library thread before importing NumPy/SciPy.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(variable, "1")

import numpy as np
import scipy
import sklearn

from numerical_andrews.core import (
    evaluate_spatial_spectral_modes,
    load_example_datasets,
    pca_scores,
    spatial_spectral_modes,
)


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_OUTPUT = RESULTS / "runtime_benchmark.csv"
DEFAULT_METADATA = (
    RESULTS / "runtime_benchmark_metadata.json"
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--cutoff", type=int, default=256)
    parser.add_argument("--grid-points", type=int, default=4097)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    arguments = parser.parse_args()
    if arguments.alpha <= 0:
        parser.error("--alpha must be positive")
    if arguments.cutoff < 1:
        parser.error("--cutoff must be at least 1")
    if arguments.grid_points < 2:
        parser.error("--grid-points must be at least 2")
    if arguments.warmups < 0:
        parser.error("--warmups must be nonnegative")
    if arguments.repetitions < 1:
        parser.error("--repetitions must be positive")
    return arguments


def measure(
    operation: Callable[[], object], warmups: int, repetitions: int
) -> dict[str, float]:
    for _ in range(warmups):
        operation()

    timings_ms: list[float] = []
    for _ in range(repetitions):
        gc.collect()
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            start = time.perf_counter_ns()
            operation()
            elapsed = time.perf_counter_ns() - start
        finally:
            if was_enabled:
                gc.enable()
        timings_ms.append(elapsed / 1_000_000.0)

    quartiles = np.percentile(timings_ms, [25.0, 50.0, 75.0])
    return {
        "q1_ms": float(quartiles[0]),
        "median_ms": float(quartiles[1]),
        "q3_ms": float(quartiles[2]),
    }


def benchmark_dataset(dataset, arguments: argparse.Namespace) -> dict[str, object]:
    values = np.asarray(dataset.values, dtype=float)
    dimension = values.shape[1]
    points = np.linspace(-0.5, 0.5, arguments.grid_points)

    modes, _ = spatial_spectral_modes(arguments.alpha, arguments.cutoff, dimension)
    basis = evaluate_spatial_spectral_modes(modes, points)
    _, scores, _ = pca_scores(values)

    def pca_phase() -> np.ndarray:
        return pca_scores(values)[1]

    def eigensolve_basis_phase() -> np.ndarray:
        selected_modes, _ = spatial_spectral_modes(
            arguments.alpha, arguments.cutoff, dimension
        )
        return evaluate_spatial_spectral_modes(selected_modes, points)

    def curve_phase() -> np.ndarray:
        return scores @ basis.T

    def combined_phase() -> np.ndarray:
        _, current_scores, _ = pca_scores(values)
        selected_modes, _ = spatial_spectral_modes(
            arguments.alpha, arguments.cutoff, dimension
        )
        current_basis = evaluate_spatial_spectral_modes(selected_modes, points)
        return current_scores @ current_basis.T

    phases = {
        "pca": measure(pca_phase, arguments.warmups, arguments.repetitions),
        "eigensolve_basis": measure(
            eigensolve_basis_phase, arguments.warmups, arguments.repetitions
        ),
        "curve_formation": measure(
            curve_phase, arguments.warmups, arguments.repetitions
        ),
        "combined": measure(
            combined_phase, arguments.warmups, arguments.repetitions
        ),
    }

    row: dict[str, object] = {
        "dataset": dataset.display_name,
        "observations": values.shape[0],
        "dimension": dimension,
        "even_eigenpairs": (dimension + 1) // 2,
        "odd_eigenpairs": dimension // 2,
        "alpha": arguments.alpha,
        "cutoff_N": arguments.cutoff,
        "grid_points_M": arguments.grid_points,
        "warmups": arguments.warmups,
        "repetitions": arguments.repetitions,
    }
    for phase, summary in phases.items():
        for statistic, value in summary.items():
            row[f"{phase}_{statistic}"] = round(value, 6)
    return row


def machine_description() -> str:
    if platform.system() == "Darwin":
        try:
            result = subprocess.run(
                ["system_profiler", "SPHardwareDataType"],
                check=True,
                capture_output=True,
                text=True,
                timeout=15,
            )
            wanted = ("Model Name", "Chip", "Total Number of Cores", "Memory")
            fields: list[str] = []
            for line in result.stdout.splitlines():
                stripped = line.strip()
                for label in wanted:
                    prefix = f"{label}:"
                    if stripped.startswith(prefix):
                        fields.append(stripped[len(prefix) :].strip())
            if fields:
                return "; ".join(fields)
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            check=True,
            capture_output=True,
            text=True,
        )
        description = result.stdout.strip()
        if description:
            return description
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass
    return platform.processor() or platform.machine()


def main() -> None:
    arguments = parse_arguments()
    datasets = load_example_datasets()
    rows = [benchmark_dataset(dataset, arguments) for dataset in datasets]

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with arguments.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "command": [sys.executable, *sys.argv],
        "method": {
            "clock": "time.perf_counter_ns",
            "summary": "median and linear-interpolation interquartile range",
            "phase_measurement": "each phase timed independently",
            "eigenpair_selection": "ceil(d/2) even and floor(d/2) odd; no extra modes",
            "excluded": [
                "dataset loading",
                "file output",
                "raster rendering",
                "Feature Congestion",
            ],
        },
        "parameters": {
            "alpha": arguments.alpha,
            "cutoff_N": arguments.cutoff,
            "grid_points_M": arguments.grid_points,
            "warmups": arguments.warmups,
            "repetitions": arguments.repetitions,
        },
        "environment": {
            "machine": machine_description(),
            "architecture": platform.machine(),
            "operating_system": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "requested_thread_limits": {
                variable: os.environ.get(variable)
                for variable in (
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
        },
    }
    arguments.metadata.parent.mkdir(parents=True, exist_ok=True)
    arguments.metadata.write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Wrote {arguments.output}")
    print(f"Wrote {arguments.metadata}")
    for row in rows:
        print(
            f"{row['dataset']}: {row['combined_median_ms']:.3f} ms "
            f"[{row['combined_q1_ms']:.3f}, {row['combined_q3_ms']:.3f}]"
        )


if __name__ == "__main__":
    main()
