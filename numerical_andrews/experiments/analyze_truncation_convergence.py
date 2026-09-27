#!/usr/bin/env python3
"""Reproduce the finite-section convergence diagnostic for Theorem 2.

The experiment uses the smallest value on the manuscript's regularization
path, alpha=0.001, and compares several symmetric Fourier cutoffs N with the
numerical reference N_ref=512.  A symmetric cutoff represents frequencies
-N,...,N, or equivalently the block matrix

    B_even(N+1) direct-sum B_odd(N).

Modes are matched by parity and within-block rank, exactly as in Theorem 2.
The script writes detailed mode and objective data, a per-dataset summary,
and the three-panel figure included in the revised manuscript.
"""

from __future__ import annotations

from numerical_andrews.paths import RESULTS
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numerical_andrews.core import load_example_datasets, spatial_spectral_modes


ALPHA = 1.0e-3
REFERENCE_CUTOFF = 512
CUTOFFS = (15, 16, 20, 24, 28, 32, 40, 48, 64, 96, 128, 192, 256)


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=RESULTS,
    )
    return parser.parse_args()


def pad_and_align(vector: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Zero-pad a finite vector and choose the closest of its two signs."""
    if vector.size > reference.size:
        raise ValueError("reference vector must be at least as long as vector")
    padded = np.zeros_like(reference)
    padded[: vector.size] = vector
    if np.dot(padded, reference) < 0.0:
        padded *= -1.0
    return padded


def compute_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mode_rows: list[dict[str, float | int | str]] = []
    objective_rows: list[dict[str, float | int | str]] = []

    for dataset in load_example_datasets():
        dimension = dataset.values.shape[1]
        # The published diagnostic requested d+3 eigenpairs in each parity block.
        # Retain that solve width: changing it shifts the last-bit bisection
        # errors plotted at the floating-point floor. The displayed bases still
        # use only the first d interlaced modes.
        reference_global, reference_sectors = spatial_spectral_modes(
            ALPHA, REFERENCE_CUTOFF, dimension, extra_per_sector=dimension // 2 + 3
        )
        reference_by_key = {
            mode.key: mode
            for sector_modes in reference_sectors.values()
            for mode in sector_modes
        }
        weights = dataset.singular_values[:dimension] ** 2 / dataset.values.shape[0]
        reference_eigenvalues = np.array(
            [reference_by_key[mode.key].eigenvalue for mode in reference_global]
        )
        reference_objective = float(np.dot(weights, reference_eigenvalues))

        for cutoff in CUTOFFS:
            _, current_sectors = spatial_spectral_modes(
                ALPHA, cutoff, dimension, extra_per_sector=dimension // 2 + 3
            )
            current_by_key = {
                mode.key: mode
                for sector_modes in current_sectors.values()
                for mode in sector_modes
            }
            current_eigenvalues: list[float] = []

            for global_rank, reference_mode in enumerate(reference_global, start=1):
                current_mode = current_by_key[reference_mode.key]
                reference_sector_mode = reference_by_key[reference_mode.key]
                aligned = pad_and_align(
                    current_mode.coefficients,
                    reference_sector_mode.coefficients,
                )
                absolute_eigenvalue_error = abs(
                    current_mode.eigenvalue - reference_sector_mode.eigenvalue
                )
                relative_eigenvalue_error = (
                    absolute_eigenvalue_error
                    / abs(reference_sector_mode.eigenvalue)
                )
                eigenvector_error = float(
                    np.linalg.norm(aligned - reference_sector_mode.coefficients)
                )
                current_eigenvalues.append(current_mode.eigenvalue)
                mode_rows.append(
                    {
                        "dataset": dataset.key,
                        "dataset_name": dataset.display_name,
                        "dimension": dimension,
                        "alpha": ALPHA,
                        "cutoff_N": cutoff,
                        "reference_cutoff_N": REFERENCE_CUTOFF,
                        "reference_global_mode": global_rank,
                        "parity": reference_mode.parity,
                        "within_block_mode": reference_mode.sector_rank + 1,
                        "eigenvalue": current_mode.eigenvalue,
                        "reference_eigenvalue": reference_sector_mode.eigenvalue,
                        "eigenvalue_absolute_error": absolute_eigenvalue_error,
                        "eigenvalue_relative_error": relative_eigenvalue_error,
                        "eigenvector_l2_error": eigenvector_error,
                    }
                )

            objective = float(np.dot(weights, current_eigenvalues))
            objective_rows.append(
                {
                    "dataset": dataset.key,
                    "dataset_name": dataset.display_name,
                    "dimension": dimension,
                    "alpha": ALPHA,
                    "cutoff_N": cutoff,
                    "reference_cutoff_N": REFERENCE_CUTOFF,
                    "weighted_objective": objective,
                    "reference_weighted_objective": reference_objective,
                    "objective_absolute_error": abs(
                        objective - reference_objective
                    ),
                    "objective_relative_error": abs(
                        objective - reference_objective
                    )
                    / abs(reference_objective),
                }
            )

    modes = pd.DataFrame(mode_rows)
    objectives = pd.DataFrame(objective_rows)
    mode_summary = (
        modes.groupby(
            [
                "dataset",
                "dataset_name",
                "dimension",
                "alpha",
                "cutoff_N",
                "reference_cutoff_N",
            ],
            as_index=False,
            sort=False,
        )
        .agg(
            max_absolute_eigenvalue_error=("eigenvalue_absolute_error", "max"),
            max_relative_eigenvalue_error=("eigenvalue_relative_error", "max"),
            max_eigenvector_l2_error=("eigenvector_l2_error", "max"),
        )
    )
    summary = mode_summary.merge(
        objectives[
            ["dataset", "cutoff_N", "objective_relative_error"]
        ],
        on=["dataset", "cutoff_N"],
        how="left",
        validate="one_to_one",
    )
    return modes, objectives, summary


def plot_diagnostics(summary: pd.DataFrame, output_directory: Path) -> None:
    datasets = list(dict.fromkeys(summary["dataset"].tolist()))
    colors = {
        "iris": "#0072B2",
        "bc": "#D55E00",
        "diabetes": "#009E73",
    }
    markers = {
        "iris": "o",
        "bc": "s",
        "diabetes": "^",
    }
    labels = {
        str(row.dataset): str(row.dataset_name)
        for row in summary[["dataset", "dataset_name"]]
        .drop_duplicates()
        .itertuples(index=False)
    }
    specifications = (
        (
            "max_absolute_eigenvalue_error",
            "Maximum eigenvalue error",
            r"$e_{\lambda,D}(N)$",
        ),
        (
            "max_eigenvector_l2_error",
            "Maximum eigenvector error",
            r"$e_{v,D}(N)$",
        ),
        (
            "objective_relative_error",
            "MMSSQV objective error",
            r"$e_{J,D}(N)$",
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(7.2, 2.85), dpi=180)
    display_floor = np.finfo(float).eps

    for axis, (column, title, ylabel) in zip(axes, specifications, strict=True):
        for dataset in datasets:
            subset = summary[summary["dataset"] == dataset].sort_values("cutoff_N")
            axis.semilogy(
                subset["cutoff_N"],
                np.maximum(subset[column].to_numpy(float), display_floor),
                color=colors[dataset],
                marker=markers[dataset],
                markersize=4.5,
                linewidth=1.4,
                label=labels[dataset],
            )
        axis.axvline(
            256,
            color="#3F3F3F",
            linestyle=(0, (3, 2)),
            linewidth=1.0,
            zorder=1,
        )
        axis.set_title(title, fontsize=9.5)
        axis.set_xlabel(r"Symmetric Fourier cutoff $N$", fontsize=8.5)
        axis.set_ylabel(ylabel, fontsize=8.5)
        axis.grid(True, which="both", alpha=0.22, linewidth=0.6)
        axis.tick_params(labelsize=7.8)
        axis.margins(x=0.035)

    axes[0].legend(frameon=False, fontsize=7.8)
    axes[2].text(
        0.97,
        0.05,
        r"dashed: $N=256$",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#3F3F3F",
    )
    figure.suptitle(
        r"Finite-section convergence at $\alpha=10^{-3}$ "
        r"using numerical reference $N_{\mathrm{ref}}=512$",
        fontsize=10.5,
        y=0.985,
    )
    figure.subplots_adjust(
        left=0.075, right=0.985, top=0.80, bottom=0.205, wspace=0.36
    )
    figure.savefig(output_directory / "truncation_convergence.png", dpi=300)
    figure.savefig(output_directory / "truncation_convergence.pdf")
    plt.close(figure)


def main() -> None:
    arguments = parse_arguments()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    modes, objectives, summary = compute_diagnostics()
    modes.to_csv(
        output_directory / "theorem2_truncation_modes.csv", index=False
    )
    objectives.to_csv(
        output_directory / "theorem2_truncation_objectives.csv", index=False
    )
    summary.to_csv(
        output_directory / "theorem2_truncation_summary.csv", index=False
    )
    plot_diagnostics(summary, output_directory)

    selected = summary[summary["cutoff_N"] == 256]
    print(
        "N=256 maxima across datasets: "
        f"absolute eigenvalue={selected['max_absolute_eigenvalue_error'].max():.6g}, "
        f"eigenvector l2={selected['max_eigenvector_l2_error'].max():.6g}, "
        f"relative objective={selected['objective_relative_error'].max():.6g}"
    )
    print(f"Wrote truncation diagnostics to {output_directory}")


if __name__ == "__main__":
    main()
