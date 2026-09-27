#!/usr/bin/env python3
"""Trace a three-number Feature Congestion profile over MMSSQV alpha.

By default, this script compares MMSSQV against the saved mean-FC optimized
MMQV baseline on its fixed optimization coordinate system. Pass --baseline-mode
two to reproduce the named cosine-first and sine-first sensitivity comparison.
It
summarizes the Rosenholtz Feature Congestion map by

1. its spatial mean (the published p=1 global FC scalar),
2. the mean of the most congested 5% of pixels, and
3. the mean of the most congested 1% of pixels.

Each quantity is reported as an absolute value and as a ratio to the specified
MMQV baseline. The shared vertical limit is fixed separately for each dataset
and covers *all* evaluated alpha values.
This prevents panel autoscaling from masquerading as a congestion change.

The illustrative panels instead fit each curve family with 5% vertical
padding. Their local maps and statistics are recomputed separately and saved
in feature_congestion_display_profiles.csv; they do not replace the
fixed-coordinate path measurements.

The script uses the controlled RGB renderer in ``numerical_andrews.rendering``
and the Feature Congestion implementation in ``numerical_andrews.congestion``.
"""

from __future__ import annotations

from numerical_andrews.paths import RESULTS
import argparse
import importlib.metadata
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from numerical_andrews.congestion import upper_tail_mean, fc_profile
import numpy as np
import pandas as pd

from numerical_andrews.congestion import feature_congestion
from numerical_andrews.rendering import (
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    CANVAS_DPI,
    LINEWIDTH_POINTS,
    LINE_ALPHA,
    VERTICAL_PADDING,
    render_curves,
)
from numerical_andrews.core import (
    Dataset,
    evaluate_mmqv_modes,
    evaluate_limiting_aligned_mmqv_modes,
    evaluate_spatial_spectral_modes,
    load_example_datasets,
    preprocessing_description,
    spatial_spectral_modes,
)


DEFAULT_ALPHA_MIN = 1.0e-3
DEFAULT_ALPHA_MAX = 1.0e2
DEFAULT_ALPHA_COUNT = 16
DEFAULT_CUTOFF = 256
DEFAULT_REFERENCE_CUTOFF = 512
COMMON_REFERENCE_ALPHA = 0.2


@dataclass(frozen=True)
class CommonAlphaDisplay:
    """Raster/map pair retained for the common-alpha manuscript figure."""

    dataset: Dataset
    alpha: float
    baseline_y_limit: tuple[float, float]
    smoothed_y_limit: tuple[float, float]
    baseline_image: np.ndarray
    baseline_map: np.ndarray
    baseline_profile: dict[str, float]
    smoothed_image: np.ndarray
    smoothed_map: np.ndarray
    smoothed_profile: dict[str, float]


def parse_arguments() -> argparse.Namespace:
    script_directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=RESULTS,
    )
    parser.add_argument('--baseline-mode', choices=['optimized', 'two'], default='optimized',
                        help='Use saved mean-FC optimized MMQV, or the two named bases')
    parser.add_argument("--alpha-min", type=float, default=DEFAULT_ALPHA_MIN)
    parser.add_argument("--alpha-max", type=float, default=DEFAULT_ALPHA_MAX)
    parser.add_argument("--alpha-count", type=int, default=DEFAULT_ALPHA_COUNT)
    parser.add_argument(
        "--cutoff",
        type=int,
        default=DEFAULT_CUTOFF,
        help="Fourier cutoff for every alpha (256 safely resolves small alpha).",
    )
    parser.add_argument(
        "--reference-cutoff",
        type=int,
        default=DEFAULT_REFERENCE_CUTOFF,
        help="Larger cutoff used to validate truncation at alpha-min.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=4097,
        help="Dense curve-evaluation grid used before rasterization.",
    )
    return parser.parse_args()


def validate_arguments(arguments: argparse.Namespace) -> None:
    if arguments.alpha_min <= 0.0 or arguments.alpha_max <= arguments.alpha_min:
        raise ValueError("Require 0 < alpha-min < alpha-max")
    if arguments.alpha_count < 3:
        raise ValueError("alpha-count must be at least 3")
    if arguments.cutoff < 256:
        raise ValueError("cutoff must be at least 256 for the small-alpha regime")
    if arguments.reference_cutoff <= arguments.cutoff:
        raise ValueError("reference-cutoff must exceed cutoff")
    if arguments.grid < 257 or arguments.grid % 2 == 0:
        raise ValueError("grid must be an odd integer of at least 257")


def truncation_validation(
    datasets: list[Dataset],
    alpha: float,
    cutoff: int,
    reference_cutoff: int,
) -> pd.DataFrame:
    """Compare cutoffs using the published diagnostic width d+3 per parity."""
    rows: list[dict[str, float | int | str]] = []
    for dataset in datasets:
        dimension = dataset.values.shape[1]
        current_global, current_sectors = spatial_spectral_modes(
            alpha, cutoff, dimension, extra_per_sector=dimension // 2 + 3
        )
        reference_global, reference_sectors = spatial_spectral_modes(
            alpha, reference_cutoff, dimension, extra_per_sector=dimension // 2 + 3
        )
        current_by_key = {
            mode.key: mode
            for sector_modes in current_sectors.values()
            for mode in sector_modes
        }
        reference_by_key = {
            mode.key: mode
            for sector_modes in reference_sectors.values()
            for mode in sector_modes
        }
        eigenvalue_errors: list[float] = []
        eigenvector_errors: list[float] = []
        for reference_mode in reference_global:
            current_mode = current_by_key[reference_mode.key]
            reference_sector_mode = reference_by_key[reference_mode.key]
            padded = np.zeros_like(reference_sector_mode.coefficients)
            padded[: current_mode.coefficients.size] = current_mode.coefficients
            if np.dot(padded, reference_sector_mode.coefficients) < 0.0:
                padded *= -1.0
            eigenvector_errors.append(
                float(np.linalg.norm(padded - reference_sector_mode.coefficients))
            )
            eigenvalue_errors.append(
                abs(current_mode.eigenvalue - reference_sector_mode.eigenvalue)
                / max(abs(reference_sector_mode.eigenvalue), 1.0)
            )

        weights = dataset.singular_values[:dimension] ** 2 / dataset.values.shape[0]
        current_objective = float(
            np.dot(weights, [mode.eigenvalue for mode in current_global])
        )
        reference_objective = float(
            np.dot(weights, [mode.eigenvalue for mode in reference_global])
        )
        rows.append(
            {
                "dataset": dataset.key,
                "dataset_name": dataset.display_name,
                "dimension": dimension,
                "alpha": alpha,
                "cutoff_N": cutoff,
                "reference_cutoff_N": reference_cutoff,
                "max_relative_eigenvalue_error": max(eigenvalue_errors),
                "max_eigenvector_l2_error": max(eigenvector_errors),
                "weighted_objective": current_objective,
                "reference_weighted_objective": reference_objective,
                "relative_weighted_objective_error": abs(
                    current_objective - reference_objective
                )
                / max(abs(reference_objective), np.finfo(float).tiny),
            }
        )
    return pd.DataFrame(rows)


def basis_cache_and_limit(
    dataset: Dataset,
    points: np.ndarray,
    alpha_values: np.ndarray,
    cutoff: int,
) -> tuple[dict[str, np.ndarray], dict[float, np.ndarray], float]:
    dimension = dataset.values.shape[1]
    mmqv_curves = {
        "cosine_first": dataset.scores @ evaluate_mmqv_modes(dimension, points).T,
        "sine_first": dataset.scores @ evaluate_limiting_aligned_mmqv_modes(dimension, points).T,
    }
    maximum = max(float(np.max(np.abs(curves))) for curves in mmqv_curves.values())
    basis_cache: dict[float, np.ndarray] = {}
    for alpha in alpha_values:
        modes = spatial_spectral_modes(float(alpha), cutoff, dimension)[0]
        basis = evaluate_spatial_spectral_modes(modes, points)
        basis_cache[float(alpha)] = basis
        curves = dataset.scores @ basis.T
        maximum = max(maximum, float(np.max(np.abs(curves))))
    return mmqv_curves, basis_cache, VERTICAL_PADDING * maximum


def optimized_curves(dataset, points):
    from numerical_andrews.experiments.optimize_mmqv_feature_congestion import mmqv_transform
    path = RESULTS/'optimized_mmqv'/f'{dataset.key}.json'
    result = json.loads(path.read_text())
    if result['status'] != 'complete':
        raise RuntimeError(f"Optimization is not complete for {dataset.key}")
    best = result['best']
    d = dataset.values.shape[1]
    transform = mmqv_transform(d, best['angles'], best['reflections'], best['constant_sign'])
    return (dataset.scores@transform.T)@evaluate_mmqv_modes(transform.shape[0], points).T, result


def evaluate_dataset(dataset, points, alpha_values, cutoff, baseline_mode="optimized"):
    mmqv_curves, basis_cache, y_limit = basis_cache_and_limit(
        dataset, points, alpha_values, cutoff
    )
    if baseline_mode == "optimized":
        curves, result = optimized_curves(dataset, points)
        if y_limit > result['y_limit']*(1+1e-12):
            raise ValueError('Current alpha path requires a wider range; rerun the optimization')
        y_limit = result['y_limit']
        mmqv_curves = {'optimized': curves}
    baselines, baseline_rows = {}, []
    for name, curves in mmqv_curves.items():
        baselines[name], _ = fc_profile(render_curves(dataset, curves, points, y_limit))
        baseline_rows.append({
            "dataset": dataset.key, "dataset_name": dataset.display_name,
            "observations": int(dataset.values.shape[0]),
            "dimension": int(dataset.values.shape[1]), "method": "MMQV",
            "baseline": name, "y_limit": y_limit, **baselines[name],
        })
    rows = []
    for alpha in alpha_values:
        alpha = float(alpha)
        curves = dataset.scores @ basis_cache[alpha].T
        profile, _ = fc_profile(render_curves(dataset, curves, points, y_limit))
        for name, baseline in baselines.items():
            row = {
                "dataset": dataset.key, "dataset_name": dataset.display_name,
                "observations": int(dataset.values.shape[0]),
                "dimension": int(dataset.values.shape[1]), "method": "MMSSQV",
                "baseline": name, "alpha": alpha, "cutoff_N": cutoff,
                "y_limit": y_limit,
                "is_common_alpha": bool(np.isclose(alpha, COMMON_REFERENCE_ALPHA,
                                                     rtol=1e-12, atol=0.0)),
                **profile,
            }
            for metric in ("mean_fc", "upper_5pct_mean_fc", "upper_1pct_mean_fc"):
                row[f"baseline_{metric}"] = baseline[metric]
                row[f"{metric}_ratio"] = profile[metric] / baseline[metric]
                row[f"{metric}_percent_change"] = 100*(row[f"{metric}_ratio"]-1)
            rows.append(row)
    return rows, baseline_rows, build_fitted_display(dataset, points, cutoff, baseline_mode)


def fitted_limits(curves: np.ndarray) -> tuple[float, float]:
    low, high = float(np.min(curves)), float(np.max(curves))
    padding = 0.05 * (high - low)
    return low - padding, high + padding


def build_fitted_display(
    dataset: Dataset, points: np.ndarray, cutoff: int, baseline_mode="optimized"
) -> CommonAlphaDisplay:
    mmqv_curves = dataset.scores @ evaluate_mmqv_modes(
        dataset.values.shape[1], points
    ).T
    if baseline_mode == "optimized":
        mmqv_curves, _ = optimized_curves(dataset, points)
    display_curves = dataset.scores @ evaluate_spatial_spectral_modes(
        spatial_spectral_modes(COMMON_REFERENCE_ALPHA, cutoff, dataset.values.shape[1])[0],
        points,
    ).T
    display_baseline_limit = fitted_limits(mmqv_curves)
    display_smoothed_limit = fitted_limits(display_curves)
    display_baseline_image = render_curves(
        dataset, mmqv_curves, points, display_baseline_limit
    )
    display_smoothed_image = render_curves(
        dataset, display_curves, points, display_smoothed_limit
    )
    display_baseline_profile, display_baseline_map = fc_profile(display_baseline_image)
    display_smoothed_profile, display_smoothed_map = fc_profile(display_smoothed_image)
    return CommonAlphaDisplay(
        dataset=dataset,
        alpha=COMMON_REFERENCE_ALPHA,
        baseline_y_limit=display_baseline_limit,
        smoothed_y_limit=display_smoothed_limit,
        baseline_image=display_baseline_image,
        baseline_map=display_baseline_map,
        baseline_profile=display_baseline_profile,
        smoothed_image=display_smoothed_image,
        smoothed_map=display_smoothed_map,
        smoothed_profile=display_smoothed_profile,
    )


def build_landmarks(profile: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    ratio_columns = (
        "mean_fc_ratio",
        "upper_5pct_mean_fc_ratio",
        "upper_1pct_mean_fc_ratio",
    )
    for (dataset_key, baseline_name), subset in profile.groupby(["dataset", "baseline"], sort=False):
        subset = subset.sort_values("alpha")
        landmarks: list[tuple[str, pd.Series]] = []
        common = subset[subset["is_common_alpha"]]
        if len(common) == 1:
            landmarks.append(("common_alpha_0.2", common.iloc[0]))
        minimax_index = subset[list(ratio_columns)].max(axis=1).idxmin()
        landmarks.append(("minimax_three_number_profile", subset.loc[minimax_index]))
        mean_index = subset["mean_fc_ratio"].idxmin()
        landmarks.append(("minimum_mean_fc", subset.loc[mean_index]))
        for label, row in landmarks:
            rows.append(
                {
                    "dataset": dataset_key,
                    "dataset_name": str(row["dataset_name"]),
                    "landmark": label,
                    "baseline": baseline_name,
                    "alpha": float(row["alpha"]),
                    "mean_fc_ratio": float(row["mean_fc_ratio"]),
                    "upper_5pct_mean_fc_ratio": float(
                        row["upper_5pct_mean_fc_ratio"]
                    ),
                    "upper_1pct_mean_fc_ratio": float(
                        row["upper_1pct_mean_fc_ratio"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def plot_profiles(profile, output_png, output_pdf):
    dataset_order = list(dict.fromkeys(profile["dataset"].tolist()))
    specs = (("mean_fc_ratio", "Mean FC", "#0072B2", "o"),
             ("upper_5pct_mean_fc_ratio", "Upper 5% mean", "#D55E00", "s"),
             ("upper_1pct_mean_fc_ratio", "Upper 1% mean", "#CC79A7", "^"))
    values = profile[[item[0] for item in specs]].to_numpy(float)
    lower, upper = min(0.97, values.min()), max(1.03, values.max())
    padding = max(0.025, 0.08*(upper-lower))
    baseline_order = list(dict.fromkeys(profile.baseline))
    row_count = len(baseline_order)
    figure, axes = plt.subplots(row_count, len(dataset_order),
                               figsize=(7.2, 3.05 if row_count == 1 else 5.1),
                               dpi=180, sharex=True, sharey=True, squeeze=False)
    handles = []
    for row, baseline in enumerate(baseline_order):
        for column, dataset in enumerate(dataset_order):
            axis = axes[row, column]
            subset = profile[(profile.dataset == dataset) &
                             (profile.baseline == baseline)].sort_values("alpha")
            alpha = subset.alpha.to_numpy(float)
            axis.axhspan(lower-padding, 1, color="#009E73", alpha=0.055)
            axis.axhline(1, color="#3F3F3F", linewidth=0.9)
            axis.axvline(COMMON_REFERENCE_ALPHA, color="#777777",
                         linestyle=(0, (2, 2)), linewidth=0.85)
            for metric, label, color, marker in specs:
                handle, = axis.plot(alpha, subset[metric], color=color, linewidth=1.4,
                                    marker=marker, markersize=4, markevery=2, label=label)
                if row == column == 0:
                    handles.append(handle)
            axis.set_xscale("log")
            axis.set_xlim(alpha.min(), alpha.max())
            axis.set_ylim(lower-padding, upper+padding)
            axis.grid(True, axis="y", alpha=0.2, linewidth=0.6)
            axis.tick_params(labelsize=7.8)
            if row == 0:
                axis.set_title(str(subset.dataset_name.iloc[0]), fontsize=9.5)
            if row == row_count-1:
                axis.set_xlabel(r"Regularization parameter $\alpha$", fontsize=8.5)
        name = baseline.replace('_', '-')
        axes[row, 0].set_ylabel(f"MMSSQV / {name} MMQV", fontsize=8.5)
    from matplotlib.lines import Line2D
    handles.append(Line2D([0], [0], color="#777777", linestyle=(0, (2, 2)),
                          label=r"Common $\alpha=0.2$"))
    figure.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7.8)
    figure.suptitle("Feature Congestion relative to " + ("optimized MMQV" if row_count == 1 else "two MMQV bases"), fontsize=10.5)
    subtitle = ("Fixed vertical scale; below 1 means lower congestion than the MMQV baseline" if row_count == 1
                else "Same rasters and vertical scale in both comparisons; below 1 means lower congestion")
    figure.text(0.5, 0.90 if row_count == 1 else 0.925, subtitle,
                ha="center", fontsize=7.8, color="#555555")
    figure.subplots_adjust(left=0.095, right=0.985, top=0.79 if row_count == 1 else 0.865, bottom=0.24 if row_count == 1 else 0.12,
                           wspace=0.22, hspace=0.18)
    figure.savefig(output_png, dpi=300)
    figure.savefig(output_pdf)
    plt.close(figure)


def plot_common_alpha_examples(
    displays: list[CommonAlphaDisplay],
    output_png: Path,
    output_pdf: Path,
    baseline_mode="optimized",
) -> None:
    """Plot the compared rasters beside their local Feature Congestion maps."""
    figure, axes = plt.subplots(
        len(displays),
        4,
        figsize=(7.2, 1.55 * len(displays) + 0.25),
        dpi=180,
        constrained_layout=True,
        squeeze=False,
    )
    column_titles = (
        "Optimized MMQV" if baseline_mode == "optimized" else "Cosine-first MMQV",
        "MMQV congestion",
        "MMSSQV\n" + rf"$\alpha={COMMON_REFERENCE_ALPHA:g}$",
        "MMSSQV congestion",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=8.6)

    for row, display in enumerate(displays):
        combined_maps = np.concatenate(
            [display.baseline_map.ravel(), display.smoothed_map.ravel()]
        )
        map_limit = float(np.quantile(combined_maps, 0.995))
        if not np.isfinite(map_limit) or map_limit <= 0.0:
            raise FloatingPointError("invalid Feature Congestion display limit")

        axes[row, 0].imshow(display.baseline_image, interpolation="none")
        axes[row, 1].imshow(
            display.baseline_map,
            cmap="magma",
            vmin=0.0,
            vmax=map_limit,
            interpolation="none",
        )
        axes[row, 2].imshow(display.smoothed_image, interpolation="none")
        heatmap = axes[row, 3].imshow(
            display.smoothed_map,
            cmap="magma",
            vmin=0.0,
            vmax=map_limit,
            interpolation="none",
        )

        axes[row, 0].set_xlabel(
            rf"$y\in[{display.baseline_y_limit[0]:.3g},{display.baseline_y_limit[1]:.3g}]$",
            fontsize=8.2,
        )
        axes[row, 2].set_xlabel(
            rf"$y\in[{display.smoothed_y_limit[0]:.3g},{display.smoothed_y_limit[1]:.3g}]$",
            fontsize=8.2,
        )
        baseline = display.baseline_profile
        smoothed = display.smoothed_profile
        axes[row, 1].set_xlabel(
            "FC profile\n"
            f"({baseline['mean_fc']:.2f}, "
            f"{baseline['upper_5pct_mean_fc']:.2f}, "
            f"{baseline['upper_1pct_mean_fc']:.2f})",
            fontsize=8.2,
        )
        axes[row, 3].set_xlabel(
            "FC profile\n"
            f"({smoothed['mean_fc']:.2f}, "
            f"{smoothed['upper_5pct_mean_fc']:.2f}, "
            f"{smoothed['upper_1pct_mean_fc']:.2f})",
            fontsize=8.2,
        )
        axes[row, 0].set_ylabel(display.dataset.display_name, fontsize=9.2)
        colorbar = figure.colorbar(
            heatmap,
            ax=(axes[row, 1], axes[row, 3]),
            location="right",
            fraction=0.032,
            pad=0.012,
            extend="max",
        )
        colorbar.set_label(r"Local FC $M_I(u)$", fontsize=8.2)
        colorbar.ax.tick_params(labelsize=7.6)

        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_visible(False)

    figure.suptitle(
        "Rendered Andrews plots and their local Feature Congestion maps",
        fontsize=10.5,
    )
    figure.savefig(output_png, dpi=300)
    figure.savefig(output_pdf)
    plt.close(figure)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    arguments = parse_arguments()
    validate_arguments(arguments)
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    alpha_values = np.logspace(
        np.log10(arguments.alpha_min),
        np.log10(arguments.alpha_max),
        arguments.alpha_count,
    )
    # Include the common illustration alongside the logarithmic sensitivity grid.
    alpha_values = np.sort(np.append(
        alpha_values[~np.isclose(alpha_values, COMMON_REFERENCE_ALPHA, rtol=1e-12, atol=0.0)],
        COMMON_REFERENCE_ALPHA,
    ))
    datasets = load_example_datasets()
    points = np.linspace(-0.5, 0.5, arguments.grid)
    truncation = truncation_validation(
        datasets,
        arguments.alpha_min,
        arguments.cutoff,
        arguments.reference_cutoff,
    )

    profile_rows: list[dict[str, float | int | str | bool]] = []
    baseline_rows: list[dict[str, float | int | str]] = []
    displays: list[CommonAlphaDisplay] = []
    for dataset in datasets:
        print(f"Evaluating fixed-scale path and fitted display: {dataset.key}", flush=True)
        rows, baseline, display = evaluate_dataset(
            dataset,
            points,
            alpha_values,
            arguments.cutoff,
            arguments.baseline_mode,
        )
        profile_rows.extend(rows)
        baseline_rows.extend(baseline)
        displays.append(display)

    display_rows = []
    for display in displays:
        for method, y_limit, values in (
            ("MMQV", display.baseline_y_limit, display.baseline_profile),
            ("MMSSQV", display.smoothed_y_limit, display.smoothed_profile),
        ):
            display_rows.append({
                "dataset": display.dataset.key,
                "method": method,
                "alpha": display.alpha if method == "MMSSQV" else None,
                "y_min": y_limit[0],
                "y_max": y_limit[1],
                "coordinate_protocol": "method_range_5pct_padding",
                **values,
            })
    pd.DataFrame(display_rows).to_csv(
        output_directory / "feature_congestion_display_profiles.csv", index=False
    )
    profile = pd.DataFrame(profile_rows)
    baseline = pd.DataFrame(baseline_rows)
    landmarks = build_landmarks(profile)
    profile_path = output_directory / "feature_congestion_alpha_profile.csv"
    baseline_path = output_directory / "feature_congestion_alpha_baseline.csv"
    landmarks_path = output_directory / "feature_congestion_alpha_landmarks.csv"
    truncation_path = output_directory / "feature_congestion_alpha_truncation.csv"
    profile.to_csv(profile_path, index=False)
    baseline.to_csv(baseline_path, index=False)
    landmarks.to_csv(landmarks_path, index=False)
    truncation.to_csv(truncation_path, index=False)
    plot_profiles(
        profile,
        output_directory / "feature_congestion_alpha_profile.png",
        output_directory / "feature_congestion_alpha_profile.pdf",
    )
    plot_common_alpha_examples(
        displays,
        output_directory / "feature_congestion_alpha_examples.png",
        output_directory / "feature_congestion_alpha_examples.pdf",
        arguments.baseline_mode,
    )

    metadata = {
        "analysis": "MMSSQV Feature Congestion alpha profile",
        "baseline_mode": arguments.baseline_mode,
        "optimization_records": "optimized_mmqv/{iris,bc,diabetes}.json" if arguments.baseline_mode == "optimized" else None,
        "alpha_grid": {
            "base": "logspace plus common illustrative alpha",
            "minimum": arguments.alpha_min,
            "maximum": arguments.alpha_max,
            "base_count": arguments.alpha_count,
            "evaluated_count": int(alpha_values.size),
            "evaluated_values": alpha_values.tolist(),
        },
        "three_number_profile": {
            "mean_fc": "arithmetic mean of the local FC map (published p=1 scalar)",
            "upper_5pct_mean_fc": (
                "mean of the largest ceil(5% of pixels) local FC values"
            ),
            "upper_1pct_mean_fc": (
                "mean of the largest ceil(1% of pixels) local FC values"
            ),
            "ratios": "MMSSQV divided by optimized MMQV" if arguments.baseline_mode == "optimized" else "MMSSQV divided by each named MMQV basis",
            "named_seed_orderings": {"cosine_first": "1, cos, sin, cos2, sin2, ...",
                                   "sine_first": "1, sin, cos, sin2, cos2, ..."},
        },
        "common_alpha_figure": {
            "alpha": COMMON_REFERENCE_ALPHA,
            "heatmap_scale": (
                "shared within each dataset pair; capped at the joint 99.5th "
                "percentile for display only"
            ),
            "statistics": "recomputed from the uncapped maps of fitted display rasters",
            "coordinate_protocol": "separate min/max limits with 5% range padding",
            "csv": "feature_congestion_display_profiles.csv",
            "png": "feature_congestion_alpha_examples.png",
            "pdf": "feature_congestion_alpha_examples.pdf",
        },
        "data": {
            dataset.key: {
                "display_name": dataset.display_name,
                "observations": int(dataset.values.shape[0]),
                "dimension": int(dataset.values.shape[1]),
                "preprocessing": preprocessing_description(dataset.key),
            }
            for dataset in datasets
        },
        "render": {
            "coordinate_protocol": (
                "fixed during optimization and comparison; covers all MMQV rotations "
                "and reflections and every evaluated MMSSQV alpha" if arguments.baseline_mode == "optimized" else
                "one range covering both named MMQV bases and all MMSSQV alphas"
            ),
            "canvas_pixels": [CANVAS_WIDTH, CANVAS_HEIGHT],
            "dpi": CANVAS_DPI,
            "line_width_points": LINEWIDTH_POINTS,
            "line_alpha": LINE_ALPHA,
            "parameter_interval": [-0.5, 0.5],
            "curve_grid_points": arguments.grid,
            "vertical_padding_factor": VERTICAL_PADDING,
            "color_mode": "RGB",
            "decorations": "none",
        },
        "numerics": {
            "fourier_cutoff_N": arguments.cutoff,
            "pca": {
                "routine": "numpy.linalg.svd",
                "full_matrices": False,
            },
            "jacobi_eigensolver": {
                "routine": "scipy.linalg.eigh_tridiagonal",
                "selection": "indexed low-order eigenpairs (select='i')",
                "lapack_driver": "stebz",
                "absolute_tolerance": 1.0e-14,
                "check_finite": False,
                "cross_parity_ordering": "strict interlacing: E1,O1,E2,O2,...",
            },
            "truncation_validation": {
                "alpha": arguments.alpha_min,
                "reference_cutoff_N": arguments.reference_cutoff,
                "csv": truncation_path.name,
                "max_relative_eigenvalue_error_across_datasets": float(
                    truncation["max_relative_eigenvalue_error"].max()
                ),
                "max_eigenvector_l2_error_across_datasets": float(
                    truncation["max_eigenvector_l2_error"].max()
                ),
                "max_relative_weighted_objective_error_across_datasets": float(
                    truncation["relative_weighted_objective_error"].max()
                ),
            },
        },
        "versions": {
            "python": platform.python_version(),
            "visual-clutter": package_version("visual-clutter"),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "matplotlib": package_version("matplotlib"),
            "Pillow": package_version("Pillow"),
            "opencv-python-headless": package_version("opencv-python-headless"),
            "scikit-image": package_version("scikit-image"),
            "pyrtools": package_version("pyrtools"),
        },
        "feature_congestion": {
            "package": "visual-clutter",
            "spatial_pooling_p": 1,
            "pyramid_levels": 3,
            "preprocessing": (
                "CIELAB constants and Gaussian-pyramid kernel restored from "
                "the authors' MATLAB distribution"
            ),
            "gaussian_pyramid_kernel": [0.05, 0.25, 0.40, 0.25, 0.05],
        },
        "drivers": {
            "compatibility_preprocessing": "numerical_andrews/congestion.py",
            "compatibility_provenance": "numerical_andrews/fc_reference_provenance.json",
            "regularization_path": "analyze_feature_congestion_alpha.py",
            "truncation_convergence": "analyze_truncation_convergence.py",
        },
        "command": " ".join(sys.argv),
        "reproduction_command": ".venv/bin/python " + " ".join(sys.argv),
    }
    (output_directory / "feature_congestion_alpha_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    printed = landmarks[
        landmarks["landmark"].isin(
            ["common_alpha_0.2", "minimax_three_number_profile"]
        )
    ]
    print(printed.to_string(index=False, float_format=lambda value: f"{value:.4g}"))
    print(f"\nWrote Feature Congestion alpha-profile outputs to {output_directory}")


if __name__ == "__main__":
    main()
