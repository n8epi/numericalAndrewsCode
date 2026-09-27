#!/usr/bin/env python3
"""Generate the two reproducible Iris figures used in the introduction."""

from __future__ import annotations

from numerical_andrews.paths import RESULTS
import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PIL import Image
from sklearn.datasets import load_iris

from numerical_andrews.core import (
    Dataset,
    PLOT_BACKGROUND,
    canonical_pca_sign_components,
    evaluate_mmqv_modes,
    evaluate_spatial_spectral_modes,
    load_example_datasets,
    preprocessing_description,
    spatial_spectral_modes,
)


DISPLAY_ALPHA = 0.2
FOURIER_CUTOFF = 256
GRID_POINTS = 4097
CANVAS_WIDTH = 768
CANVAS_HEIGHT = 512
CANVAS_DPI = 100
LINEWIDTH_POINTS = 0.65
LINE_ALPHA = 0.52
VERTICAL_PADDING = 1.05


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=RESULTS,
    )
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def render_curves(
    dataset: Dataset,
    curves: np.ndarray,
    points: np.ndarray,
    y_limit: float,
) -> np.ndarray:
    """Render an introductory plot with explicit coordinates and fitted limits."""
    figure = Figure(
        figsize=(CANVAS_WIDTH / CANVAS_DPI, CANVAS_HEIGHT / CANVAS_DPI),
        dpi=CANVAS_DPI,
        facecolor=PLOT_BACKGROUND,
    )
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.10, 0.12, 0.88, 0.86))
    axis.set_facecolor(PLOT_BACKGROUND)
    for label, color in enumerate(dataset.colors):
        indices = np.flatnonzero(dataset.labels == label)
        axis.plot(
            points,
            curves[indices].T,
            color=color,
            linewidth=LINEWIDTH_POINTS,
            alpha=LINE_ALPHA,
        )
    axis.set_xlim(float(points[0]), float(points[-1]))
    axis.set_ylim(-y_limit, y_limit)
    axis.set_xlabel(r"$t$", fontsize=12)
    axis.set_ylabel(r"$\Phi(x)(t)$", fontsize=12)
    axis.set_xticks(np.linspace(-0.5, 0.5, 5))
    axis.tick_params(labelsize=10, colors="#444444")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        axis.spines[side].set_color("#999999")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.asarray(rgba[..., :3], dtype=np.uint8).copy()


def main() -> None:
    arguments = parse_arguments()
    output_directory = arguments.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)

    iris = next(
        dataset for dataset in load_example_datasets() if dataset.key == "iris"
    )
    raw_iris = load_iris()
    raw_values = np.asarray(raw_iris.data, dtype=float)
    feature_means = np.mean(raw_values, axis=0)
    _, _, right_vectors_t = np.linalg.svd(iris.values, full_matrices=False)
    pca_directions = canonical_pca_sign_components(right_vectors_t.T)
    if not np.allclose(iris.values @ pca_directions, iris.scores):
        raise RuntimeError("worked-example PCA directions do not reproduce scores")

    points = np.linspace(-0.5, 0.5, GRID_POINTS)
    dimension = iris.values.shape[1]
    mmqv_curves = iris.scores @ evaluate_mmqv_modes(dimension, points).T

    display_modes = spatial_spectral_modes(
        DISPLAY_ALPHA, FOURIER_CUTOFF, dimension
    )[0]
    display_curves = iris.scores @ evaluate_spatial_spectral_modes(
        display_modes, points
    ).T

    check_points = np.array([0.0, 0.125, 0.25])
    check_basis = evaluate_spatial_spectral_modes(display_modes, check_points)
    first_curve_values = iris.scores[0] @ check_basis.T
    mode_eigenvalues = np.array(
        [mode.eigenvalue for mode in display_modes], dtype=float
    )
    mean_objective = float(
        np.dot(iris.singular_values**2, mode_eigenvalues)
        / iris.values.shape[0]
    )

    y_limits = {
        "mmqv": VERTICAL_PADDING * float(np.max(np.abs(mmqv_curves))),
        "mmssqv": VERTICAL_PADDING * float(np.max(np.abs(display_curves))),
    }
    outputs = {
        "mmqv": "iris_intro_mmqv.png",
        "mmssqv": "iris_intro_mmssqv_alpha_0p2.png",
    }
    for key, curves in (("mmqv", mmqv_curves), ("mmssqv", display_curves)):
        image = render_curves(iris, curves, points, y_limits[key])
        Image.fromarray(image).save(
            output_directory / outputs[key],
            dpi=(CANVAS_DPI, CANVAS_DPI),
        )

    metadata = {
        "analysis": "introductory Iris MMQV/MMSSQV comparison",
        "data": {
            "display_name": iris.display_name,
            "observations": int(iris.values.shape[0]),
            "dimension": int(iris.values.shape[1]),
            "preprocessing": preprocessing_description(iris.key),
        },
        "numerics": {
            "display_alpha": DISPLAY_ALPHA,
            "fourier_cutoff_N": FOURIER_CUTOFF,
            "curve_grid_points": GRID_POINTS,
            "parameter_interval": [-0.5, 0.5],
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
        },
        "worked_iris_example": {
            "feature_order": [str(name) for name in raw_iris.feature_names],
            "raw_first_observation": raw_values[0].tolist(),
            "feature_means": feature_means.tolist(),
            "centered_first_observation": iris.values[0].tolist(),
            "pca_directions_as_columns": pca_directions.tolist(),
            "singular_values": iris.singular_values.tolist(),
            "first_score_vector": iris.scores[0].tolist(),
            "alpha": DISPLAY_ALPHA,
            "fourier_cutoff_N": FOURIER_CUTOFF,
            "modes": [
                {
                    "global_rank": rank,
                    "parity": mode.parity,
                    "within_block_rank": mode.sector_rank + 1,
                    "eigenvalue": mode.eigenvalue,
                }
                for rank, mode in enumerate(display_modes, start=1)
            ],
            "evaluation_points": check_points.tolist(),
            "basis_values_rows_by_point": check_basis.tolist(),
            "first_curve_values": first_curve_values.tolist(),
            "mean_spatial_spectral_objective": mean_objective,
        },
        "render": {
            "coordinate_protocol": (
                "separate symmetric vertical ranges fitted to each displayed "
                "curve family, with 5% padding and visible amplitude ticks"
            ),
            "symmetric_y_limits": y_limits,
            "canvas_pixels": [CANVAS_WIDTH, CANVAS_HEIGHT],
            "dpi": CANVAS_DPI,
            "line_width_points": LINEWIDTH_POINTS,
            "line_alpha": LINE_ALPHA,
            "background": PLOT_BACKGROUND,
            "color_order": list(iris.colors),
            "row_order": "scikit-learn Iris dataset order",
            "decorations": "parameter and amplitude axes; species colors in caption",
        },
        "outputs": outputs,
        "versions": {
            "python": platform.python_version(),
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
            "scikit-learn": package_version("scikit-learn"),
            "matplotlib": package_version("matplotlib"),
            "Pillow": package_version("Pillow"),
        },
        "command": " ".join(sys.argv),
    }
    (output_directory / "intro_iris_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Wrote introductory Iris figures to {output_directory}")


if __name__ == "__main__":
    main()
