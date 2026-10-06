"""Data-only Agg rasterization with the manuscript rendering defaults."""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
from __future__ import annotations
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from .core import Dataset, PLOT_BACKGROUND

CANVAS_WIDTH = 768
CANVAS_HEIGHT = 512
CANVAS_DPI = 100
LINEWIDTH_POINTS = 0.65
LINE_ALPHA = 0.52


def fitted_y_limits(curves: np.ndarray, padding: float = 0.05) -> tuple[float, float]:
    """Fit the displayed extrema, padding each end by a fraction of their span."""
    lower, upper = float(np.min(curves)), float(np.max(curves))
    if not np.isfinite([lower, upper, padding]).all() or padding < 0:
        raise ValueError("Curve extrema and padding must be finite; padding must be nonnegative")
    span = upper - lower
    if span == 0:
        margin = max(padding, 0.05) * max(abs(lower), 1.0)
    else:
        margin = padding * span
    return lower - margin, upper + margin


def render_curves(
    dataset: Dataset,
    curves: np.ndarray,
    points: np.ndarray,
    y_limit: float | tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a data-only RGB canvas, fitting its limits unless supplied."""
    figure = Figure(
        figsize=(CANVAS_WIDTH / CANVAS_DPI, CANVAS_HEIGHT / CANVAS_DPI),
        dpi=CANVAS_DPI,
        facecolor=PLOT_BACKGROUND,
    )
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
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
    if y_limit is None:
        axis.set_ylim(*fitted_y_limits(curves))
    elif isinstance(y_limit, tuple):
        axis.set_ylim(*y_limit)
    else:
        axis.set_ylim(-y_limit, y_limit)
    axis.axis("off")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.asarray(rgba[..., :3], dtype=np.uint8).copy()
