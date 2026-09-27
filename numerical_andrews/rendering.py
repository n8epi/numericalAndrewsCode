"""Data-only Agg rasterization with the manuscript rendering defaults."""
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
VERTICAL_PADDING = 1.05

def render_curves(
    dataset: Dataset,
    curves: np.ndarray,
    points: np.ndarray,
    y_limit: float | tuple[float, float],
) -> np.ndarray:
    """Render a data-only RGB canvas under the fixed comparison protocol."""
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
    if isinstance(y_limit, tuple):
        axis.set_ylim(*y_limit)
    else:
        axis.set_ylim(-y_limit, y_limit)
    axis.axis("off")
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return np.asarray(rgba[..., :3], dtype=np.uint8).copy()

