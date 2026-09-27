"""Compatibility preprocessing for visual-clutter 1.0.7 and June 2007 FC code.

See FC_COMPATIBILITY.md and fc_reference_provenance.json for source hashes,
reference checks, and every departure from the unmodified Python package.
The historical RGB/XYZ scaling is deliberately preserved: this is NOT a
standalone standards-normalized CIELAB converter. uint8 RGB rasters and
three pyramid levels are the manuscript's supported input protocol.

This module changes only preprocessing on each fresh Vlc instance. It does
not modify installed files or enable the optional arithmetic acceleration.
"""
from __future__ import annotations
import numpy as np
from scipy.ndimage import convolve1d



def official_rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Reproduce the RGB2Lab.m conversion distributed with the FC code."""
    rgb = np.asarray(image, dtype=float) / 255.0
    rgb = np.where(
        rgb > 0.04045,
        ((rgb + 0.055) / 1.055) ** 2.4,
        rgb / 12.92,
    )
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    x_value = 0.4124 * red + 0.3576 * green + 0.1805 * blue
    y_value = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    z_value = 0.0193 * red + 0.1192 * green + 0.9505 * blue
    xyz = np.stack(
        [x_value / 95.047, y_value / 100.0, z_value / 108.883], axis=-1
    )
    transformed = np.where(
        xyz > 0.008856,
        np.cbrt(xyz),
        7.787 * xyz + 16.0 / 116.0,
    )
    return np.stack(
        [
            116.0 * transformed[..., 1] - 16.0,
            500.0 * (transformed[..., 0] - transformed[..., 1]),
            200.0 * (transformed[..., 1] - transformed[..., 2]),
        ],
        axis=-1,
    )


def official_gaussian_pyramid(channel: np.ndarray, levels: int) -> dict:
    """Build RRgaussianPyramid.m levels with whole-sample reflection."""
    kernel = np.array([0.05, 0.25, 0.40, 0.25, 0.05], dtype=float)
    pyramid: dict[tuple[int, int], np.ndarray] = {}
    current = np.asarray(channel, dtype=float)
    for level in range(levels):
        pyramid[(level, 0)] = current
        if level + 1 < levels:
            current = convolve1d(
                current, kernel, axis=1, mode="mirror"
            )[:, ::2]
            current = convolve1d(
                current, kernel, axis=0, mode="mirror"
            )[::2, :]
    return pyramid


def feature_congestion(image: np.ndarray) -> tuple[float, np.ndarray]:
    """Evaluate default-p Rosenholtz FC with official Lab/pyramid preprocessing."""
    try:
        from visual_clutter import Vlc
    except ImportError as error:
        raise ImportError("Install the optional Feature Congestion dependencies described in numerical_andrews/README.md") from error
    rgb = np.asarray(image, dtype=np.uint8)
    calculator = Vlc(
        rgb,
        numlevels=3,
        contrast_filt_sigma=1,
        contrast_pool_sigma=3,
        color_pool_sigma=3,
    )
    lab = official_rgb_to_lab(rgb)
    calculator.Lab = lab
    calculator.L = lab[..., 0]
    calculator.a = lab[..., 1]
    calculator.b = lab[..., 2]
    calculator.L_pyr = official_gaussian_pyramid(calculator.L, 3)
    calculator.a_pyr = official_gaussian_pyramid(calculator.a, 3)
    calculator.b_pyr = official_gaussian_pyramid(calculator.b, 3)
    scalar, local_map = calculator.getClutter_FC(p=1, pix=0)
    scalar = float(scalar)
    local_map = np.asarray(local_map, dtype=float)
    if not np.isfinite(scalar) or not np.all(np.isfinite(local_map)):
        raise FloatingPointError("Feature Congestion produced a non-finite value")
    return scalar, local_map



def upper_tail_mean(values: np.ndarray, tail_fraction: float) -> float:
    """Mean of the largest exact pixel fraction (an empirical upper CVaR)."""
    flat = np.ravel(np.asarray(values, dtype=float))
    count = max(1, int(np.ceil(tail_fraction * flat.size)))
    threshold_index = flat.size - count
    tail = np.partition(flat, threshold_index)[threshold_index:]
    return float(np.mean(tail))


def fc_profile(image: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    mean_fc, local_map = feature_congestion(image)
    return (
        {
            "mean_fc": float(mean_fc),
            "upper_5pct_mean_fc": upper_tail_mean(local_map, 0.05),
            "upper_1pct_mean_fc": upper_tail_mean(local_map, 0.01),
        },
        local_map,
    )

