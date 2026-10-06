"""Render reference Feature Congestion maps for the six selected plot rasters.

The saved maps are uncapped float64 arrays. Only the displayed colors saturate
at each dataset pair's pooled 99.5th percentile; scalar scores use every raw
pixel. This module reads completed search results and never reruns a search.
"""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from numerical_andrews.congestion import feature_congestion
from numerical_andrews.paths import RESULTS

KEYS = ("iris", "bc", "diabetes")
METHODS = ("mmqv", "mmssqv")
NAMES = {"iris": "Iris", "bc": "WDBC (raw)", "diabetes": "Diabetes"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def generate_heatmaps(output: Path) -> dict:
    output = Path(output).resolve()
    start = time.perf_counter()
    maps, rows = {}, []
    figure, axes = plt.subplots(3, 3, figsize=(7.4, 7.8),
        gridspec_kw={"width_ratios": [1., 1., .045]}, layout="constrained")
    for index, key in enumerate(KEYS):
        record_path = output / f"{key}.json"
        record = json.loads(record_path.read_text())
        if record["status"] != "complete":
            raise ValueError(f"The {key} search is incomplete")
        row = dict(dataset=key, alpha=record["alpha"],
                   record_sha256=sha256(record_path), methods={})
        for method in METHODS:
            raster_path = output / f"{key}_{method}.png"
            raster = np.asarray(Image.open(raster_path).convert("RGB"))
            mean, local_map = feature_congestion(raster)
            local_map = np.asarray(local_map, dtype=np.float64)
            if local_map.shape != raster.shape[:2]:
                raise ValueError("Congestion-map and raster dimensions differ")
            median = float(np.median(local_map))
            expected = record["methods"][method]["best"]
            np.testing.assert_allclose(median, expected["reference_median_fc"],
                                       rtol=0., atol=1e-8)
            np.testing.assert_allclose(mean, expected["reference_mean_fc"],
                                       rtol=0., atol=1e-8)
            maps[f"{key}_{method}"] = local_map
            row["methods"][method] = dict(
                raster_file=raster_path.name, raster_sha256=sha256(raster_path),
                shape=list(local_map.shape), dtype=str(local_map.dtype),
                raw_minimum=float(local_map.min()), raw_maximum=float(local_map.max()),
                raw_mean_fc=mean, raw_median_fc=median,
                y_limits=expected["y_limits"], curve_y_extrema=expected["curve_y_extrema"],
                expected_median_fc=expected["reference_median_fc"],
                median_validation_absolute_error=abs(median - expected["reference_median_fc"]),
                mean_validation_absolute_error=abs(mean - expected["reference_mean_fc"]))
        pooled = np.concatenate([maps[f"{key}_{method}"].ravel() for method in METHODS])
        cap = float(np.quantile(pooled, .995))
        row.update(display_minimum=0., display_maximum=cap,
                   pooled_quantile=.995, pooled_raw_maximum=float(pooled.max()),
                   pooled_fraction_above_display_cap=float(np.mean(pooled > cap)))
        for column, method in enumerate(METHODS):
            axis = axes[index, column]
            local_map = maps[f"{key}_{method}"]
            limits = record["methods"][method]["best"]["y_limits"]
            axis.set_box_aspect(local_map.shape[0] / local_map.shape[1])
            display = axis.imshow(local_map, cmap="magma", vmin=0., vmax=cap,
                origin="upper", extent=(-.5, .5, *limits),
                aspect="auto", interpolation="none", rasterized=True)
            axis.set_title("MMQV" if column == 0 else
                           rf"MMSSQV, $\alpha={record['alpha']:.4g}$", fontsize=9)
            axis.set_xticks([-.5, -.25, 0, .25, .5])
            axis.tick_params(labelsize=7)
            axis.set_xlabel(r"$t$", fontsize=8, labelpad=1)
            if column == 0:
                axis.set_ylabel(NAMES[key], fontsize=9)
            for spine in axis.spines.values():
                spine.set_linewidth(.5)
                spine.set_color("#777777")
        colorbar = figure.colorbar(display, cax=axes[index, 2], extend="max")
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label("Local FC", fontsize=8)
        rows.append(row)
        print(f"{key}: verified both medians; shared display cap {cap:.9g}", flush=True)
    maps_path = output / "heatmaps.npz"
    np.savez_compressed(maps_path, **maps)
    # Ensure the artifact retains every map value without display clipping.
    with np.load(maps_path) as saved:
        for key, local_map in maps.items():
            np.testing.assert_array_equal(saved[key], local_map)
    # Fixed-aspect maps can be shorter than their GridSpec cells. Freeze the
    # settled layout, then give each colorbar the adjacent map's vertical span.
    figure.canvas.draw()
    figure.set_layout_engine(None)
    for index in range(len(KEYS)):
        map_bounds = axes[index, 1].get_position()
        colorbar_bounds = axes[index, 2].get_position()
        axes[index, 2].set_position([colorbar_bounds.x0, map_bounds.y0,
                                    colorbar_bounds.width, map_bounds.height])
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"heatmaps.{suffix}", dpi=220,
                       bbox_inches="tight", pad_inches=.08)
    plt.close(figure)
    root = Path(__file__).resolve().parents[1]
    sources = ("experiments/render_median_heatmaps.py", "congestion.py")
    protocol_path = output / "protocol.json"
    metadata = dict(
        scorer="Original unaccelerated feature_congestion implementation",
        input_protocol_file=protocol_path.name,
        input_protocol_sha256=sha256(protocol_path),
        saved_maps_file=maps_path.name, saved_maps_sha256=sha256(maps_path),
        saved_maps="Six uncapped float64 arrays; lossless compressed NPZ",
        colormap="magma", scale="linear, common within each dataset row",
        display_cap="99.5th percentile pooled over both uncapped maps in each row",
        colorbar_extend="max", score_pooling="All raw pixels; no visual cap applied",
        orientation="origin=upper matches source rasters; each extent is [-0.5,0.5] x the individual plot's saved y_limits",
        vertical_range_protocol="Each source plot's own minimum/maximum curve heights plus 5% of its span at each end",
        rows=rows, elapsed_seconds=time.perf_counter() - start,
        source_sha256={name: sha256(root / name) for name in sources},
        versions={name: importlib.metadata.version(name) for name in
                  ("numpy", "matplotlib", "Pillow", "visual-clutter")})
    (output / "heatmaps_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path,
                        default=RESULTS / "median_comparison")
    args = parser.parse_args()
    generate_heatmaps(args.output_directory)


if __name__ == "__main__":
    main()
