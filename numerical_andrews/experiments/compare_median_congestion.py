"""Compare sampled MMQV and MMSSQV minimizers by median pixel congestion.

This experiment is self-contained: the three datasets come from scikit-learn,
and no previous searches or figures are inputs. Alpha is selected by numerical
localization before congestion is evaluated. Each candidate uses its own
minimum and maximum curve heights plus 5% of their span on each end. The reported
Iris MMSSQV pool covers all 16 sign patterns; the other pools are sampled.
"""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

# Avoid multiplying numerical-library threads by the number of dataset workers.
THREAD_VARIABLES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS")
for variable in THREAD_VARIABLES:
    os.environ.setdefault(variable, "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.spatial.distance import pdist

from numerical_andrews.congestion import feature_congestion
from numerical_andrews.core import (
    PLOT_BACKGROUND, evaluate_mmqv_modes, evaluate_spatial_spectral_modes,
    load_example_datasets, preprocessing_description, spatial_spectral_modes,
)
from numerical_andrews.mmqv_family import mmqv_transform
from numerical_andrews.paths import RESULTS
from numerical_andrews.rendering import (
    CANVAS_WIDTH, CANVAS_HEIGHT, CANVAS_DPI, LINEWIDTH_POINTS, LINE_ALPHA,
    fitted_y_limits, render_curves,
)
from numerical_andrews.union_fc_acceleration import union_accelerated_fc

KEYS = ("iris", "bc", "diabetes")
NAMES = {"iris": "Iris", "bc": "WDBC (raw)", "diabetes": "Diabetes"}
SEEDS = {"iris": (202610031, 202610032), "bc": (202610033, 202610034),
         "diabetes": (202610035, 202610036)}
ALPHAS = sorted(set(np.logspace(-3, 2, 16).tolist() + [.2]))
POOL_SIZE = 256
SCORE_TOLERANCE = 1e-8
PAIRWISE_ENERGY_TARGET = .75


def dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def candidate_pools(key: str, dimension: int) -> dict[str, list[dict]]:
    angles_count, pairs = dimension // 2, (dimension - 1) // 2
    mmqv = [dict(index=0, source="cosine_first", angles=[0.] * angles_count,
                 reflections=[1] * pairs, constant_sign=1),
            dict(index=1, source="sine_first", angles=[float(np.pi / 2)] * angles_count,
                 reflections=[-1] * pairs, constant_sign=1)]
    rng = np.random.default_rng(SEEDS[key][0])
    for index in range(2, POOL_SIZE):
        mmqv.append(dict(index=index, source="random", angles=rng.uniform(
            0., 2 * np.pi, angles_count).tolist(),
            reflections=rng.choice([-1, 1], pairs).tolist(),
            constant_sign=int(rng.choice([-1, 1]))))
    rng = np.random.default_rng(SEEDS[key][1])
    mmssqv = [dict(index=index, source="random_signs",
                  signs=rng.choice([-1, 1], dimension).tolist())
              for index in range(POOL_SIZE)]
    return {"mmqv": mmqv, "mmssqv": mmssqv}


def sign_coverage(candidates: list[dict], dimension: int) -> dict:
    """Record actual sign-family coverage, independently of duplicate scores."""
    unique = len({tuple(row["signs"]) for row in candidates})
    return dict(unique_sign_patterns=unique, total_sign_patterns=2**dimension,
                exhaustive=unique == 2**dimension,
                scope="Fixed alpha, finite cutoff, and stated rendering/scoring protocol")


def pairwise_energy_check(dataset, alpha: float, cutoff: int,
                          quadrature_order: int) -> dict:
    """Integrate the dataset's normalized aggregate pairwise energy density.

    Centered PCA scores have diagonal Gram matrix diag(sigma_i**2). Thus the
    sum over all unordered pairwise squared curve differences is proportional
    to sum_i sigma_i**2 * phi_i(t)**2, independently of the eigenvector signs.
    """
    modes, _ = spatial_spectral_modes(alpha, cutoff, dataset.values.shape[1])
    nodes, weights = roots_legendre(quadrature_order)
    basis = evaluate_spatial_spectral_modes(modes, .25 * nodes)
    per_mode_inside = .25 * np.sum(weights[:, None] * basis**2, axis=0)
    variance_weights = dataset.singular_values**2
    variance_weights /= np.sum(variance_weights)
    inside = float(np.dot(variance_weights, per_mode_inside))
    return dict(alpha=float(alpha), cutoff=cutoff,
                quadrature_order=quadrature_order, window=[-.25, .25],
                aggregate_pairwise_inside_fraction=inside,
                aggregate_pairwise_outside_fraction=1. - inside,
                per_mode_inside_energy=per_mode_inside.tolist(),
                normalized_squared_singular_values=variance_weights.tolist())


def select_alpha(dataset) -> tuple[float, dict]:
    screening = [pairwise_energy_check(dataset, alpha, 256, 256)
                 for alpha in ALPHAS]
    brackets = [dict(lower_alpha=left["alpha"], upper_alpha=right["alpha"],
                     lower_inside_fraction=left["aggregate_pairwise_inside_fraction"],
                     upper_inside_fraction=right["aggregate_pairwise_inside_fraction"])
                for left, right in zip(screening[:-1], screening[1:])
                if left["aggregate_pairwise_inside_fraction"] >= PAIRWISE_ENERGY_TARGET
                and right["aggregate_pairwise_inside_fraction"] <= PAIRWISE_ENERGY_TARGET
                and left["aggregate_pairwise_inside_fraction"] > right["aggregate_pairwise_inside_fraction"]]
    if not brackets:
        raise RuntimeError(f"No downward 75% energy bracket found for {dataset.key}")
    selected = brackets[-1]
    evaluations = []

    def residual(log_alpha):
        row = pairwise_energy_check(dataset, float(np.exp(log_alpha)), 256, 256)
        evaluations.append(row)
        return row["aggregate_pairwise_inside_fraction"] - PAIRWISE_ENERGY_TARGET

    root = brentq(residual, np.log(selected["lower_alpha"]),
                  np.log(selected["upper_alpha"]), xtol=1e-12, rtol=1e-12)
    alpha = float(np.exp(root))
    selected_check = pairwise_energy_check(dataset, alpha, 256, 256)
    verification = pairwise_energy_check(dataset, alpha, 512, 512)
    if abs(verification["aggregate_pairwise_inside_fraction"] - PAIRWISE_ENERGY_TARGET) > 1e-8:
        raise RuntimeError(f"The refined 75% energy check fails for {dataset.key}")
    return alpha, dict(
        rule="Target 75% aggregate pairwise squared L2 separation in the central half; refine the highest-alpha downward grid bracket in log(alpha)",
        target_inside_fraction=PAIRWISE_ENERGY_TARGET, window=[-.25, .25],
        grid=ALPHAS, screening=screening, downward_brackets=brackets,
        selected_bracket=selected, root_solver="scipy.optimize.brentq in log(alpha)",
        root_evaluations=evaluations, selected_check=selected_check,
        refined_verification=verification, monotonicity_assumed=False,
        status="Floating-point convergence check, not an interval certificate")


def validate_pairwise_energy(dataset, modes, pools: dict, selection: dict) -> dict:
    """Check the weighted identity against all actual observation pairs."""
    nodes, weights = roots_legendre(256)
    basis = evaluate_spatial_spectral_modes(modes, .25 * nodes)
    signs = np.asarray(pools["mmssqv"][0]["signs"])
    curves = (dataset.scores * signs[None, :]) @ basis.T
    numerator = float(np.sum(pdist(curves * np.sqrt(.25 * weights)[None, :], "sqeuclidean")))
    denominator = float(np.sum(pdist(dataset.values, "sqeuclidean")))
    direct_fraction = numerator / denominator
    weighted_fraction = selection["selected_check"]["aggregate_pairwise_inside_fraction"]
    np.testing.assert_allclose(direct_fraction, weighted_fraction, rtol=0, atol=1e-11)
    dimension = dataset.values.shape[1]
    trig = evaluate_mmqv_modes(dimension + (dimension % 2 == 0), .25 * nodes)
    variance_weights = dataset.singular_values**2
    variance_weights /= np.sum(variance_weights)
    mmqv_checks = []
    for index in (0, 1, 2, 127, 255):
        row = pools["mmqv"][index]
        transform = mmqv_transform(dimension, row["angles"], row["reflections"], row["constant_sign"])
        functions = trig @ transform
        fraction = float(np.dot(variance_weights, .25 * np.sum(weights[:, None] * functions**2, axis=0)))
        np.testing.assert_allclose(fraction, .5, rtol=0, atol=1e-11)
        mmqv_checks.append(dict(candidate_index=index, aggregate_pairwise_inside_fraction=fraction))
    return dict(direct_all_pairs_inside_fraction=direct_fraction,
                weighted_identity_absolute_error=abs(direct_fraction - weighted_fraction),
                number_of_unordered_pairs=dataset.values.shape[0] * (dataset.values.shape[0] - 1) // 2,
                mmqv_exact_inside_fraction=.5,
                mmqv_reason="Each unit trigonometric mode has one-half squared norm on an interval of length one-half; centered PCA cancels cross terms",
                mmqv_numerical_checks=mmqv_checks)


def run_dataset(key: str, destination: str) -> dict:
    start = time.perf_counter()
    output = Path(destination)
    dataset = next(d for d in load_example_datasets() if d.key == key)
    dimension = dataset.values.shape[1]
    alpha, alpha_selection = select_alpha(dataset)
    print(f"{key}: alpha={alpha:.12g}; aggregate pairwise central-half fraction "
          f"{alpha_selection['refined_verification']['aggregate_pairwise_inside_fraction']:.12g}", flush=True)
    points = np.linspace(-.5, .5, 4097)
    modes, _ = spatial_spectral_modes(alpha, 256, dimension)
    reference_modes, _ = spatial_spectral_modes(alpha, 512, dimension)
    basis = evaluate_spatial_spectral_modes(modes, points)
    reference_basis = evaluate_spatial_spectral_modes(reference_modes, points)
    mode_l2_errors = []
    for current, reference in zip(modes, reference_modes):
        padded = np.zeros_like(reference.coefficients)
        padded[:current.coefficients.size] = current.coefficients
        mode_l2_errors.append(float(min(np.linalg.norm(padded - reference.coefficients),
                                        np.linalg.norm(padded + reference.coefficients))))
    trig = evaluate_mmqv_modes(dimension + (dimension % 2 == 0), points)
    pools = candidate_pools(key, dimension)
    energy_validation = validate_pairwise_energy(dataset, modes, pools, alpha_selection)

    def curves(method: str, candidate: dict) -> np.ndarray:
        if method == "mmqv":
            transform = mmqv_transform(dimension, candidate["angles"],
                candidate["reflections"], candidate["constant_sign"])
            return (dataset.scores @ transform.T) @ trig.T
        return (dataset.scores * np.asarray(candidate["signs"])[None, :]) @ basis.T

    background = np.empty((CANVAS_HEIGHT, CANVAS_WIDTH, 3), dtype=np.uint8)
    background[:] = tuple(int(PLOT_BACKGROUND[i:i + 2], 16) for i in (1, 3, 5))
    with union_accelerated_fc():
        _, background_map = feature_congestion(background)
    background_median = float(np.median(background_map))
    result = dict(
        dataset=key, display_name=NAMES[key], observations=dataset.values.shape[0],
        dimension=dimension, preprocessing=preprocessing_description(key),
        data_sha256=digest(dataset.values), scores_sha256=digest(dataset.scores),
        singular_values=dataset.singular_values.tolist(), seeds=dict(zip(
            ("mmqv", "mmssqv"), SEEDS[key])), alpha=alpha,
        alpha_selection=alpha_selection,
        pairwise_energy_density=dict(points=points.tolist(),
            normalized_density=(basis**2 @ (dataset.singular_values**2 /
                                          np.sum(dataset.singular_values**2))).tolist()),
        pairwise_energy_validation=energy_validation,
        numerics=dict(cutoff=256, comparison_cutoff=512, grid_points=points.size,
            maximum_mode_l2_change=max(mode_l2_errors),
            maximum_eigenvalue_change=max(abs(a.eigenvalue - b.eigenvalue) for a, b in zip(modes, reference_modes)),
            maximum_sampled_basis_change=float(np.max(np.abs(basis - reference_basis))),
            eigenvalues=[m.eigenvalue for m in modes]),
        range_protocol="Each candidate uses its own minimum/maximum curve heights, plus 5% of its span at each end",
        padding_fraction_of_span=.05,
        background_median_fc=background_median, score_tie_tolerance=SCORE_TOLERANCE,
        global_optimality_claim=False,
        mmssqv_sign_coverage=sign_coverage(pools["mmssqv"], dimension),
        methods={}, status="running")
    dump(output / f"{key}.json", result)
    for method, candidates in pools.items():
        scoring_start = time.perf_counter()
        cache = {}
        for row in candidates:
            signature = tuple(row["signs"]) if method == "mmssqv" else row["index"]
            if signature in cache:
                row.update(cache[signature])
                row["reused_duplicate"] = True
                continue
            ordinates = curves(method, row)
            limits = fitted_y_limits(ordinates)
            raster = render_curves(dataset, ordinates, points, limits)
            with union_accelerated_fc():
                mean, local_map = feature_congestion(raster)
            score = dict(median_fc=float(np.median(local_map)), mean_fc=mean,
                         y_limits=list(limits),
                         curve_y_extrema=[float(ordinates.min()), float(ordinates.max())])
            row.update(score)
            cache[signature] = score
            if len(cache) % 64 == 0:
                print(f"{key}: {method} scored {len(cache)} unique candidates", flush=True)
        best = min(candidates, key=lambda row: row["median_fc"]).copy()
        winner_curves = curves(method, best)
        np.testing.assert_array_equal(fitted_y_limits(winner_curves), best["y_limits"])
        winner_raster = render_curves(dataset, winner_curves, points, tuple(best["y_limits"]))
        reference_mean, reference_map = feature_congestion(winner_raster)
        with union_accelerated_fc():
            accelerated_mean, accelerated_map = feature_congestion(winner_raster)
        np.testing.assert_allclose(accelerated_map, reference_map, rtol=1e-7, atol=1e-8)
        reference_median = float(np.median(reference_map))
        np.testing.assert_allclose(best["median_fc"], reference_median, rtol=1e-7, atol=1e-8)
        tied = [row["index"] for row in candidates
                if abs(row["median_fc"] - best["median_fc"]) <= SCORE_TOLERANCE]
        best.update(reference_median_fc=reference_median, reference_mean_fc=reference_mean,
                    at_background_floor=abs(reference_median - background_median) <= SCORE_TOLERANCE)
        Image.fromarray(winner_raster).save(output / f"{key}_{method}.png")
        if method == "mmqv":
            transform = mmqv_transform(dimension, best["angles"], best["reflections"], best["constant_sign"])
            gram_error = float(np.max(np.abs(transform.T @ transform - np.eye(dimension))))
        else:
            gram_error = max(float(np.max(np.abs(np.column_stack([m.coefficients for m in modes if m.parity == parity]).T @
                np.column_stack([m.coefficients for m in modes if m.parity == parity]) -
                np.eye(sum(m.parity == parity for m in modes))))) for parity in ("even", "odd"))
        assert gram_error < 1e-10
        result["methods"][method] = dict(candidates=candidates, pool_size=len(candidates),
            unique_evaluations=len(cache), best=best, tied_best_candidate_indices=tied,
            gram_error=gram_error, elapsed_seconds=time.perf_counter() - scoring_start,
            reference_validation=dict(maximum_map_absolute_error=float(np.max(np.abs(accelerated_map - reference_map))),
                median_absolute_error=abs(reference_median - best["median_fc"]),
                mean_absolute_error=abs(reference_mean - accelerated_mean)))
        dump(output / f"{key}.json", result)
        print(f"{key}: {method} best median={reference_median:.12g}; tied={len(tied)}; "
              f"background_floor={best['at_background_floor']}", flush=True)
    result["status"] = "complete"
    result["elapsed_seconds"] = time.perf_counter() - start
    result["between_method_tie"] = abs(result["methods"]["mmssqv"]["best"]["reference_median_fc"] -
        result["methods"]["mmqv"]["best"]["reference_median_fc"]) <= SCORE_TOLERANCE
    dump(output / f"{key}.json", result)
    return result


def plot_pairs(records: list[dict], output: Path) -> None:
    datasets = {d.key: d for d in load_example_datasets()}
    figure, axes = plt.subplots(3, 2, figsize=(7.4, 9.3), squeeze=False)
    for row, record in enumerate(records):
        dataset = datasets[record["dataset"]]
        for column, method in enumerate(("mmqv", "mmssqv")):
            axis = axes[row, column]
            axis.set_box_aspect(CANVAS_HEIGHT / CANVAS_WIDTH)
            image = np.asarray(Image.open(output / f"{record['dataset']}_{method}.png"))
            limits = record["methods"][method]["best"]["y_limits"]
            axis.imshow(image, extent=(-.5, .5, *limits), aspect="auto", interpolation="none")
            axis.set_xticks([-.5, -.25, 0, .25, .5])
            axis.tick_params(labelsize=7)
            axis.set_xlabel(r"$t$", fontsize=8, labelpad=1)
            axis.set_title("MMQV" if column == 0 else rf"MMSSQV, $\alpha={record['alpha']:.4g}$", fontsize=9)
            if column == 0:
                axis.set_ylabel(NAMES[record["dataset"]], fontsize=9)
            handles = [Line2D([0], [0], color=color, lw=2, label=label)
                       for label, color in zip(dataset.label_names, dataset.colors)]
            axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.23),
                        ncol=len(handles), frameon=False, fontsize=6.8,
                        handlelength=1.4, columnspacing=1.)
            for spine in axis.spines.values():
                spine.set_color("#777777")
                spine.set_linewidth(.5)
    figure.subplots_adjust(left=.10, right=.985, top=.96, bottom=.07, hspace=.60, wspace=.24)
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"examples.{suffix}", dpi=220, bbox_inches="tight", pad_inches=.08)
    plt.close(figure)


def write_table(records: list[dict], output: Path, worst_rows: list[dict]) -> None:
    worst_by_dataset = {row["dataset"]: row for row in worst_rows}
    lines = [r"\begin{table}[ht]", r"\centering",
        r"\caption{Median pixel Feature Congestion for the best sampled MMQV plot and the best and worst sampled MMSSQV sign choices. Every plot uses its own curve minimum and maximum, with $5\%$ of that span added at each end. The background score is approximately $1.248283$.}",
        r"\label{tabMedianCongestion}", r"\begin{tabular}{lrrr}", r"\toprule",
        r"Dataset & MMQV & \multicolumn{2}{c}{MMSSQV} \\",
        r"\cmidrule(lr){2-2}\cmidrule(lr){3-4}",
        r" & Best & Best & Worst \\", r"\midrule"]
    for row in records:
        left = row["methods"]["mmqv"]["best"]["reference_median_fc"]
        right = row["methods"]["mmssqv"]["best"]["reference_median_fc"]
        worst = worst_by_dataset[row["dataset"]]["worst_mmssqv_median_fc"]
        lines.append(f"{NAMES[row['dataset']]} & {left:.6f} & {right:.6f} & {worst:.6f} " + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output / "table.tex").write_text("\n".join(lines) + "\n")


def write_alpha_table(records: list[dict], output: Path) -> None:
    """Generate Table 1 from the selected parameters, without manual transcription."""
    lines = [r"\begin{table}[ht]",
        r"\caption{Spatial-spectral weights selected for moderate central emphasis.",
        r"The selected parameters place $75\%$ of aggregate pairwise squared",
        r"separation in the central half, compared with $50\%$ for MMQV.",
        r"Values are rounded for display; computations use full precision.}",
        r"\label{tabSelectedAlpha}", r"\centering", r"\begin{tabular}{lrr}",
        r"\toprule", r"Dataset & $d$ & Selected $\alpha$\\", r"\midrule"]
    names = {"iris": "Iris", "bc": "Breast cancer", "diabetes": "Diabetes"}
    for row in records:
        lines.append(f"{names[row['dataset']]} & {row['dimension']} & ${row['alpha']:.6g}$" + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    (output / "selected_alpha_table.tex").write_text("\n".join(lines) + "\n")


def refresh_worst_comparison(output: Path) -> dict:
    """Validate saved sign-pool maxima without changing the original search."""
    start = time.perf_counter()
    output = Path(output)
    datasets = {dataset.key: dataset for dataset in load_example_datasets()}
    records, rows = [], []
    for key in KEYS:
        record_path = output / f"{key}.json"
        record = json.loads(record_path.read_text())
        if record["status"] != "complete":
            raise ValueError(f"The {key} search is incomplete")
        dataset = datasets[key]
        assert digest(dataset.values) == record["data_sha256"]
        assert digest(dataset.scores) == record["scores_sha256"]
        candidates = record["methods"]["mmssqv"]["candidates"]
        worst = max(candidates, key=lambda candidate: candidate["median_fc"])
        modes, _ = spatial_spectral_modes(record["alpha"], record["numerics"]["cutoff"],
                                         dataset.values.shape[1])
        points = np.linspace(-.5, .5, record["numerics"]["grid_points"])
        basis = evaluate_spatial_spectral_modes(modes, points)
        curves = (dataset.scores * np.asarray(worst["signs"])[None, :]) @ basis.T
        np.testing.assert_array_equal(fitted_y_limits(curves), worst["y_limits"])
        raster = render_curves(dataset, curves, points, tuple(worst["y_limits"]))
        reference_mean, local_map = feature_congestion(raster)
        reference_median = float(np.median(local_map))
        np.testing.assert_allclose(reference_median, worst["median_fc"], rtol=0, atol=1e-8)
        best_mmqv = record["methods"]["mmqv"]["best"]["reference_median_fc"]
        row = dict(dataset=key, alpha=record["alpha"],
            input_record_sha256=hashlib.sha256(record_path.read_bytes()).hexdigest(),
            mmssqv_sample_count=len(candidates),
            mmssqv_sign_coverage=sign_coverage(candidates, dataset.values.shape[1]),
            worst_candidate=worst,
            tied_worst_candidate_indices=[candidate["index"] for candidate in candidates
                if abs(candidate["median_fc"] - worst["median_fc"]) <= SCORE_TOLERANCE],
            worst_mmssqv_median_fc=reference_median,
            worst_mmssqv_mean_fc=reference_mean,
            best_mmqv_median_fc=best_mmqv,
            best_mmssqv_median_fc=record["methods"]["mmssqv"]["best"]["reference_median_fc"],
            worst_minus_best_mmqv=reference_median - best_mmqv,
            worst_to_best_mmqv_percent_change=100 * (reference_median / best_mmqv - 1),
            validation=dict(scorer="Original unaccelerated feature_congestion",
                median_absolute_error=abs(reference_median - worst["median_fc"]),
                mean_absolute_error=abs(reference_mean - worst["mean_fc"]),
                reconstructed_raster_rgb_sha256=digest(raster),
                raster_shape=list(raster.shape), y_limits=worst["y_limits"],
                curve_y_extrema=worst["curve_y_extrema"]))
        records.append(record)
        rows.append(row)
        print(f"{key}: worst sampled MMSSQV median={reference_median:.15g}; "
              f"change from best MMQV={row['worst_to_best_mmqv_percent_change']:+.8g}%", flush=True)
    root = Path(__file__).resolve().parents[1]
    sources = ("experiments/compare_median_congestion.py", "core.py", "rendering.py", "congestion.py")
    metadata = dict(
        scope="Maximum median pixel FC over the existing 256 sampled MMSSQV sign choices per dataset; not a bound over unsampled signs",
        selection="Maximum saved median_fc; first saved candidate breaks exact ties",
        score_tie_tolerance=SCORE_TOLERANCE,
        input_protocol_sha256=hashlib.sha256((output / "protocol.json").read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        versions={name: importlib.metadata.version(name) for name in
                  ("numpy", "scipy", "scikit-learn", "matplotlib", "Pillow", "visual-clutter")},
        elapsed_seconds=time.perf_counter() - start, rows=rows)
    dump(output / "worst_summary.json", metadata)
    write_table(records, output, rows)
    write_alpha_table(records, output)
    return metadata


def preview_alpha_choices(output: Path) -> None:
    """Preview the first sampled sign vector before any congestion search."""
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), squeeze=False)
    records = []
    datasets = {dataset.key: dataset for dataset in load_example_datasets()}
    for column, key in enumerate(KEYS):
        dataset = datasets[key]
        alpha, selection = select_alpha(dataset)
        points = np.linspace(-.5, .5, 4097)
        modes, _ = spatial_spectral_modes(alpha, 256, dataset.values.shape[1])
        basis = evaluate_spatial_spectral_modes(modes, points)
        signs = candidate_pools(key, dataset.values.shape[1])["mmssqv"][0]["signs"]
        curves = (dataset.scores * np.asarray(signs)[None, :]) @ basis.T
        limits = fitted_y_limits(curves)
        raster = render_curves(dataset, curves, points, limits)
        Image.fromarray(raster).save(output / f"{key}_preview.png")
        axis = axes[0, column]
        axis.set_box_aspect(CANVAS_HEIGHT / CANVAS_WIDTH)
        axis.imshow(raster, extent=(-.5, .5, *limits), aspect="auto", interpolation="none")
        axis.set_title(rf"{NAMES[key]}: MMSSQV, $\alpha={alpha:.5g}$", fontsize=10)
        axis.set_xticks([-.5, -.25, 0, .25, .5])
        axis.set_xlabel(r"$t$", fontsize=9)
        axis.tick_params(labelsize=8)
        handles = [Line2D([0], [0], color=color, lw=2, label=label)
                   for label, color in zip(dataset.label_names, dataset.colors)]
        axis.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.21),
                    ncol=len(handles), frameon=False, fontsize=7)
        row = dict(dataset=key, alpha=alpha, alpha_selection=selection,
                   preview_signs=signs, preview_y_limits=list(limits),
                   preview_curve_y_extrema=[float(curves.min()), float(curves.max())],
                   preview_range_protocol="Each plot fits its own minimum/maximum curve heights, with 5% of its span added at each end",
                   pairwise_energy_density=dict(points=points.tolist(),
                       normalized_density=(basis**2 @ (dataset.singular_values**2 /
                                                     np.sum(dataset.singular_values**2))).tolist()))
        records.append(row)
        print(f"{key}: alpha={alpha:.15g}, central-half energy="
              f"{selection['refined_verification']['aggregate_pairwise_inside_fraction']:.15g}, "
              f"downward_brackets={len(selection['downward_brackets'])}", flush=True)
    figure.subplots_adjust(left=.055, right=.985, bottom=.23, top=.91, wspace=.26)
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"alpha_preview.{suffix}", dpi=180,
                       bbox_inches="tight", pad_inches=.08)
    plt.close(figure)
    dump(output / "alpha_preview.json", records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=RESULTS / "median_comparison")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--preview-only", action="store_true",
                        help="Select alpha and render the first random sign draw without scoring candidates")
    parser.add_argument("--refresh-worst-only", action="store_true",
                        help="Validate worst saved MMSSQV candidates and update the table without rerunning searches")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    output = args.output_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.preview_only and args.refresh_worst_only:
        parser.error("Choose only one of --preview-only and --refresh-worst-only")
    if args.refresh_worst_only:
        refresh_worst_comparison(output)
        return
    if args.preview_only:
        preview_alpha_choices(output)
        return
    start = time.perf_counter()
    completed = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_dataset, key, str(output)): key for key in KEYS}
        for future in as_completed(futures):
            record = future.result()
            completed[record["dataset"]] = record
    records = [completed[key] for key in KEYS]
    plot_pairs(records, output)
    source_root = Path(__file__).resolve().parents[1]
    sources = ["experiments/compare_median_congestion.py",
               "mmqv_family.py", "core.py", "congestion.py",
               "rendering.py", "union_fc_acceleration.py", "fast_feature_congestion.py"]
    metadata = dict(objective="Median over every pixel of the uncapped local Feature Congestion map",
        alpha_selection="75% aggregate pairwise squared L2 separation in the central half; refine the highest-alpha downward grid bracket before FC",
        candidate_pool_size_per_method=POOL_SIZE,
        mmqv_sampling="Two named bases and 254 independent uniform phases, O(2) reflections, and constant signs",
        mmssqv_sampling="256 independent uniform {-1,+1} sign vectors; duplicate evaluations cached",
        range_protocol="Each candidate uses its own minimum/maximum curve heights, plus 5% of its span at each end",
        padding_fraction_of_span=.05,
        raster=dict(width=CANVAS_WIDTH, height=CANVAS_HEIGHT, dpi=CANVAS_DPI,
            line_width_points=LINEWIDTH_POINTS, line_alpha=LINE_ALPHA,
            decorations="none in scored rasters", background=PLOT_BACKGROUND),
        seeds=SEEDS, score_tie_tolerance=SCORE_TOLERANCE, global_optimality_claim=False,
        mmssqv_sign_coverage={row["dataset"]: row["mmssqv_sign_coverage"] for row in records},
        elapsed_seconds=time.perf_counter() - start, workers=args.workers,
        numerical_library_threads={name: os.environ.get(name) for name in THREAD_VARIABLES},
        platform=platform.platform(), machine=platform.machine(), python=platform.python_version(),
        package_versions={name: importlib.metadata.version(name) for name in (
            "numpy", "scipy", "scikit-learn", "matplotlib", "Pillow", "visual-clutter")},
        source_sha256={name: hashlib.sha256((source_root / name).read_bytes()).hexdigest() for name in sources})
    dump(output / "protocol.json", metadata)
    summary = [dict(dataset=row["dataset"], alpha=row["alpha"],
                    aggregate_pairwise_inside_fraction=row["alpha_selection"]["refined_verification"]["aggregate_pairwise_inside_fraction"],
                    mmqv_y_limits=row["methods"]["mmqv"]["best"]["y_limits"],
                    mmssqv_y_limits=row["methods"]["mmssqv"]["best"]["y_limits"],
                    mmqv_median_fc=row["methods"]["mmqv"]["best"]["reference_median_fc"],
                    mmssqv_median_fc=row["methods"]["mmssqv"]["best"]["reference_median_fc"],
                    between_method_tie=row["between_method_tie"],
                    maximum_mode_l2_change=row["numerics"]["maximum_mode_l2_change"],
                    maximum_eigenvalue_change=row["numerics"]["maximum_eigenvalue_change"],
                    maximum_sampled_basis_change=row["numerics"]["maximum_sampled_basis_change"],
                    elapsed_seconds=row["elapsed_seconds"])
               for row in records]
    dump(output / "summary.json", summary)
    refresh_worst_comparison(output)
    from numerical_andrews.experiments.render_median_heatmaps import generate_heatmaps
    generate_heatmaps(output)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
