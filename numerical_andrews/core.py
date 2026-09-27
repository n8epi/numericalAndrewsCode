"""PCA, example datasets, and interlaced finite-section Fourier synthesis."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np
from scipy.linalg import eigh_tridiagonal
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris

SQRT2 = np.sqrt(2.0)
PLOT_BACKGROUND = "#E5ECF6"

@dataclass(frozen=True)
class Dataset:
    key: str
    display_name: str
    values: np.ndarray
    labels: np.ndarray
    label_names: tuple[str, ...]
    colors: tuple[str, ...]
    scores: np.ndarray
    singular_values: np.ndarray


@dataclass(frozen=True)
class Mode:
    parity: str
    sector_rank: int
    eigenvalue: float
    coefficients: np.ndarray
    kinetic: float
    spectral: float

    @property
    def key(self) -> tuple[str, int]:
        return self.parity, self.sector_rank


def canonical_pca_sign_components(components: np.ndarray) -> np.ndarray:
    """Choose deterministic PCA signs using the largest loading in each column."""
    result = components.copy()
    for column in range(result.shape[1]):
        pivot = int(np.argmax(np.abs(result[:, column])))
        if result[pivot, column] < 0:
            result[:, column] *= -1.0
    return result


def pca_scores(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = np.asarray(values, dtype=float) - np.mean(values, axis=0)
    _, singular_values, right_vectors_t = np.linalg.svd(
        centered, full_matrices=False
    )
    components = canonical_pca_sign_components(right_vectors_t.T)
    scores = centered @ components
    return centered, scores, singular_values


def preprocessing_description(dataset_key: str) -> str:
    """Describe the feature representation used before the common PCA step."""
    if dataset_key == "bc_standardized":
        return "WDBC features centered and divided by population standard deviation (ddof=0), before PCA"
    if dataset_key == "diabetes":
        return (
            "scikit-learn load_diabetes(scaled=True) representation; "
            "recentered only; no additional rescaling"
        )
    return "raw scikit-learn features; feature-wise centering; no variance scaling"


def load_example_datasets(include_standardized: bool = False) -> list[Dataset]:
    iris = load_iris()
    cancer = load_breast_cancer()
    diabetes = load_diabetes(scaled=True)

    quantile_edges = np.quantile(diabetes.target, [0.25, 0.50, 0.75])
    diabetes_labels = np.digitize(diabetes.target, quantile_edges, right=True)

    specifications = [
        (
            "iris",
            "Iris",
            iris.data,
            iris.target,
            tuple(str(name) for name in iris.target_names),
            ("#E41A1C", "#377EB8", "#4DAF4A"),
        ),
        (
            "bc",
            "Breast cancer",
            cancer.data,
            cancer.target,
            tuple(str(name) for name in cancer.target_names),
            ("#E41A1C", "#377EB8"),
        ),
        (
            "diabetes",
            "Diabetes",
            diabetes.data,
            diabetes_labels,
            ("Q1", "Q2", "Q3", "Q4"),
            ("#E41A1C", "#FF9F40", "#4DAF4A", "#168AD1"),
        ),
    ]

    if include_standardized:
        standardized = (cancer.data - cancer.data.mean(axis=0)) / cancer.data.std(axis=0, ddof=0)
        specifications.append(("bc_standardized", "Breast cancer (standardized)",
                               standardized, cancer.target,
                               tuple(str(name) for name in cancer.target_names),
                               ("#E41A1C", "#377EB8")))

    result: list[Dataset] = []
    for key, display_name, values, labels, names, colors in specifications:
        centered, scores, singular_values = pca_scores(values)
        if np.linalg.matrix_rank(centered) < centered.shape[1]:
            raise RuntimeError(f"{display_name} is not full column rank after centering")
        result.append(
            Dataset(
                key=key,
                display_name=display_name,
                values=centered,
                labels=np.asarray(labels, dtype=int),
                label_names=names,
                colors=colors,
                scores=scores,
                singular_values=singular_values,
            )
        )
    return result


def _canonical_mode_sign(coefficients: np.ndarray, parity: str) -> np.ndarray:
    coefficients = coefficients.copy()
    if parity == "even":
        anchor = coefficients[0] + SQRT2 * np.sum(coefficients[1:])
    else:
        frequencies = np.arange(1, coefficients.size + 1, dtype=float)
        anchor = float(np.dot(frequencies, coefficients))
    scale = np.linalg.norm(coefficients)
    if abs(anchor) <= 100.0 * np.finfo(float).eps * scale:
        anchor = coefficients[int(np.argmax(np.abs(coefficients)))]
    if anchor < 0:
        coefficients *= -1.0
    return coefficients


def _sector_modes(
    alpha: float, cutoff: int, count: int, parity: str
) -> list[Mode]:
    if parity == "even":
        frequencies = np.arange(0, cutoff + 1, dtype=float)
        off_diagonal = -np.ones(cutoff, dtype=float)
        if cutoff >= 1:
            off_diagonal[0] = -SQRT2
    elif parity == "odd":
        frequencies = np.arange(1, cutoff + 1, dtype=float)
        off_diagonal = -np.ones(max(cutoff - 1, 0), dtype=float)
    else:
        raise ValueError(f"Unknown parity: {parity}")

    dimension = frequencies.size
    selected = min(count, dimension)
    if selected == 0:
        return []
    diagonal = alpha * frequencies**2 + 2.0
    eigenvalues, eigenvectors = eigh_tridiagonal(
        diagonal,
        off_diagonal,
        select="i",
        select_range=(0, selected - 1),
        check_finite=False,
        lapack_driver="stebz",
        tol=1.0e-14,
    )

    modes: list[Mode] = []
    for rank in range(selected):
        coefficients = _canonical_mode_sign(eigenvectors[:, rank], parity)
        kinetic = float(np.dot(frequencies**2, coefficients**2))
        eigenvalue = float(eigenvalues[rank])
        spectral = max(0.0, eigenvalue - alpha * kinetic)
        modes.append(
            Mode(
                parity=parity,
                sector_rank=rank,
                eigenvalue=eigenvalue,
                coefficients=coefficients,
                kinetic=kinetic,
                spectral=spectral,
            )
        )
    return modes


def spatial_spectral_modes(
    alpha: float, cutoff: int, count: int, extra_per_sector: int = 0
) -> tuple[list[Mode], dict[str, list[Mode]]]:
    """Select interlaced modes, optionally adding same-parity diagnostic neighbors."""
    if cutoff < 1 or count < 1 or count > 2 * cutoff + 1:
        raise ValueError("Require cutoff >= 1 and 1 <= count <= 2*cutoff+1")
    if extra_per_sector < 0:
        raise ValueError("extra_per_sector must be nonnegative")
    sectors = {
        "even": _sector_modes(alpha, cutoff, (count + 1) // 2 + extra_per_sector, "even"),
        "odd": _sector_modes(alpha, cutoff, count // 2 + extra_per_sector, "odd"),
    }
    # Strict interlacing fixes the order, including unresolved cross-parity gaps.
    ordered = [sectors["even" if i % 2 == 0 else "odd"][i // 2]
               for i in range(count)]
    return ordered, sectors


def evaluate_spatial_spectral_modes(
    modes: Iterable[Mode], points: np.ndarray
) -> np.ndarray:
    modes = list(modes)
    basis = np.empty((points.size, len(modes)), dtype=float)
    for column, mode in enumerate(modes):
        coefficients = mode.coefficients
        if mode.parity == "even":
            value = np.full(points.size, coefficients[0], dtype=float)
            if coefficients.size > 1:
                frequencies = np.arange(1, coefficients.size, dtype=float)
                value += SQRT2 * (
                    np.cos(2.0 * np.pi * np.outer(points, frequencies))
                    @ coefficients[1:]
                )
        else:
            frequencies = np.arange(1, coefficients.size + 1, dtype=float)
            value = SQRT2 * (
                np.sin(2.0 * np.pi * np.outer(points, frequencies))
                @ coefficients
            )
        basis[:, column] = value
    return basis


def evaluate_mmqv_modes(dimension: int, points: np.ndarray) -> np.ndarray:
    """The manuscript's standard ordering: 1, cos(1), sin(1), cos(2), ..."""
    basis = np.empty((points.size, dimension), dtype=float)
    basis[:, 0] = 1.0
    for column in range(1, dimension):
        frequency = (column + 1) // 2
        phase = 2.0 * np.pi * frequency * points
        if column % 2 == 1:
            basis[:, column] = SQRT2 * np.cos(phase)
        else:
            basis[:, column] = SQRT2 * np.sin(phase)
    return basis


def evaluate_limiting_aligned_mmqv_modes(
    dimension: int, points: np.ndarray
) -> np.ndarray:
    """MMQV basis aligned with the resolved alpha-to-infinity branches.

    With the strict-interlacing order used by ``spatial_spectral_modes``,
    the limiting order is 1, sin(1), cos(1), sin(2), cos(2), ... .  This is
    another valid MMQV minimizer and isolates finite-alpha deformation from
    the otherwise arbitrary ordering within each sine/cosine eigenspace.
    """
    basis = np.empty((points.size, dimension), dtype=float)
    basis[:, 0] = 1.0
    for column in range(1, dimension):
        frequency = (column + 1) // 2
        phase = 2.0 * np.pi * frequency * points
        if column % 2 == 1:
            basis[:, column] = SQRT2 * np.sin(phase)
        else:
            basis[:, column] = SQRT2 * np.cos(phase)
    return basis


def mmqv_frequencies(dimension: int) -> np.ndarray:
    frequencies = np.zeros(dimension, dtype=float)
    for column in range(1, dimension):
        frequencies[column] = (column + 1) // 2
    return frequencies

