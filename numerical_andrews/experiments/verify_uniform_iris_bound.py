"""Verify the Iris uniform truncation bound using exact rational arithmetic.

The certificate applies the explicit cutoff formula of Theorem 3 using only
alpha, the tolerance, and the dataset. It bounds the difference between exact
finite-eigenpair curves and the limiting embeddings. Eigensolver, PCA, and
curve-evaluation rounding errors are not included.

Run with ``python -m numerical_andrews.experiments.verify_uniform_iris_bound``.
"""
# Source repository: https://github.com/n8epi/numericalAndrewsCode

from __future__ import annotations

import argparse
from fractions import Fraction as F
import json
from pathlib import Path

from sklearn.datasets import load_iris

from numerical_andrews.paths import RESULTS


def _require(condition: bool, message: str) -> None:
    """Keep certificate checks active even when Python runs with optimization."""
    if not condition:
        raise ValueError(message)


def verify_uniform_iris_bound() -> dict:
    """Verify the explicit cutoff in Theorem 3 without spectral calculations."""
    alpha, dimension = F(1, 5), 4
    tolerance = F(1, 10**12)

    # Center the decimal Iris measurements exactly. Orthogonal PCA preserves
    # these norms, so no floating-point SVD is needed for the dataset bound.
    raw = [[F(str(value)) for value in row] for row in load_iris().data]
    _require(len(raw) == 150 and all(len(row) == 4 for row in raw), "Unexpected Iris shape")
    _require(
        all(value.denominator in (1, 2, 5, 10) for row in raw for value in row),
        "Iris measurements are not decimal tenths",
    )
    means = [sum(row[j] for row in raw) / len(raw) for j in range(dimension)]
    norms_squared = [
        sum((row[j] - means[j]) ** 2 for j in range(dimension)) for row in raw
    ]
    max_norm_squared = max(norms_squared)
    norm_upper = F(4)
    _require(max_norm_squared < norm_upper**2, "Centered Iris norm bound failed")

    # sqrt(5) < 47/21 gives 7*(d + 2 + 3/sqrt(alpha)) < 89.
    inverse_alpha_sqrt_upper = F(47, 21)
    _require(
        1 / alpha < inverse_alpha_sqrt_upper**2,
        "Inverse square-root bound failed",
    )
    leading_term_upper = 7 * (dimension + 2 + 3 * inverse_alpha_sqrt_upper)
    _require(leading_term_upper == 89, "Unexpected leading cutoff bound")
    dimension_sqrt = 2
    _require(dimension_sqrt**2 == dimension, "Dimension square-root check failed")
    prefactor_upper = dimension_sqrt * norm_upper

    # Evaluate ceil(log_4(1 + prefactor_upper/tolerance)) by exact integer
    # powers; no floating-point logarithm participates in the certificate.
    logarithm_argument_upper = 1 + prefactor_upper / tolerance
    added_frequencies = 0
    while 4**added_frequencies < logarithm_argument_upper:
        added_frequencies += 1
    _require(
        4**(added_frequencies - 1) < logarithm_argument_upper
        <= 4**added_frequencies,
        "Logarithmic ceiling check failed",
    )
    cutoff = int(leading_term_upper) + added_frequencies
    uniform_upper = prefactor_upper / 4**added_frequencies
    _require(cutoff == 111, "Unexpected sufficient cutoff")
    _require(uniform_upper == F(1, 2**41), "Unexpected uniform error fraction")
    _require(uniform_upper < F(5, 10**13) < tolerance, "Uniform error target was not met")

    return dict(
        alpha=alpha,
        dataset_dimension=dimension,
        requested_uniform_tolerance=tolerance,
        exact_centered_iris_max_norm_squared=max_norm_squared,
        centered_iris_norm_upper=norm_upper,
        logarithmic_cutoff_checks=dict(
            inverse_alpha_square_root_upper=inverse_alpha_sqrt_upper,
            inverse_alpha_square_root_upper_squared=inverse_alpha_sqrt_upper**2,
            leading_term_upper=leading_term_upper,
            dimension_square_root=dimension_sqrt,
            prefactor_upper=prefactor_upper,
            logarithm_argument_upper=logarithm_argument_upper,
            preceding_power_of_four=4**(added_frequencies - 1),
            enclosing_power_of_four=4**added_frequencies,
            added_frequencies=added_frequencies,
        ),
        symmetric_cutoff=cutoff,
        global_uniform_curve_error_upper=uniform_upper,
        global_uniform_curve_error_upper_display=float(uniform_upper),
        statement=(
            "Exact finite-eigenpair truncation error, before floating-point "
            "eigensolver, PCA, and curve-evaluation error. The cutoff uses "
            "rational upper bounds in the explicit formula of Theorem 3, "
            "with no spectral calculations or fitted parameters."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "iris_uniform_truncation_certificate.json",
        help="Destination for the exact rational certificate",
    )
    args = parser.parse_args()
    result = verify_uniform_iris_bound()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print("All rational checks passed.")
    print("Largest centered Iris norm squared:", result["exact_centered_iris_max_norm_squared"])
    print("Certified symmetric cutoff:", result["symmetric_cutoff"])
    print("Global cutoff uniform error upper (rounded display):", float(result["global_uniform_curve_error_upper"]))
    print("Certificate concerns exact finite eigenpairs; floating-point rounding error is separate.")
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
