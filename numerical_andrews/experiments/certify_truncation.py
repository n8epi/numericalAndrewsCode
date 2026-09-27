"""Validated infinite Jacobi eigenpair certificates (rational alpha).

Floating-point eigenvectors are candidate generators only. All spectral
brackets, Rayleigh quotients, residuals, and acceptance tests use Fraction.
The certificate applies to the exact rational alpha specified on the CLI.
"""
from fractions import Fraction as F
from math import lcm
import argparse
import json

import numpy as np
from scipy.linalg import eigh_tridiagonal


def matrix_data(alpha, sector, size):
    alpha = F(alpha)
    if alpha <= 0 or sector not in ('even', 'odd') or size < 2:
        raise ValueError('Require positive alpha, even/odd sector, and size >= 2')
    start = 0 if sector == 'even' else 1
    diagonal = [alpha * n * n + 2 for n in range(start, start + size)]
    off_squared = [F(1)] * (size - 1)
    if sector == 'even':
        off_squared[0] = F(2)
    return diagonal, off_squared


def sturm_count(diagonal, off_squared, point):
    """Number of eigenvalues strictly below point; exact integer recurrence.

    Zero determinants are omitted from sign variation. Interior zeros have
    opposite-sign neighbors for an irreducible Jacobi matrix. A final zero
    therefore excludes the eigenvalue at the endpoint.
    """
    point = F(point)
    scale = lcm(point.denominator, *(d.denominator for d in diagonal))
    previous, current = 0, 1
    last_sign, changes = 1, 0
    for i, d in enumerate(diagonal):
        coefficient = (d - point) * scale
        assert coefficient.denominator == 1
        coupling = off_squared[i-1] * scale * scale if i else F(0)
        assert coupling.denominator == 1
        following = coefficient.numerator * current - coupling.numerator * previous
        previous, current = current, following
        if current:
            sign = 1 if current > 0 else -1
            changes += sign != last_sign
            last_sign = sign
    return changes


def eigen_bracket(diagonal, off_squared, index, bits=48):
    """Closed rational enclosure of the one-based indexed eigenvalue."""
    if not 1 <= index <= len(diagonal) or bits < 1:
        raise ValueError('Invalid eigenvalue index or precision')
    # Every row's sum of off-diagonal magnitudes is < 3 in both sectors.
    if any(q <= 0 or q > 2 for q in off_squared):
        raise ValueError('Unsupported off-diagonal coefficients')
    lower, upper = min(diagonal) - 3, max(diagonal) + 3
    tolerance = F(1, 2**bits)
    while upper - lower > tolerance:
        midpoint = (lower + upper) / 2
        if sturm_count(diagonal, off_squared, midpoint) < index:
            lower = midpoint
        else:
            upper = midpoint
    return lower, upper


def rational_candidate(diagonal, off_squared, sector, index):
    _, vectors = eigh_tridiagonal(
        np.array([float(d) for d in diagonal]),
        -np.sqrt(np.array([float(q) for q in off_squared])),
        select='i', select_range=(index-1, index-1))
    z = vectors[:, 0].copy()
    if sector == 'even':
        z[1:] /= np.sqrt(2)
    return [F.from_float(float(x)) for x in z]


def rayleigh_residual(diagonal, sector, z):
    """Exact rho and squared full residual in the rational coordinates z."""
    z = [F(x) for x in z]
    if len(z) != len(diagonal) or not any(z):
        raise ValueError('Candidate must be nonzero and have the matrix dimension')
    weights = [1] * len(z) if sector == 'odd' else [1] + [2] * (len(z)-1)
    az = [d * x for d, x in zip(diagonal, z)]
    for i in range(len(z)-1):
        az[i] -= (2 if sector == 'even' and i == 0 else 1) * z[i+1]
        az[i+1] -= z[i]
    norm2 = sum(q*x*x for q, x in zip(weights, z))
    rho = sum(q*x*y for q, x, y in zip(weights, z, az)) / norm2
    internal = sum(q*(y-rho*x)**2 for q, x, y in zip(weights, z, az)) / norm2
    boundary = weights[-1]*z[-1]**2 / norm2
    return rho, internal + boundary, internal, boundary


def certify(alpha, sector, size, index, bits=48, candidate=None):
    """Return exact certificate, or raise ValueError when a test fails.

    The vector certificate concerns S*z / ||S*z|| with S=I (odd) or
    diag(1,sqrt(2),...) (even); z consists of exactly represented rationals.
    """
    alpha = F(alpha)
    if size < max(index+1, 2) or index < 1:
        raise ValueError('Require size >= max(index+1, 2)')
    diagonal, off_squared = matrix_data(alpha, sector, size)
    finite = eigen_bracket(diagonal, off_squared, index, bits)
    next_upper = eigen_bracket(diagonal, off_squared, index+1, bits)[1]
    last_frequency = size-1 if sector == 'even' else size
    tau = alpha*(last_frequency+1)**2
    barrier = next_upper + 1
    if barrier >= tau:
        raise ValueError('Tail bound insufficient: increase size')
    corrected = diagonal.copy()
    corrected[-1] -= 1/(tau-barrier)
    lower = eigen_bracket(corrected, off_squared, index, bits)[0]
    b = eigen_bracket(corrected, off_squared, index+1, bits)[0]
    a = F(-1) if index == 1 else eigen_bracket(diagonal, off_squared, index-1, bits)[1]
    z = rational_candidate(diagonal, off_squared, sector, index) if candidate is None else [F(x) for x in candidate]
    rho, r2, internal, boundary = rayleigh_residual(diagonal, sector, z)
    if not a < rho < b or not r2 < (rho-a)*(b-rho):
        raise ValueError('Residual/separation test failed: refine vector or increase size')
    interval = (max(lower, rho-r2/(b-rho)), min(finite[1], rho+r2/(rho-a)))
    if index == 1:
        interval = (interval[0], min(interval[1], rho))
    delta = min(rho-a, b-rho)
    return dict(alpha=alpha, sector=sector, size=size, index=index,
                tail_lower=tau, barrier=barrier,
                finite_enclosure=finite, corrected_lower=lower,
                lower_neighbor_upper_bound=a, upper_neighbor_lower_bound=b,
                rayleigh=rho, residual_squared=r2,
                internal_residual_squared=internal, boundary_residual_squared=boundary,
                eigenvalue_interval=interval,
                eigenvector_error_squared_upper=2*r2/delta**2,
                candidate_rational_coordinates=z)


def json_exact(obj):
    if isinstance(obj, F):
        return str(obj)
    if isinstance(obj, dict):
        return {key: json_exact(value) for key, value in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [json_exact(value) for value in obj]
    return obj


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--alpha', default='0.2', help='Exact decimal or rational')
    parser.add_argument('--sector', choices=['even', 'odd'], default='even')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--index', type=int, default=1)
    parser.add_argument('--bits', type=int, default=48)
    parser.add_argument('--output', help='Write exact rational certificate to JSON')
    args = parser.parse_args()
    result = certify(args.alpha, args.sector, args.size, args.index, args.bits)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(json.dumps(json_exact(result), indent=2)+'\n')
    lo, hi = result['eigenvalue_interval']
    print(f"Certified {args.sector} eigenpair {args.index}, alpha={result['alpha']}, N={args.size}")
    print(f"Eigenvalue center (rounded display): {float((lo+hi)/2):.16g}")
    print(f"Interval width (rounded display): {float(hi-lo):.3e}")
    print(f"Vector error bound (rounded display): {np.sqrt(float(result['eigenvector_error_squared_upper'])):.3e}")
    print('Exact rational interval and squared vector bound are stored in --output JSON.')
