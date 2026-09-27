"""Direct numerical localization checks; no monotonicity in alpha assumed.

These floating-point calculations are convergence diagnostics, not validated
infinite-dimensional certificates. Integrate the outside region directly to
avoid subtracting nearly equal matrices for strongly localized modes.
"""
from numerical_andrews.paths import RESULTS
import json
from pathlib import Path

import numpy as np
from scipy.special import roots_legendre

from numerical_andrews.core import spatial_spectral_modes, evaluate_spatial_spectral_modes


def check(alpha, cutoff, quadrature_order, dimensions=(4, 10, 30), window=0.25):
    modes, _ = spatial_spectral_modes(alpha, cutoff, max(dimensions))
    nodes, weights = roots_legendre(quadrature_order)
    outside = np.zeros((max(dimensions), max(dimensions)))
    for left, right in ((-0.5, -window), (window, 0.5)):
        points = (left+right)/2 + (right-left)*nodes/2
        basis = evaluate_spatial_spectral_modes(modes, points)
        outside += basis.T @ ((weights*(right-left)/2)[:, None]*basis)
    return [dict(alpha=alpha, cutoff=cutoff, quadrature_order_per_interval=quadrature_order,
                 dimension=d, window_half_width=window,
                 maximum_outside_energy_fraction=float(np.linalg.eigvalsh(outside[:d, :d])[-1]))
            for d in dimensions]


if __name__ == '__main__':
    rows = []
    for cutoff, order in ((256, 256), (256, 512), (512, 512)):
        rows.extend(check(0.001, cutoff, order))
    result = dict(
        method='Direct Gauss-Legendre integration on [-1/2,-w] and [w,1/2]',
        status='Floating-point numerical check, not a validated certificate',
        rows=rows)
    path = RESULTS/'localization_checks.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
