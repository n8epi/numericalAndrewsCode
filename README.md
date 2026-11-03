# Numerical Andrews

This repo maintains code and recorded results for *Numerical Approximation of Andrews Plots with Optimal Spatial-Spectral Smoothing* by Mitchell Rimerman and Nate Strawn.

## Install

Use **Python 3.12**:

```sh
git clone https://github.com/n8epi/numericalAndrewsCode.git
cd numericalAndrewsCode
python3.12 bootstrap.py
```

This creates `.venv` and installs the pinned dependencies. Use the bootstrap
because the upstream `visual-clutter` package declares obsolete dependency pins.
Installation needs internet access; the three datasets come with scikit-learn.
On Windows, use `py -3.12 bootstrap.py` and replace `.venv/bin/python` below
with `.venv\Scripts\python.exe`.

## Reproduce the experiments

```sh
# Generate the paper's figures, tables, and numerical checks.
.venv/bin/python -m numerical_andrews --workers 3

# Optional: run the numerical checks without the congestion search.
.venv/bin/python -m numerical_andrews --until numerics

# Run the tests.
.venv/bin/python tests.py
```

Outputs go to `results/`. Rerun the same command to resume completed stages;
use `--output /path/to/new/results` after changing the code or dependencies.
`--plan` lists the stages. The congestion search takes several minutes;
`--workers 1` reduces parallel memory use.

| Output | Location under `results/` |
| --- | --- |
| Introductory Iris plots | `iris_intro_*.png` |
| Iris cutoff calculation and mode convergence | `iris_uniform_truncation_certificate.json`, `iris_moderate_alpha_convergence.csv` |
| Convergence data, plot, and Table 2 | `theorem3_truncation_*.csv`, `truncation_convergence.*`, `truncation_convergence_table.tex` |
| Selected weights and congestion tables | `median_comparison/selected_alpha_table.tex`, `median_comparison/table.tex` |
| Selected plots and congestion maps | `median_comparison/examples.pdf`, `median_comparison/heatmaps.pdf` |
| Search settings, candidates, scores, and rasters | `median_comparison/` JSON and PNG files |
| Construction timings and machine details | `runtime_*` |

`reference_results/` contains the paper's recorded convergence data, search
histories, tables, and selected rasters for comparison. Fresh experiments
compute their results independently of these files. Small numerical or raster
differences can occur across platforms.

## Experiment settings

The datasets are Iris, Wisconsin Diagnostic Breast Cancer, and diabetes.
All principal components are retained. Iris and breast-cancer features are
centered without variance scaling; diabetes uses scikit-learn's supplied
scaling. Labels determine plot colors, and do not enter PCA or the objective.

The Fourier cutoff is `N=256` and the plotting grid has 4097 points.
The even and odd tridiagonal blocks are solved with SciPy's
`eigh_tridiagonal` (`stebz`, tolerance `1e-14`), then interleaved by parity
and rank. Even coefficients use frequencies `0,...,N`; odd coefficients
use `1,...,N`. Code ranks start at zero; CSV mode ranks start at one.

For each dataset, alpha is chosen to put 75% of aggregate pairwise squared
L2 separation in the central half of the plotting interval. A logarithmic scan
and bracketed solve select the parameter, using 256-point quadrature and
checking the result with both cutoff and quadrature order doubled to 512.
This is a display convention; variational optimality applies at each fixed
alpha. The convergence study separately uses alpha `0.001` and reference
cutoff `512`; the worked Iris calculation uses alpha `0.2`.

Each method is compared over 256 candidates per dataset. MMQV uses the two
named bases plus 254 random phase/reflection/sign choices; MMSSQV samples
eigenfunction signs, caching duplicates. Candidates are ranked by the median
of the uncapped local Feature Congestion map, with the first candidate breaking
an exact tie. Each plot uses its own vertical range with 5% padding. Scoring
uses data-only 768-by-512 rasters, 0.65-point lines, opacity 0.52, and background
`#E5ECF6`. Seeds, parameters, fitted limits, and scores are saved in the JSON
records. Selected plots are checked with the unaccelerated scorer.
The saved Iris run covers all 16 sign patterns; other searches cover sampled
candidates. The comparison concerns this raster statistic, rather than human
reading performance.

For `n >= d`, numerical construction typically costs
`O(n*d^2 + N*d + G*N*d + n*G*d)` for PCA, eigenpairs, Fourier synthesis,
and curve formation (`G` is the plotting-grid size). Eigenpair
reorthogonalization can increase that term to `O(N*d^2)`; rendering and
congestion scoring are separate. Reference construction medians were
14.2 ms (Iris), 103.5 ms (breast cancer), and 35.2 ms (diabetes), at alpha
`0.2`, `N=256`, `G=4097`, on an Apple M4 Max with 48 GB memory and one
numerical-library thread. Three warmups preceded 20 repetitions; dataset
loading and file output were excluded. Timings vary by machine.

Feature Congestion uses `visual-clutter` 1.0.7 with preprocessing following
Rosenholtz, Li, and Nakano's June 2007 software. Attribution and implementation
provenance are in `numerical_andrews/fc_reference_provenance.json`.
