# Numerical Andrews: reproduction code

This repository provides a standalone reproduction of the numerical experiments in *Numerical
Approximation of Andrews Plots with Optimal Spatial-Spectral Smoothing* by Rimerman and Strawn.
The source code and reproduction instructions are hosted at
[github.com/n8epi/numericalAndrewsCode](https://github.com/n8epi/numericalAndrewsCode).
All commands and file paths below are relative to a checkout of that repository.
No saved search results, datasets, figures, or tables are needed as inputs.
The Iris, breast-cancer, and diabetes datasets are included with scikit-learn.

## Install

Install **Python 3.12**, then clone the repository and run:

```sh
git clone https://github.com/n8epi/numericalAndrewsCode.git
cd numericalAndrewsCode
python3.12 bootstrap.py
```

On Windows, use `py -3.12 bootstrap.py`. The script creates `.venv`, installs
all explicitly pinned dependencies in `requirements.txt`, installs this package,
and checks that the congestion scorer runs. Internet access is needed for this
installation; the experiments do not download datasets.

The upstream `visual-clutter` package declares obsolete dependency versions.
The bootstrap therefore installs the complete pinned dependency set with
`--no-deps`, then installs this project. This intentionally bypasses those
obsolete declarations while preserving the upstream scoring code. Use the
bootstrap rather than a bare `pip install -r requirements.txt`. The versions
match the reference environment; cross-platform floating-point and raster
results are not promised to be bitwise identical.

## Reproduce the paper

```sh
# See the stages without creating files.
.venv/bin/python -m numerical_andrews --plan

# Reproduce the figures, tables, and supporting numerical checks.
.venv/bin/python -m numerical_andrews --workers 3

# Optional: stop after the numerical checks, before congestion scoring.
.venv/bin/python -m numerical_andrews --until numerics
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.
Outputs go to `results/`, which Git ignores. The workflow records completed
stages, source hashes, package versions, timings, and logs. Rerun the same
command to resume completed stages; an interrupted search stage restarts its
three deterministic candidate pools. Pass `--output /absolute/path` to choose
another output directory. Direct experiment commands use
`NUMERICAL_ANDREWS_RESULTS` or their `--output`/`--output-directory` option.
Use a new output directory after changing source or dependencies. The default
workflow generates the paper's figures, tables, and supporting numerical checks
from the supplied datasets.

The median comparison compares 256 candidate plots per method and
dataset, with duplicate MMSSQV sign vectors cached. It can take several
minutes. `--workers` controls parallel dataset processes; use 1 to reduce
memory use. Its JSON records preserve the sampled parameters, signs, scores,
selected members, and run timings. All selected rasters are checked again
using the original, unaccelerated scorer.

| Paper material | Location under `results/` |
| --- | --- |
| Introductory Iris figures | `iris_intro_*.png`, `intro_iris_metadata.json` |
| Iris mode convergence at alpha=0.2 (Section 3.2) | `iris_moderate_alpha_convergence.csv` |
| Explicit Theorem 3 cutoff calculation | `iris_uniform_truncation_certificate.json` |
| Table 1: selected spatial-spectral weights | `median_comparison/selected_alpha_table.tex` |
| Selected alpha checks and full median-search histories | `median_comparison/` JSON files |
| Paired MMQV/MMSSQV figure | `median_comparison/examples.pdf` |
| Local Feature Congestion heat maps | `median_comparison/heatmaps.pdf` |
| Uncapped local maps and rendering checks | `median_comparison/heatmaps.npz`, `median_comparison/heatmaps_metadata.json` |
| Best/worst sampled median Feature Congestion table | `median_comparison/table.tex` |
| Worst sampled MMSSQV scores and reference checks | `median_comparison/worst_summary.json` |
| Data-only rasters actually scored | `median_comparison/` PNG files |
| Supporting convergence diagnostics | `truncation_convergence.*`, `theorem3_truncation_*.csv` |
| Supporting runtime measurements | `runtime_*` files |

The two manuscript tables are generated from the same full-precision records
used for the figures. The package also generates the supporting convergence
diagnostics and runtime measurements described below. Runtime measurements
vary by machine and load.

To preview the alpha choices without congestion scoring:

```sh
.venv/bin/python -m numerical_andrews.experiments.compare_median_congestion --preview-only --output-directory alpha_preview
```

This preview shows the first sampled sign vector. Both the preview and
the final comparison fit each plot's own minimum and maximum with 5% of
the span added at each end; the final comparison selects the lowest-median
members.

The full median-comparison command also generates the heat maps. To rebuild
only those maps from existing selected rasters and search records:

```sh
.venv/bin/python -m numerical_andrews.experiments.render_median_heatmaps
```

To refresh the best/worst table and independently recheck the three worst
sampled MMSSQV plots from existing candidate histories:

```sh
.venv/bin/python -m numerical_andrews.experiments.compare_median_congestion --refresh-worst-only
```

This writes `worst_summary.json` and `table.tex`; it preserves the original
search records, timing metadata, selected plots, and heat maps.

## Plot selection and scoring details

Alpha is chosen to put **75% of aggregate pairwise squared L2 separation**
in the central half [-0.25, 0.25]. For centered data this fraction equals
`sum(singular_values**2 * central_mode_energies) / sum(singular_values**2)`;
PCA score cross-products cancel. The same fraction is exactly 50% for every
MMQV basis. The 75% display target retains a moderate central emphasis and
leaves 25% of the aggregate separation outside. It does not require each
individual pair or each low-variance feature direction to be concentrated,
and it is invariant under the remaining eigenfunction signs.

We first evaluate the 16 values `10**(-3+j/3)`, `j=0,...,15`, together with
0.2. We collect adjacent brackets that cross 0.75 downwards, choose the one
with largest alpha, and refine within it using a bracketed solve in log alpha.
This procedure does not assume monotonicity or claim to find every possible
crossing between grid points. Central mode energies use 256-point
Gauss-Legendre quadrature and Fourier cutoff 256. The selected fraction is
checked again with both cutoff and quadrature order 512 and must agree with
0.75 within 1e-8. Full-precision alpha values are used to make the plots;
printed values are rounded. These are floating-point checks, not validated
infinite-dimensional certificates. Congestion plays no role in choosing alpha.

The numerical construction centers the features, computes a thin SVD, and
orders PCA scores by decreasing singular value. Each loading is signed so
its largest-magnitude coordinate is positive. Iris and WDBC use unscaled
features; diabetes uses scikit-learn's supplied scaling (division by feature
standard deviation times the square root of sample size). Labels set curve colors and therefore affect the raster-based selection;
they do not enter PCA or the variational objectives. Diabetes response quartiles supply four display groups.
All observations and features are retained; these datasets contain no
missing or nonfinite entries.

The real even and odd tridiagonal Fourier blocks are solved separately at
cutoff 256 with SciPy `eigh_tridiagonal`, LAPACK driver `stebz`, indexed
low-eigenvalue selection, and absolute tolerance `1e-14`. They are interleaved
by parity and within-block index, beginning with the even block. Only the first `d` modes are used. Their initial signs have
even value at zero or odd derivative at zero nonnegative; the random search
then multiplies every mode by an independent sign. Curves are synthesized
at 4097 equally spaced points on [-0.5, 0.5]. Eigenpairs and the sampled basis
can be reused for all observations and sign choices.

Each MMQV pool contains the cosine-first and sine-first named bases, plus
254 random bases. For each frequency pair we sample a uniform angle on
[0, 2*pi) and a reflection sign; the constant sign is also random. When the
last pair is incomplete its remaining mode still has its own angle. The
MMSSQV pool contains 256 independently sampled vectors in {-1,+1}^d.
Duplicated sign vectors are recorded but scored once. Each pool is ranked
by the median over **all pixels** of the combined local congestion map.
The first sampled minimizer breaks exact score ties. The MMQV/MMSSQV seed pairs are 202610031/202610032 (Iris),
202610033/202610034 (WDBC), and 202610035/202610036 (diabetes). These seeds
and the candidate histories fully specify the search. For Iris, the seeded
MMSSQV pool contains all 16 possible sign patterns: its best and worst are
exhaustive over that sign family at the chosen alpha, cutoff, and rendering
protocol. The other MMSSQV pools and all MMQV pools remain sampled searches.
Each dataset record reports the number of unique signs, the full family size,
and whether coverage is exhaustive.

The table also reports the largest pixel median among the 256 sampled
MMSSQV sign vectors. This is a maximum over candidate plots, not over pixels.
Each maximizing candidate is reconstructed using its saved signs, alpha,
and individually fitted vertical limits, then checked with the original
scorer. The exact scores and percentage changes relative to the best
sampled MMQV score are recorded in `worst_summary.json`.
The exhaustive Iris exception does not extend to the other datasets or to
the continuous MMQV families.

Before scoring, every plot is fitted independently: if its displayed curve
extrema are `a` and `b`, its y-limits are
`(a - 0.05*(b-a), b + 0.05*(b-a))`. The same rule applies to the introductory
figures, previews, and all MMQV/MMSSQV search candidates. The helper
`fitted_y_limits` uses a nonzero fallback margin for a constant plot.
Each candidate record stores its extrema and resulting limits. The
rendering uses a data-only 768 x 512 Agg raster, 100 dpi, background #E5ECF6,
0.65-point lines, opacity 0.52, and identical palettes, label-group draw order,
and antialiasing. Axes, ticks, titles, and legends do not enter the score.
The publication figure displays those same rasters with axes and annotations
added outside them. This distinction matters because scoring decorated
figures would compare their text and margins as well as their curves.

The heat-map driver recomputes the unaccelerated local Feature Congestion
maps from the six saved selected rasters and checks that each median matches
its recorded score. Each map retains its selected plot's individual y-limits,
coordinates, and aspect ratio. MMQV and MMSSQV share one linear color scale per dataset,
from zero to the pooled 99.5th percentile of that row's two maps; brighter
colors mean greater congestion. Values above that threshold saturate only
in the displayed colors. The NPZ archive retains the complete uncapped maps,
and all numerical scores use these original values. The accompanying metadata
records raster and source hashes, color limits, raw extrema, and median checks.

The median is not the published global (mean-pooled) Feature Congestion
statistic. It describes a typical pixel and can tie at the background floor
when large parts of an image are quiet. Congestion scores do not establish
class separation: both embeddings preserve every pairwise
Euclidean distance as an L2 curve distance.

## Numerical cost and hardware measurements

For `n >= d`, centering, thin SVD, and PCA scores cost O(n*d^2); generally,
the thin SVD costs O(n*d*min(n,d)). Forming the two Jacobi blocks costs O(N).
Computing their selected low modes typically costs O(N*d), or conservatively
O(N*d^2) if inverse iteration needs reorthogonalization for tightly clustered
eigenvalues, and stores O(N*d) coefficients. Direct Fourier synthesis on an
M-point grid costs O(M*N*d), and forming all curves costs O(n*M*d). Thus the
typical total is O(n*d^2 + N*d + M*N*d + n*M*d). The basis depends only on
(alpha, N, d, M) and can be cached. PCA need only be computed once when
alpha changes. Rendering and Feature Congestion are separate costs and are
not required to compute an MMSSQV embedding.

The following numerical-construction timings were measured at
alpha=0.2, N=256, M=4097 on a **16-core Apple M4 Max MacBook Pro with 48 GB
memory, macOS 26.6.2**, using Python 3.12.7, the pinned dependencies, and one
numerical-library thread. Three warmups preceded 20 measured repetitions.
Dataset loading, file output, raster rendering, and congestion scoring were
excluded. Each phase and the complete construction were timed independently;
the complete time is not the sum of the phase medians. Randomized
median-search timings are reported separately below.

| Dataset (n,d) | PCA/scores ms | Jacobi+basis ms | Curves ms | Complete median [IQR] ms |
| --- | ---: | ---: | ---: | ---: |
| Iris (150,4) | 0.115 | 13.974 | 0.406 | 14.217 [14.016,14.445] |
| WDBC (569,30) | 0.407 | 101.649 | 0.577 | 103.477 [102.581,105.140] |
| Diabetes (442,10) | 0.142 | 34.728 | 0.352 | 35.240 [34.785,35.772] |

The convergence experiment compares cutoffs against N=512 at alpha=0.001,
matching eigenpairs by parity and within-block rank and aligning their signs.
At N=256, maxima over the three datasets are 5.33e-15 (absolute eigenvalue
difference), 1.08e-15 (coefficient-vector norm difference), and 9.78e-14
(relative singular-value-weighted objective difference). The plot is retained
in `truncation_convergence.pdf` as a supporting diagnostic. Its errors have
already reached their floating-point plateau by N=96; the larger reference
is a numerical approximation, not an exact infinite-dimensional solution.
The driver requests `d+3` eigenpairs per parity block for this roundoff
diagnostic. This is the small-alpha
supporting study, not the alpha=0.2 worked example.

For the worked Iris example at alpha=0.2, the same driver separately compares
the first four modes at cutoffs 8 through 256 against N=512, using just the
required two modes per parity sector. By N=16, the maximum coefficient-vector
error is below 2.3e-16 and the maximum eigenvalue error is below 5.4e-15.
`iris_moderate_alpha_convergence.csv` records this floating-point comparison.
The independent rational Theorem 3 calculation gives the sufficient cutoff
N=111 for uniform truncation error below 1e-12 for all centered Iris curves.
It applies to exact finite eigenpairs; the empirical mode errors are not a
certificate for the limiting curves or for rounding errors. The explicit
rule is conservative but modest in absolute size at this parameter.

The cutoff's leading term is `7*(d+2) + 21/sqrt(alpha)`. At alpha=0.001 the
alpha-dependent term is about 664, and the full leading term is about 888
when d=30. The selected plotting parameters (3.87045, 4.24567, and 0.315437)
are all above 0.31, keeping the alpha-dependent term below 38.

The recorded three-worker search completed in approximately 290 seconds;
worst-candidate checks and heat-map rendering are separate.
Its `median_comparison/protocol.json` records the elapsed
time, worker count, numerical-library thread settings, package versions,
and platform. Each dataset JSON records its complete alpha scan, root
bracket, refined alpha, doubled-resolution check, candidate scores, and
reference-scorer validations. The weighted energy identity is checked against
explicit distances between every observation pair; five MMQV candidates
are also checked numerically against the exact 50% benchmark. Duplicate sign
vectors are cached; the seeded MMSSQV pools contain 16 distinct vectors for Iris, 256 for WDBC, and 231
for diabetes. All 256 sampled vectors per dataset remain in the histories.
Score differences below 1e-8 are treated as numerical ties in the diagnostics.

## Checks and implementation

```sh
.venv/bin/python tests.py
```

`numerical_andrews/` contains PCA, the finite-section eigensolver, Fourier
synthesis, rendering, scoring, and table formatting. The `experiments/`
subdirectory contains reproduction drivers. Individual drivers use the
writable location selected by `NUMERICAL_ANDREWS_RESULTS`, defaulting to
`results/` in the repository checkout.

Feature Congestion uses the `visual-clutter` 1.0.7 implementation
by Amir Hossein Kargaran, with reference-compatible preprocessing in
`numerical_andrews/congestion.py`. The RGB/XYZ constants, float64 arithmetic,
Gaussian kernel, and reflected boundaries follow Rosenholtz, Li, and Nakano's
June 2007 software. RGB is normalized to [0,1] while the historical XYZ reference
divisors near 100 are retained; this deliberate convention differs from a
standard normalized CIELAB conversion. Tests independently check the source
operations; no end-to-end MATLAB equivalence is claimed. Search acceleration
is process-local, and winners are rescored without acceleration.

The numerical tests also cover the full Iris sign pool, aggregate energy
against explicit pairwise distances, independently fitted y-limits, the exact
cutoff calculation, and accelerated versus original local congestion maps.

Source attribution and reference-file checksums are preserved in
`numerical_andrews/fc_reference_provenance.json`. Reference MATLAB files and
third-party package sources are not bundled.
