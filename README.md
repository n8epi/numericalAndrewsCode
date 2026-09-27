# Numerical Andrews: minimal reproduction repository

This reproduces the numerical experiments in *Numerical
Approximation of Andrews Plots with Optimal Spatial-Spectral Smoothing*.
Use this directory as the root of a GitHub repository. No sibling directories,
saved search results, datasets, figures, or tables are needed as inputs.
The Iris, breast-cancer, and diabetes datasets are included with scikit-learn.

## Install

Install **Python 3.12**, then run from this directory:

```sh
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

## Reproduce

```sh
# Inspect the workflow without creating output or running experiments.
.venv/bin/python -m numerical_andrews --plan

# Run all paper and supplementary experiments from an empty results directory.
.venv/bin/python -m numerical_andrews --workers 4
```

On Windows, replace `.venv/bin/python` with `.venv\Scripts\python.exe`.
Outputs go to `results/`, which Git ignores. Choose another location with
`--output /path/to/results`. The workflow records completed steps, source hashes,
package versions, timings, and per-step logs in that directory. Rerun the same
command to resume; completed stages and completed independent searches are
reused. An interrupted unfinished search restarts deterministically from its
seed. Use a new output directory after changing the code or environment.

**The full run is computationally expensive:** it includes 177 fitted-range
searches with 114,432 candidate evaluations, in addition to the fixed-range
searches and original-scorer validation. Runtime depends on hardware and can be
many hours. `--workers` controls parallel search processes; decrease it to reduce
memory use. The workflow fixes numerical-library threads to one per process.

For a short first run that excludes all congestion searches:

```sh
.venv/bin/python -m numerical_andrews --until numerics
```

This generates the introductory figures, convergence data and figure,
localization checks, an eigenpair certificate, and the runtime table. Later,
run the full command with the same output directory to continue. Alternatively,
`--until fixed` stops after the fixed-range and standardization analyses.
The full workflow performs these steps in dependency order:

1. Numerical checks, introductory figures, and runtime measurements.
2. Named-basis score paths, then their archive for subsequent comparisons.
3. Initial MMQV searches: budgets 96 (Iris), 192 (raw WDBC), 128 (diabetes),
   and 192 (standardized WDBC), seed 20260923.
4. Independent fixed-range searches: seeds 20260924–20260926, with cumulative
   checkpoints at 16, 32, and 64 evaluations per phase angle.
5. Fixed-range and standardization comparisons, search diagnostics, and audits.
6. Per-parameter common fitted-range searches, original-scorer validation,
   final comparisons, and both main-paper tables.

The convergence diagnostic retains the original request for `d+3` eigenpairs
per parity block, so its floating-point error floor matches the published
figure. Display construction still selects only the first `d` interlaced modes.

All searches retain the paper's cutoff 256, 4097-point curve grid, 768×512
raster, and 17-parameter grid. Candidates minimize mean Feature Congestion;
upper-5% and upper-1% means are reported without independently optimizing tails.
If a named basis is retained, the shared range remains the one set by the
search winner. Runtime measurements will naturally differ on another machine.

## Generated material

| Material | Location under `results/` |
| --- | --- |
| Figures 1–2 and worked Iris calculation | `iris_intro_*.png`, `intro_iris_metadata.json` |
| Figure 3 and convergence data | `truncation_convergence.*`, `theorem2_truncation_*.csv` |
| Figures 4–5 | `union_fitted/ratios.*`, `union_fitted/examples.*` |
| Tables 1–2 | `runtime_table.tex`, `union_fitted/table.tex` |
| Fixed-range sensitivity figures and data | `feature_congestion_alpha_*`, `feature_congestion_display_profiles.csv` |
| Standardization sensitivity | `standardized_breast_cancer/` |
| Fixed-range search diagnostics | `optimized_mmqv_search.*`, `optimized_mmqv_*.csv` |
| Initial, expanded, and retained searches | `optimized_mmqv_initial/`, `optimized_mmqv_expanded/`, `optimized_mmqv/` |
| Fitted-range searches and diagnostics | `union_fitted/search/`, its accompanying CSV/JSON files |
| Localization, certificate, timing, and scorer checks | corresponding `localization_*`, `truncation_certificate_*`, `runtime_*`, `fc_*` files |

There are no precomputed publication numbers to load or silently substitute.
CSV/JSON outputs supply the detailed supplementary tables and diagnostics;
the manuscript text and editorial supplement are not duplicated here.

## Checks and implementation

The full workflow was verified on 26 September 2026 with Python 3.12.7 and the
pinned environment: all 193 searches (122,144 candidates) were recomputed,
all five paper figures matched, and Table 2 was byte-identical. All 24 package
tests and 38 checks of printed numerical claims passed. Archived search scores
agreed within 5.33e-15; timings and source-provenance fields naturally differ.

```sh
.venv/bin/python tests.py
```

`numerical_andrews/` contains the shared PCA, finite-section eigensolver,
Fourier synthesis, controlled rendering, scorer, and table formatting.
`numerical_andrews/experiments/` contains only retained reproduction drivers.
To run a driver individually, first create a writable results directory, set
`NUMERICAL_ANDREWS_RESULTS` to its absolute path, and invoke
`python -m numerical_andrews.experiments.<driver>`. The top-level workflow
handles these paths and inter-stage archives automatically.

Feature Congestion uses the unchanged `visual-clutter` 1.0.7 implementation
by Amir Hossein Kargaran, with reference-compatible preprocessing in
`numerical_andrews/congestion.py`. The RGB/XYZ constants, float64 arithmetic,
Gaussian kernel, and reflected boundaries follow Rosenholtz, Li, and Nakano's
June 2007 software. RGB is normalized to [0,1] while the historical XYZ reference
divisors near 100 are retained; this deliberate convention differs from a
standard normalized CIELAB conversion. Tests independently check the source
operations; no end-to-end MATLAB equivalence is claimed. Search acceleration
is process-local, and winners are rescored without acceleration.

Source attribution and reference-file checksums are preserved in
`numerical_andrews/fc_reference_provenance.json`. Reference MATLAB files and
third-party package sources are not bundled.
