"""Supporting runtime table generated from numerical benchmark results."""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
import csv
from pathlib import Path


def write_runtime_table(csv_path, output):
    with Path(csv_path).open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    lines = [RUNTIME_HEADER.rstrip('\n')]
    for row in rows:
        lines.append(
            f"{row['dataset']} $({row['observations']},{row['dimension']})$ & "
            f"{float(row['pca_median_ms']):.3f} & "
            f"{float(row['eigensolve_basis_median_ms']):.3f} & "
            f"{float(row['curve_formation_median_ms']):.3f} & "
            f"{float(row['combined_median_ms']):.3f} "
            f"[{float(row['combined_q1_ms']):.3f}, {float(row['combined_q3_ms']):.3f}]"
            + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    Path(output).write_text('\n'.join(lines) + '\n')

RUNTIME_HEADER = '\\begin{table}[ht]\n\\caption{Median wall-clock time in milliseconds. Each phase was measured\nindependently; brackets give the interquartile range for the complete numerical\nconstruction. The complete time is measured separately and is not the sum\nof the phase medians.}\n\\label{tabRuntime}\n\\centering\n\\small\n\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lrrrr}\n\\toprule\nDataset $(n,d)$ & PCA/scores & Jacobi+basis & Curves & Complete [IQR]\\\\\n\\midrule\n'
