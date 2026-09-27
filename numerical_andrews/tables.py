"""Publication tables generated from saved numerical results."""
import csv
from pathlib import Path

METRICS = ['mean_fc', 'upper_5pct_mean_fc', 'upper_1pct_mean_fc']


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

def write_fitted_table(df, output):
    lines=[r'\begin{table}[ht]',r'\caption{Comparisons using a common fitted range at each $\alpha$. All methods in a row use the same vertical range. $C$, $S$, and $*$ give absolute mean FC for the cosine-first, sine-first, and reoptimized MMQV bases. The retained basis $B$ is the lowest-mean member of $C$, $S$, and $*$ at the final common range; the three ratios divide MMSSQV scores by those of $B$. The two rows per dataset use the common illustrative parameter and the minimax point on the grid for this protocol. Searches minimize mean FC, not tails or ratios, and do not certify a global optimum.}',r'\label{tabUnionFitted}',r'\centering\small',r'\setlength{\tabcolsep}{3pt}',r'\resizebox{\textwidth}{!}{\begin{tabular}{llrrrrcrrr}',r'\toprule',r'Dataset & Choice & $\alpha$ & $C$ & $S$ & $*$ & $B$ & Mean & Upper $5\%$ & Upper $1\%$\\',r'\midrule']
    for _,r in df.iterrows():
        name={'iris':'Iris','diabetes':'Diabetes','bc':'BC (raw)','bc_standardized':'BC (std.)'}[r.dataset]
        vals=[r['cosine_first_mean_fc'],r['sine_first_mean_fc'],r['optimized_mean_fc']]+[r[m+'_ratio'] for m in METRICS]
        lines.append(name+' & '+r.selection+f' & {r.alpha:.4g} & '+' & '.join(f'{v:.3f}' for v in vals[:3])+' & '+{'optimized':'$*$','cosine_first':'$C$','sine_first':'$S$'}[r.retained_baseline]+' & '+' & '.join(f'{v:.3f}' for v in vals[3:])+r'\\')
    lines.extend([r'\bottomrule',r'\end{tabular}}',r'\end{table}'])
    output.write_text('\n'.join(lines)+'\n')

