"""Generate both main-paper tables from the bundled or exported CSV results."""
import argparse
import pandas as pd
from numerical_andrews.paths import RESULTS
from numerical_andrews.tables import write_runtime_table, write_fitted_table


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime-only', action='store_true')
    args = parser.parse_args()
    write_runtime_table(RESULTS / 'runtime_benchmark.csv', RESULTS / 'runtime_table.tex')
    if args.runtime_only:
        return
    fitted = RESULTS / 'union_fitted'
    write_fitted_table(pd.read_csv(fitted / 'landmarks.csv'), fitted / 'table.tex')
    print(f'Wrote both paper tables to {RESULTS}')


if __name__ == '__main__':
    main()
