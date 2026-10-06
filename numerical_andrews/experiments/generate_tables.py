"""Generate the supporting runtime table from a fresh benchmark CSV."""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
from numerical_andrews.paths import RESULTS
from numerical_andrews.tables import write_runtime_table


def main():
    write_runtime_table(RESULTS / 'runtime_benchmark.csv', RESULTS / 'runtime_table.tex')
    print(f'Wrote supporting runtime table to {RESULTS}')


if __name__ == '__main__':
    main()
