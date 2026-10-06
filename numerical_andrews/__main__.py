"""Reproduce the paper and supporting analyses from source, without saved data."""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PHASES = ('numerics', 'all')


def plan(workers):
    """Each item is (name, phase, experiment module, arguments)."""
    return [
        ('intro', 'numerics', 'generate_intro_figures', []),
        ('uniform_cutoff', 'numerics', 'verify_uniform_iris_bound', []),
        ('convergence', 'numerics', 'analyze_truncation_convergence', []),
        ('runtime', 'numerics', 'benchmark_runtime', []),
        ('runtime_table', 'numerics', 'generate_tables', []),
        ('median_comparison', 'all', 'compare_median_congestion',
         ['--workers', str(workers)]),
    ]


def source_hash():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*.py')):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment_record():
    names = ('numpy', 'scipy', 'pandas', 'matplotlib', 'scikit-learn', 'Pillow',
             'visual-clutter', 'pyrtools', 'scikit-image', 'opencv-python-headless')
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = 'not installed'
    return {'python': sys.version, 'packages': versions}


def save_state(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2) + '\n')
    temporary.replace(path)


def execute(module, arguments, results, environment, log):
    command = [sys.executable, '-m', 'numerical_andrews.experiments.' + module,
               *(item.replace('{results}', str(results)) for item in arguments)]
    with log.open('w') as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--workers', type=int, default=3, help='Parallel dataset processes (default: 3)')
    parser.add_argument('--until', choices=PHASES, default='all',
                        help='Stop after numerical checks or reproduce all paper figures, tables, and supporting checks')
    parser.add_argument('--plan', action='store_true', help='Print the ordered steps without creating files')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    steps = [step for step in plan(args.workers) if PHASES.index(step[1]) <= PHASES.index(args.until)]
    if args.plan:
        for index, (name, phase, module, arguments) in enumerate(steps, 1):
            print(f'{index:2}. {name}: {module} {" ".join(arguments)}')
        return
    results = args.output.expanduser().resolve()
    package = Path(__file__).resolve().parent
    if results == package or package in results.parents:
        parser.error('Choose an output directory outside the installed source package')
    state_path = results / 'reproduction.json'
    fingerprint = source_hash()
    environment_info = environment_record()
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['source_sha256'] != fingerprint or state['environment'] != environment_info:
            parser.error('Code or environment changed since the previous run; choose a new --output directory')
    else:
        if results.exists() and any(results.iterdir()):
            parser.error('Output must be empty or belong to this workflow; choose a new --output directory')
        results.mkdir(parents=True, exist_ok=True)
        state = {'source_sha256': fingerprint, 'environment': environment_info,
                 'completed': [], 'seconds': {}}
        save_state(state_path, state)
    environment = dict(os.environ, NUMERICAL_ANDREWS_RESULTS=str(results),
                       MPLCONFIGDIR=str(results / '.matplotlib'), PYTHONDONTWRITEBYTECODE='1')
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        environment[name] = '1'
    logs = results / 'logs'
    logs.mkdir(exist_ok=True)
    for name, phase, module, arguments in steps:
        if name in state['completed']:
            print(f'Skipping completed step: {name}', flush=True)
            continue
        log = logs / (name + '.log')
        print(f'Running {name}; log: {log}', flush=True)
        started = time.monotonic()
        try:
            execute(module, arguments, results, environment, log)
        except subprocess.CalledProcessError as error:
            print(f'Step failed: {name}. See {log}. Rerun the same command to resume.', file=sys.stderr)
            raise SystemExit(error.returncode)
        state['completed'].append(name)
        state['seconds'][name] = time.monotonic() - started
        save_state(state_path, state)
    print(f'Completed through {args.until}. Outputs: {results}')


if __name__ == '__main__':
    main()
