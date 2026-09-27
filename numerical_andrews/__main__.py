"""Reproduce the paper and supporting analyses from source, without saved data."""
import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

RAW_DATASETS = ('iris', 'bc', 'diabetes')
INITIAL_BUDGETS = {'iris': 96, 'bc': 192, 'diabetes': 128, 'bc_standardized': 192}
PHASES = ('numerics', 'fixed', 'all')


def plan(workers):
    """Each item is (name, phase, module or local action, arguments)."""
    steps = [
        ('intro', 'numerics', 'generate_intro_figures', []),
        ('convergence', 'numerics', 'analyze_truncation_convergence', []),
        ('localization', 'numerics', 'check_localization', []),
        ('runtime', 'numerics', 'benchmark_runtime', []),
        ('certificate', 'numerics', 'certify_truncation',
         ['--alpha', '0.2', '--sector', 'even', '--size', '24', '--index', '3',
          '--output', '{results}/truncation_certificate_example.json']),
        ('runtime_table', 'numerics', 'generate_tables', ['--runtime-only']),
        ('named_baselines', 'fixed', 'analyze_feature_congestion_alpha', ['--baseline-mode', 'two']),
        ('archive_named', 'fixed', '@archive_named', []),
    ]
    for key in RAW_DATASETS:
        steps.append((f'initial_{key}', 'fixed', 'optimize_mmqv_feature_congestion',
                      ['--dataset', key, '--budget', str(INITIAL_BUDGETS[key])]))
    steps += [
        ('archive_raw', 'fixed', '@archive_initial', list(RAW_DATASETS)),
        ('prepare_standardized', 'fixed', 'analyze_standardized_breast_cancer', ['--stage', 'prepare']),
        ('initial_standardized', 'fixed', 'optimize_mmqv_feature_congestion',
         ['--dataset', 'bc_standardized', '--budget', '192']),
        ('archive_standardized', 'fixed', '@archive_initial', ['bc_standardized']),
        ('expanded_searches', 'fixed', 'expand_mmqv_optimization',
         ['--workers', str(workers), '--datasets', *RAW_DATASETS, 'bc_standardized']),
        ('audit_raw', 'fixed', 'audit_expanded_mmqv', []),
        ('audit_standardized', 'fixed', 'audit_expanded_mmqv', ['--datasets', 'bc_standardized']),
        ('summarize_fixed', 'fixed', 'summarize_expanded_mmqv', []),
        ('compare_fixed', 'fixed', 'compare_optimized_mmqv', []),
        ('finalize_standardized', 'fixed', 'analyze_standardized_breast_cancer', ['--stage', 'finalize']),
        ('verify_scores', 'fixed', 'verify_fc_compatibility', []),
        ('validate_acceleration', 'fixed', '@acceleration', []),
        ('validate_union_acceleration', 'all', '@union_acceleration', []),
        ('fitted_searches', 'all', 'compare_union_fitted', ['--workers', str(workers)]),
        ('summarize_fitted', 'all', 'summarize_union_fitted', []),
        ('tables', 'all', 'generate_tables', []),
    ]
    return steps


def archive_named(results):
    # These are new outputs of the preceding stage, never bundled prerequisites.
    files = sorted(results.glob('feature_congestion_alpha_*'))
    if not files:
        raise FileNotFoundError('The named-baseline stage has not produced its outputs')
    for source in files:
        target = results / source.name.replace('feature_congestion_alpha_', 'feature_congestion_two_baseline_', 1)
        shutil.copy2(source, target)


def archive_initial(results, datasets):
    destination = results / 'optimized_mmqv_initial'
    destination.mkdir(exist_ok=True)
    for key in datasets:
        for suffix in ('.json', '_best.png', '_history.csv'):
            shutil.copy2(results / 'optimized_mmqv' / (key + suffix), destination / (key + suffix))


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
    if module == '@archive_named':
        archive_named(results)
    elif module == '@archive_initial':
        archive_initial(results, arguments)
    else:
        target = ('numerical_andrews.union_fc_acceleration' if module == '@union_acceleration'
                  else 'numerical_andrews.fast_feature_congestion' if module == '@acceleration'
                  else 'numerical_andrews.experiments.' + module)
        command = [sys.executable, '-m', target,
                   *(item.replace('{results}', str(results)) for item in arguments)]
        with log.open('w') as stream:
            subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--workers', type=int, default=4, help='Parallel search processes (default: 4)')
    parser.add_argument('--until', choices=PHASES, default='all',
                        help='Stop after numerical checks, fixed-range analyses, or the full workflow')
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
