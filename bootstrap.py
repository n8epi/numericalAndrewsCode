#!/usr/bin/env python3
"""Create the Python 3.12 environment used by the reproduction workflow."""
import argparse
from pathlib import Path
import subprocess
import sys
import venv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--venv', type=Path, default=Path(__file__).resolve().parent / '.venv')
    args = parser.parse_args()
    if sys.version_info[:2] != (3, 12):
        parser.error('Run this script with Python 3.12, e.g. python3.12 bootstrap.py')
    root = Path(__file__).resolve().parent
    destination = args.venv.expanduser().resolve()
    venv.EnvBuilder(with_pip=True).create(destination)
    python = destination / ('Scripts/python.exe' if sys.platform == 'win32' else 'bin/python')
    pip = [str(python), '-m', 'pip', '--disable-pip-version-check']
    subprocess.run(pip + ['install', 'setuptools==84.0.0', 'wheel==0.48.0'], check=True)
    # Every runtime dependency is pinned explicitly. Avoid the upstream scorer's
    # obsolete NumPy/SciPy/Pillow pins while keeping its code unchanged.
    subprocess.run(pip + ['install', '--no-deps', '-r', str(root / 'requirements.txt')], check=True)
    subprocess.run(pip + ['install', '--no-deps', '--no-build-isolation', str(root)], check=True)
    subprocess.run([str(python), '-c',
        'import numpy, scipy, pandas, sklearn, matplotlib, PIL, cv2, skimage, pyrtools, visual_clutter; '
        'from numerical_andrews.congestion import fc_profile; '
        'scores, _ = fc_profile(numpy.full((64, 64, 3), 128, dtype=numpy.uint8)); '
        'assert all(numpy.isfinite(v) for v in scores.values()); print("Environment verified")'], check=True)
    print(f'\nReady. Run:\n  "{python}" -m numerical_andrews --plan\n'
          f'  "{python}" -m numerical_andrews --workers 4')


if __name__ == '__main__':
    main()
