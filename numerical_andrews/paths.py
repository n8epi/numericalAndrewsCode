"""All generated data belongs to the selected writable results directory."""
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get('NUMERICAL_ANDREWS_RESULTS', 'results')).expanduser().resolve()
