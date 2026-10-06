"""All generated data belongs to the selected writable results directory."""
# Source repository: https://github.com/n8epi/numericalAndrewsCode
import os
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get('NUMERICAL_ANDREWS_RESULTS', 'results')).expanduser().resolve()
