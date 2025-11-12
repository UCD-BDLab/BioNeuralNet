import sys
from pathlib import Path

# Ensure repository root is on sys.path so tests can import 'bioneuralnet'
REPO_ROOT = Path(__file__).resolve().parent
repo_str = str(REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)
