from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SUBMISSION_ROOT = PACKAGE_ROOT.parent
DEFAULT_DATA_ROOT = SUBMISSION_ROOT / "cache" / "datasets"
