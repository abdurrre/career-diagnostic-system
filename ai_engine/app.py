import os
import sys
from pathlib import Path

from scripts.download_models import download_models

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ.setdefault("HF_MODEL_REPO_ID", "rizsd21/career-diagnostic-models")

download_models()

from src.api import app  # noqa: E402
