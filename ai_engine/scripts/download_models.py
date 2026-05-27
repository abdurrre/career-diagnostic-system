import os
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO_ID = os.getenv("HF_MODEL_REPO_ID")
HF_TOKEN = os.getenv("HF_TOKEN")

if not REPO_ID:
    raise RuntimeError("HF_MODEL_REPO_ID is not set")

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

for filename in ["ner_model.keras", "gap_model.keras"]:
    target = MODELS_DIR / filename

    if target.exists():
        print(f"{filename} already exists")
        continue

    print(f"Downloading {filename} from {REPO_ID}")
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=filename,
        token=HF_TOKEN,
    )

    target.write_bytes(Path(downloaded).read_bytes())
    print(f"Saved {target}")