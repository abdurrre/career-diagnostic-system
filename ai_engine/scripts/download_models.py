import os
from pathlib import Path
from huggingface_hub import hf_hub_download


def download_models():
    repo_id = os.getenv("HF_MODEL_REPO_ID")
    hf_token = os.getenv("HF_TOKEN")

    if not repo_id:
        raise RuntimeError("HF_MODEL_REPO_ID is not set")

    # Models to download into the 'models' directory
    model_files = ["ner_model.keras", "gap_model.keras"]
    models_dir = Path(__file__).resolve().parents[1] / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Pickle data files to download into the 'data' directory
    data_files = ["ner_tokenizer.pkl", "skill_binarizer.pkl", "tokenizer.pkl", "job_encoder.pkl"]
    data_dir = Path(__file__).resolve().parents[1] / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download models
    for filename in model_files:
        target = models_dir / filename

        if target.exists():
            print(f"{filename} already exists")
            continue

        print(f"Downloading {filename} from {repo_id}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token,
        )

        target.write_bytes(Path(downloaded).read_bytes())
        print(f"Saved {target}")

    # Download data pickles
    for filename in data_files:
        target = data_dir / filename

        if target.exists():
            print(f"{filename} already exists")
            continue

        print(f"Downloading {filename} from {repo_id}")
        downloaded = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=hf_token,
        )

        target.write_bytes(Path(downloaded).read_bytes())
        print(f"Saved {target}")


if __name__ == "__main__":
    download_models()
