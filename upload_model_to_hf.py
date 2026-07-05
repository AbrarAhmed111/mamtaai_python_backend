"""
Upload the fine-tuned wav2vec2 model to HuggingFace Hub.

One-time setup (and again after each retrain):
    1. Create a token with WRITE access: https://huggingface.co/settings/tokens
    2. Set it:   $env:HF_TOKEN = "hf_..."        (PowerShell)
                 export HF_TOKEN=hf_...           (bash)
    3. Run:      python upload_model_to_hf.py [repo_id]

Default repo_id: abbyabrar/mamtaai-cry-classifier
"""
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

MODEL_DIR = Path("models/wav2vec2_cry_classifier")
DEFAULT_REPO = "abbyabrar/mamtaai-cry-classifier"
REQUIRED = ("config.json", "label_map.json", "preprocessor_config.json", "model.safetensors")


def main():
    repo_id = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_REPO
    token = os.getenv("HF_TOKEN")
    if not token:
        print("ERROR: Set HF_TOKEN environment variable (write-access token).")
        print("Create one at https://huggingface.co/settings/tokens")
        sys.exit(1)

    missing = [f for f in REQUIRED if not (MODEL_DIR / f).exists()]
    if missing:
        print(f"ERROR: Missing model files in {MODEL_DIR}: {missing}")
        sys.exit(1)

    api = HfApi(token=token)

    print(f"Creating repo '{repo_id}' (private) if it does not exist ...")
    api.create_repo(repo_id=repo_id, repo_type="model", private=True, exist_ok=True)

    print(f"Uploading {MODEL_DIR} -> https://huggingface.co/{repo_id}")
    api.upload_folder(
        folder_path=str(MODEL_DIR),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned wav2vec2 cry classifier (79.40% test accuracy)",
    )

    print("\nDone. To use it on a deployed backend, set these env vars:")
    print(f"  WAV2VEC2_HF_REPO={repo_id}")
    print("  HF_TOKEN=<read-access token>   (required because the repo is private)")


if __name__ == "__main__":
    main()
