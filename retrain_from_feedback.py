"""
Retrain the baby cry classifier using user-submitted feedback corrections.

This script:
  1. Fetches all cry_feedback rows from Supabase
  2. Downloads the corresponding audio file for each feedback row
  3. Extracts features from each audio (same pipeline as training)
  4. Merges corrected samples with the base training dataset
  5. Retrains all model types; saves the best one

Requirements (env vars):
  SUPABASE_URL      — e.g. https://xxxx.supabase.co
  SUPABASE_SERVICE_KEY — service-role key (bypass RLS so we can read all feedback)

Run from mamtaai_python_backend/:
  python retrain_from_feedback.py
"""
import io
import json
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import httpx
import librosa
import numpy as np

DATASET_JSON = Path("datasets/training_dataset.json")
MODEL_NAME = "baby_cry_classifier"

SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")  # must be service-role key


# ---------------------------------------------------------------------------
# Supabase helpers (thin REST wrappers — no heavy SDK needed)
# ---------------------------------------------------------------------------

def _headers() -> dict:
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }


def fetch_feedback() -> list[dict]:
    """Return all cry_feedback rows joined with the recording's file_url."""
    url = (
        f"{SUPABASE_URL}/rest/v1/cry_feedback"
        "?select=id,recording_id,predicted_cry_type,corrected_cry_type,"
        "recordings(file_url)"
        "&order=created_at.asc"
    )
    resp = httpx.get(url, headers=_headers(), timeout=30)
    resp.raise_for_status()
    return resp.json()


def download_audio(file_url: str) -> bytes:
    """Download raw audio bytes from any URL (Supabase storage or otherwise)."""
    resp = httpx.get(file_url, timeout=60, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


# ---------------------------------------------------------------------------
# Feature extraction (mirrors services/audio.py + services/classification.py)
# ---------------------------------------------------------------------------

def extract_features_from_audio(audio: np.ndarray, sr: int) -> np.ndarray | None:
    """Extract the 133-dimensional feature vector used during training."""
    try:
        from services.audio import extract_mfcc, analyze_pitch_and_frequency

        mfcc_data = extract_mfcc(audio, sr)
        pitch_data = analyze_pitch_and_frequency(audio, sr)

        mfcc_mean = mfcc_data.get("mfcc_mean", [])
        mfcc_std = mfcc_data.get("mfcc_std", [])
        delta_mean = mfcc_data.get("delta_mfcc_mean", [])
        delta_std = mfcc_data.get("delta_mfcc_std", [])
        delta2_mean = mfcc_data.get("delta2_mfcc_mean", [])
        delta2_std = mfcc_data.get("delta2_mfcc_std", [])

        spectral = [
            pitch_data.get("spectral_centroid_mean", 0),
            pitch_data.get("spectral_bandwidth_mean", 0),
            pitch_data.get("spectral_rolloff_mean", 0),
            pitch_data.get("zero_crossing_rate_mean", 0),
        ]
        contrast = pitch_data.get("spectral_contrast_mean", [0] * 5)
        chroma = pitch_data.get("chroma_mean", [0] * 12)
        rms = [pitch_data.get("rms_mean", 0)]
        pitch_range = [
            pitch_data.get("pitch_min", 0),
            pitch_data.get("pitch_max", 0),
        ]

        feature_vector = (
            list(mfcc_mean)
            + list(mfcc_std)
            + list(delta_mean)
            + list(delta_std)
            + list(delta2_mean)
            + list(delta2_std)
            + spectral
            + list(contrast)
            + list(chroma)
            + rms
            + pitch_range
        )
        return np.array(feature_vector, dtype=np.float32)

    except Exception as exc:
        print(f"  [warn] feature extraction failed: {exc}")
        return None


def audio_bytes_to_array(raw: bytes) -> tuple[np.ndarray, int] | tuple[None, None]:
    try:
        audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        return audio, sr
    except Exception as exc:
        print(f"  [warn] librosa.load failed: {exc}")
        return None, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not SUPABASE_URL or not SUPABASE_KEY:
        print("ERROR: Set SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables.")
        sys.exit(1)

    # 1. Load base training dataset
    print(f"Loading base dataset: {DATASET_JSON}")
    if not DATASET_JSON.exists():
        print(f"ERROR: {DATASET_JSON} not found. Run augment_dataset.py first.")
        sys.exit(1)

    with open(DATASET_JSON) as f:
        training_data: list[dict] = json.load(f)
    print(f"  {len(training_data)} base samples loaded.")

    # 2. Fetch feedback from Supabase
    print("\nFetching feedback from Supabase...")
    try:
        rows = fetch_feedback()
    except Exception as exc:
        print(f"ERROR fetching feedback: {exc}")
        sys.exit(1)

    print(f"  {len(rows)} feedback rows found.")
    if not rows:
        print("No feedback to incorporate. Nothing to retrain.")
        return

    # 3. Build corrected samples
    feedback_samples: list[dict] = []
    for i, row in enumerate(rows, 1):
        recording = row.get("recordings") or {}
        file_url = recording.get("file_url") if isinstance(recording, dict) else None
        corrected = row.get("corrected_cry_type", "")
        predicted = row.get("predicted_cry_type", "")

        if not file_url or not corrected:
            print(f"  [{i}/{len(rows)}] skip — missing file_url or corrected_cry_type")
            continue

        # Skip "confirmed correct" feedback (predicted == corrected) unless you
        # want to reinforce correct predictions (uncomment next lines):
        # if predicted == corrected:
        #     print(f"  [{i}/{len(rows)}] skip — confirmed correct ({corrected})")
        #     continue

        print(f"  [{i}/{len(rows)}] downloading {corrected} ({file_url[-40:]})")
        try:
            raw = download_audio(file_url)
        except Exception as exc:
            print(f"    [warn] download failed: {exc}")
            continue

        audio, sr = audio_bytes_to_array(raw)
        if audio is None:
            continue

        features = extract_features_from_audio(audio, sr)
        if features is None:
            continue

        feedback_samples.append({"label": corrected, "features": features.tolist()})

    print(f"\n  {len(feedback_samples)} feedback samples extracted successfully.")

    if not feedback_samples:
        print("No usable feedback samples. Exiting without retraining.")
        return

    # 4. Merge and re-balance
    # Repeat each feedback sample a few times so they're not drowned out
    FEEDBACK_WEIGHT = 5  # treat each corrected sample as 5 training examples
    merged = training_data + feedback_samples * FEEDBACK_WEIGHT
    print(f"  Total merged samples: {len(merged)}")

    counts = Counter(s["label"] for s in merged)
    print("\nSamples per label (after merge):")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:20} {cnt}")

    # 5. Retrain
    from services.classification import BabyCryClassifier, set_model, DEFAULT_CRY_TYPES
    from train_model import _print_results

    results = {}
    for model_type in ("random_forest", "gradient_boosting", "xgboost", "voting"):
        print(f"\nTraining {model_type.replace('_', ' ').title()}...")
        clf = BabyCryClassifier(model_type=model_type, cry_types=DEFAULT_CRY_TYPES)
        res = clf.train(training_data=merged, test_size=0.2, validation_size=0.1)
        results[model_type] = (clf, res)
        _print_results(model_type.replace("_", " ").title(), res, counts)

    best_type = max(results, key=lambda k: results[k][1]["metrics"]["test_accuracy"])
    best_clf, best_res = results[best_type]

    print(f"\nWinner: {best_type}  (accuracy={best_res['metrics']['test_accuracy']:.4f})")

    model_path = best_clf.save(MODEL_NAME)
    set_model(best_clf, model_path)
    print(f"Model saved to: {model_path}")
    print("\nDone. Restart the API server to load the new model.")


if __name__ == "__main__":
    main()
