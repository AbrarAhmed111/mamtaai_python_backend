"""
Re-extract features from the already-merged audio files (datasets/merged/).
Use this after enhancing the feature extraction pipeline — no need to re-download
or re-merge datasets, just re-run feature extraction with the new richer features.

Run from mamtaai_python_backend/:
    python reextract_features.py
"""
from pathlib import Path
from utils.dataset_preparation import prepare_dataset_from_directory

MERGED_DIR  = Path("datasets/merged")
OUTPUT_JSON = Path("datasets/training_dataset.json")


def main():
    if not MERGED_DIR.exists():
        print(f"ERROR: {MERGED_DIR} does not exist — run prepare_all_datasets.py first.")
        return

    labels = [d.name for d in MERGED_DIR.iterdir() if d.is_dir()]
    total = sum(
        len([f for f in (MERGED_DIR / lbl).iterdir() if f.suffix.lower() in
             {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}])
        for lbl in labels
    )
    print(f"Re-extracting features for {total} audio files in {MERGED_DIR}/")
    print(f"Labels: {sorted(labels)}")
    print(f"Output: {OUTPUT_JSON}\n")

    training_data = prepare_dataset_from_directory(
        dataset_dir=str(MERGED_DIR),
        output_file=str(OUTPUT_JSON),
        audio_format="wav",
        n_mfcc=13,
        remove_noise=True,
        normalize=True,
        label_from_folder=True,
        keep_full_features=False,
    )

    from collections import Counter
    counts = Counter(s["label"] for s in training_data)
    print(f"\nDone — {len(training_data)} samples saved to {OUTPUT_JSON}")
    print("Counts:")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:20} {cnt}")


if __name__ == "__main__":
    main()
