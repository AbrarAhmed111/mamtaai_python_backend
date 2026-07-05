"""
Augment the training dataset to simulate real-world microphone recording conditions.

For each audio file in datasets/merged/, generates 2 augmented copies:
  - variant 0: room reverb + background noise
  - variant 1: noise + phone mic coloring

This triples the effective training set (~22,000 samples) and teaches the model
that hungry/pain/tired audio still sounds the same when recorded through a mic
in a room — fixing the major source of live-recording misclassification.

Run from mamtaai_python_backend/:
    python augment_dataset.py
"""
import shutil
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path
from utils.augmentation import augment_sample
from utils.dataset_preparation import prepare_dataset_from_directory

MERGED_DIR   = Path("datasets/merged")
AUG_DIR      = Path("datasets/merged_augmented")
OUTPUT_JSON  = Path("datasets/training_dataset.json")

VARIANTS = [0, 1]   # two augmented copies per original file
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}


def build_augmented_dir():
    if AUG_DIR.exists():
        shutil.rmtree(AUG_DIR)
    AUG_DIR.mkdir(parents=True)

    labels = [d for d in MERGED_DIR.iterdir() if d.is_dir()]
    total_original = sum(
        len([f for f in lbl.iterdir() if f.suffix.lower() in AUDIO_EXTS])
        for lbl in labels
    )
    print(f"Original files: {total_original}")
    print(f"Will produce:   {total_original * (1 + len(VARIANTS))} total files "
          f"(original + {len(VARIANTS)} augmented per file)")
    print()

    processed = 0
    errors = 0

    for label_dir in sorted(labels):
        label = label_dir.name
        out_label = AUG_DIR / label
        out_label.mkdir()

        audio_files = [f for f in label_dir.iterdir() if f.suffix.lower() in AUDIO_EXTS]

        for src in audio_files:
            processed += 1
            pct = processed / total_original * 100
            print(f"  [{processed}/{total_original}] ({pct:.1f}%)  {label}/{src.name}")

            try:
                audio, sr = librosa.load(str(src), sr=None, mono=True)
            except Exception as e:
                print(f"    [SKIP] Could not load: {e}")
                errors += 1
                continue

            # Copy original as WAV
            orig_dest = out_label / (src.stem + ".wav")
            sf.write(str(orig_dest), audio, sr)

            # Write augmented variants
            for v in VARIANTS:
                try:
                    aug_audio = augment_sample(audio, sr, variant=v)
                    aug_name = f"{src.stem}_aug{v}.wav"
                    sf.write(str(out_label / aug_name), aug_audio, sr)
                except Exception as e:
                    print(f"    [WARN] variant {v} failed: {e}")

    print(f"\nDone. {errors} files skipped due to load errors.")
    counts = {d.name: len(list(d.iterdir())) for d in AUG_DIR.iterdir() if d.is_dir()}
    print("\nFiles per label in augmented dir:")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:20} {cnt}")
    return counts


def main():
    print("=" * 60)
    print("Step 1: Building augmented audio dataset")
    print("=" * 60)
    build_augmented_dir()

    print()
    print("=" * 60)
    print("Step 2: Extracting features from augmented dataset")
    print(f"Output: {OUTPUT_JSON}")
    print("=" * 60)

    training_data = prepare_dataset_from_directory(
        dataset_dir=str(AUG_DIR),
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
    print("Samples per label:")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:20} {cnt}")
    print("\nNow run: python train_model.py")


if __name__ == "__main__":
    main()
