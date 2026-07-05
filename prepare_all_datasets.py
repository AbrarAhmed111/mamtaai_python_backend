"""
Remap, merge, and extract features from all downloaded datasets into a single
training JSON file ready for model training.

Run from mamtaai_python_backend/:
    python prepare_all_datasets.py

Balancing targets (1,500 samples per class):
  - Majority classes are capped via MAX_PER_LABEL (undersampling).
  - Minority classes are boosted by pulling in every available source file.
  - 'attention' is excluded — insufficient public data exists for this class.
"""
import random
import shutil
from pathlib import Path
from utils.dataset_preparation import prepare_dataset_from_directory

random.seed(42)

DATASETS_DIR = Path("datasets")
MERGED_DIR   = DATASETS_DIR / "merged"
OUTPUT_JSON  = DATASETS_DIR / "training_dataset.json"

TARGET = 1500

# Hard cap per label during the merge (file-copy) stage.
# Labels not listed here are uncapped (collect everything available).
MAX_PER_LABEL = {
    "hungry":     TARGET,  # 2,702 available → cap to 1,500
    "discomfort": TARGET,  # 1,743 available → cap to 1,500
    "pain":       TARGET,  # 2,071 available after large dataset → cap to 1,500
    "overstimulated": TARGET,  # 1,521 available after large dataset → cap to 1,500
}

# ---------------------------------------------------------------------------
# Label mapping rules applied across all datasets.
# Keys are source folder names (case-sensitive). None = skip that folder.
# 'attention' is intentionally absent — class removed from the pipeline.
# ---------------------------------------------------------------------------
LABEL_MAP = {
    # direct matches
    "hungry":         "hungry",
    "tired":          "tired",
    "discomfort":     "discomfort",
    "pain":           "pain",
    # remapped
    "belly_pain":     "pain",
    "belly pain":     "pain",
    "physical_pain":  "pain",
    "burping":        "discomfort",
    "cold_hot":       "discomfort",
    "scared":         "overstimulated",
    # skip — removed class or no useful mapping
    "attention":      None,
    "lonely":         None,   # was attention proxy — class removed
    "laugh":          None,
    # skip — too ambiguous or non-cry
    "needs":          None,
    "noise":          None,
    "silence":        None,
    "cry":            None,
    "not_cry":        None,
}

# Each entry: (source_dir_relative_to_datasets, description)
SOURCES = [
    ("donateacry-corpus",                          "Donate-a-Cry Corpus"),
    ("infant-cry-corpus/donateacry_corpus",        "Infant Cry Audio Corpus"),
    ("baby-crying-sounds/Baby Crying Sounds",      "Baby Crying Sounds"),
    ("baby-cry-sense/Baby Cry Sence Dataset",      "Baby Cry Sense Dataset"),
    ("ESC-50-baby-cry",                            "ESC-50 Baby Cry subset"),
    ("infant-cry-4classes/audio",                  "Infant Cry 4-Classes"),
    ("baby-cry-pattern-archive/cry",               "Baby Cry Pattern Archive"),
    ("baby-cry-sounds2/Baby Dataset",              "Baby Cry Sounds 2"),
    ("decoding-cries-baby/BABY CRY",               "Decoding Cries Baby"),
    # New sources — major boost for tired, overstimulated, pain
    ("baby-cry-sense2/Baby Cry Dataset",           "Baby Cry Sense v2"),
    ("baby-crying-dataset-large/Baby crying",      "Baby Crying Dataset Large"),
]


def merge_datasets():
    """
    Copy audio files from every source into merged/<label>/ folders.
    Labels in MAX_PER_LABEL are randomly downsampled to their cap before
    any files are written, so the merged folder never exceeds the target.
    """
    if MERGED_DIR.exists():
        shutil.rmtree(MERGED_DIR)
    MERGED_DIR.mkdir(parents=True)

    audio_exts = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}

    # First pass: collect all file paths per mapped label across every source
    all_files: dict[str, list[tuple[Path, str]]] = {}  # label -> [(file, dest_name)]

    for rel_path, desc in SOURCES:
        src = DATASETS_DIR / rel_path
        if not src.exists():
            print(f"  [SKIP] Not found: {src}")
            continue

        print(f"\n--- {desc} ---")
        slug = rel_path.replace("/", "_").replace(" ", "_")

        for folder in src.iterdir():
            if not folder.is_dir():
                continue

            mapped = LABEL_MAP.get(folder.name)
            if mapped is None:
                if folder.name in LABEL_MAP:
                    print(f"  [SKIP] {folder.name}")
                else:
                    print(f"  [UNKNOWN] {folder.name} — not in LABEL_MAP, skipping")
                continue

            files = [f for f in folder.iterdir() if f.suffix.lower() in audio_exts]
            entries = [(f, f"{slug}__{f.name}") for f in files]

            if mapped not in all_files:
                all_files[mapped] = []
            all_files[mapped].extend(entries)
            print(f"  {folder.name:20} -> {mapped:15} (+{len(files)} files)")

    print(f"\n{'='*60}")
    print("Pre-cap totals:")
    for label, entries in sorted(all_files.items()):
        cap = MAX_PER_LABEL.get(label)
        cap_str = f"  -> will cap to {cap}" if cap else ""
        print(f"  {label:20} {len(entries):5}{cap_str}")

    # Second pass: apply undersampling cap, then copy
    print(f"\n{'='*60}")
    print("Copying files to merged/...")
    total_copied = 0
    label_counts: dict[str, int] = {}

    for label, entries in sorted(all_files.items()):
        cap = MAX_PER_LABEL.get(label)
        if cap and len(entries) > cap:
            entries = random.sample(entries, cap)

        dest_dir = MERGED_DIR / label
        dest_dir.mkdir(exist_ok=True)

        copied = 0
        seen_names: set[str] = set()
        for src_file, dest_name in entries:
            # resolve duplicate dest names across sources
            if dest_name in seen_names:
                dest_name = f"{src_file.parent.name}__{dest_name}"
            seen_names.add(dest_name)
            shutil.copy2(src_file, dest_dir / dest_name)
            copied += 1

        label_counts[label] = copied
        total_copied += copied

    print(f"\nFinal merged counts (target = {TARGET}):")
    for label, count in sorted(label_counts.items()):
        gap = TARGET - count
        status = "OK" if count >= TARGET else f"{gap} SHORT"
        bar = "#" * (count // 20)
        print(f"  {label:20} {count:5}  [{status}]  {bar}")

    print(f"\nTotal: {total_copied} files in {MERGED_DIR}/")
    return label_counts


def main():
    print("=" * 60)
    print("Step 1: Merging, remapping, and balancing datasets")
    print(f"        Target: {TARGET} samples per class")
    print("=" * 60)
    label_counts = merge_datasets()

    if not any(label_counts.values()):
        print("No files were merged — check DATASETS_DIR paths.")
        return

    print(f"\n{'='*60}")
    print("Step 2: Extracting audio features")
    print(f"Output: {OUTPUT_JSON}")
    print("=" * 60)
    print("This will take a while (each file is fully processed)...\n")

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

    print(f"\n{'='*60}")
    print(f"DONE — {len(training_data)} samples saved to {OUTPUT_JSON}")
    print("=" * 60)
    print("\nFinal sample counts:")
    from collections import Counter
    counts = Counter(s["label"] for s in training_data)
    for label, count in sorted(counts.items()):
        gap = TARGET - count
        status = "OK" if count >= TARGET else f"{gap} SHORT"
        print(f"  {label:20} {count:5}  [{status}]")


if __name__ == "__main__":
    main()
