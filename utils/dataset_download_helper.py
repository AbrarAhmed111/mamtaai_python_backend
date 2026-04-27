"""
Dataset Download Helper
Provides utilities to help download and prepare well-known baby cry datasets.
"""
import os
import json
import requests
from pathlib import Path
from typing import Dict, Optional
import zipfile
import shutil


def download_kaggle_dataset(
    kaggle_dataset_name: str,
    output_dir: str,
    kaggle_username: Optional[str] = None,
    kaggle_key: Optional[str] = None
) -> str:
    """
    Download a dataset from Kaggle.
    
    Requires Kaggle API credentials. Set them up:
    1. Go to https://www.kaggle.com/account
    2. Create API token
    3. Place kaggle.json in ~/.kaggle/ or provide username/key
    
    Args:
        kaggle_dataset_name: Dataset name (e.g., "username/dataset-name")
        output_dir: Directory to save downloaded dataset
        kaggle_username: Kaggle username (or use kaggle.json)
        kaggle_key: Kaggle API key (or use kaggle.json)
    
    Returns:
        Path to downloaded dataset directory
    """
    try:
        import kaggle
    except ImportError:
        raise ImportError(
            "kaggle package required. Install with: pip install kaggle\n"
            "Then set up credentials: https://www.kaggle.com/docs/api"
        )
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading Kaggle dataset: {kaggle_dataset_name}")
    print(f"Output directory: {output_path}")
    
    # Download dataset
    kaggle.api.dataset_download_files(
        kaggle_dataset_name,
        path=str(output_path),
        unzip=True
    )
    
    print(f"[OK] Dataset downloaded to: {output_path}")
    return str(output_path)


def map_baby_crying_sounds_labels(
    dataset_dir: str,
    output_dir: Optional[str] = None,
    include_laugh: bool = False
) -> Dict[str, str]:
    """
    Map Baby Crying Sounds dataset labels to your system labels.
    
    Expected structure:
    dataset/
        belly pain/
        burping/
        cold_hot/
        discomfort/
        hungry/
        laugh/
        noise/
        silence/
        tired/
    
    Creates mapped structure:
    output/
        hungry/
        tired/
        discomfort/
        pain/
        attention/ (optional)
    
    Args:
        dataset_dir: Path to Baby Crying Sounds dataset
        output_dir: Output directory for mapped dataset (default: dataset_dir + "_mapped")
        include_laugh: Whether to include laugh sounds as attention (default: False)
    
    Returns:
        Dictionary mapping old labels to new labels
    """
    source_path = Path(dataset_dir)
    if not source_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    if output_dir is None:
        output_dir = str(source_path.parent / f"{source_path.name}_mapped")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Label mapping for Baby Crying Sounds dataset
    label_mapping = {
        "hungry": "hungry",                    # Direct match
        "tired": "tired",                      # Direct match
        "discomfort": "discomfort",            # Direct match
        "belly pain": "pain",                  # Maps to pain
        "cold_hot": "discomfort",              # Temperature discomfort
        "burping": "discomfort",               # Digestive discomfort
        "laugh": "attention" if include_laugh else None,  # Optional attention
        # Skip: noise, silence
    }
    
    print("Mapping Baby Crying Sounds dataset labels...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    if include_laugh:
        print("Including laugh sounds as attention")
    else:
        print("Skipping laugh, noise, and silence")
    print()
    
    mapped_count = 0
    skipped_count = 0
    
    # Process each folder
    for folder in source_path.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        mapped_label = label_mapping.get(folder_name)
        
        # Skip if not in mapping (noise, silence, or laugh if not included)
        if mapped_label is None:
            print(f"  [SKIP] Skipping: {folder_name} (not mapped)")
            skipped_count += 1
            continue
        
        # Create output folder
        output_label_folder = output_path / mapped_label
        output_label_folder.mkdir(exist_ok=True)
        
        # Copy audio files
        audio_files = [f for f in folder.iterdir() if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm']]
        
        for audio_file in audio_files:
            dest_file = output_label_folder / audio_file.name
            # Handle filename conflicts
            if dest_file.exists():
                # Add prefix to avoid overwriting
                dest_file = output_label_folder / f"{folder_name}_{audio_file.name}"
            shutil.copy2(audio_file, dest_file)
            mapped_count += 1
        
        print(f"  [OK] {folder_name:15} -> {mapped_label:12} ({len(audio_files)} files)")
    
    print(f"\n[OK] Mapping complete!")
    print(f"  Mapped: {mapped_count} files")
    print(f"  Skipped: {skipped_count} folders")
    print(f"  Output: {output_path}")
    
    # Show final distribution
    print(f"\nFinal distribution:")
    for label_folder in sorted(output_path.iterdir()):
        if label_folder.is_dir():
            count = len(list(label_folder.glob("*.*")))
            print(f"  {label_folder.name}: {count} files")
    
    return label_mapping


def map_donate_a_cry_labels(dataset_dir: str, output_dir: Optional[str] = None) -> Dict[str, str]:
    """
    Map Donate-a-Cry Corpus labels to your system labels.
    
    Expected Donate-a-Cry structure:
    dataset/
        Hunger/
        Pain/
        Sleepiness/
        Discomfort/
        Normal/
    
    Creates mapped structure:
    output/
        hungry/
        pain/
        tired/
        discomfort/
        attention/
    
    Args:
        dataset_dir: Path to Donate-a-Cry dataset
        output_dir: Output directory for mapped dataset (default: dataset_dir + "_mapped")
    
    Returns:
        Dictionary mapping old labels to new labels
    """
    source_path = Path(dataset_dir)
    if not source_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    if output_dir is None:
        output_dir = str(source_path.parent / f"{source_path.name}_mapped")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Label mapping
    label_mapping = {
        "Hunger": "hungry",
        "Pain": "pain",
        "Sleepiness": "tired",
        "Discomfort": "discomfort",
        "Normal": "attention",  # Map normal cries to attention
        # Handle variations
        "hunger": "hungry",
        "pain": "pain",
        "sleepiness": "tired",
        "sleepy": "tired",
        "discomfort": "discomfort",
    }
    
    print("Mapping Donate-a-Cry labels...")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    mapped_count = 0
    skipped_count = 0
    
    # Process each folder
    for folder in source_path.iterdir():
        if not folder.is_dir():
            continue
        
        folder_name = folder.name
        mapped_label = label_mapping.get(folder_name, None)
        
        if not mapped_label:
            # Try case-insensitive match
            folder_lower = folder_name.lower()
            for key, value in label_mapping.items():
                if key.lower() == folder_lower:
                    mapped_label = value
                    break
        
        if not mapped_label:
            print(f"  ⚠ Skipping unknown label: {folder_name}")
            skipped_count += 1
            continue
        
        # Create output folder
        output_label_folder = output_path / mapped_label
        output_label_folder.mkdir(exist_ok=True)
        
        # Copy audio files
        audio_files = [f for f in folder.iterdir() if f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.ogg']]
        
        for audio_file in audio_files:
            dest_file = output_label_folder / audio_file.name
            shutil.copy2(audio_file, dest_file)
            mapped_count += 1
        
        print(f"  [OK] {folder_name} -> {mapped_label} ({len(audio_files)} files)")
    
    print(f"\n[OK] Mapping complete!")
    print(f"  Mapped: {mapped_count} files")
    print(f"  Skipped: {skipped_count} folders")
    print(f"  Output: {output_path}")
    
    return label_mapping


def create_dataset_info(dataset_dir: str, output_file: Optional[str] = None) -> Dict:
    """
    Create a summary/info file for a dataset.
    
    Args:
        dataset_dir: Path to dataset directory
        output_file: Path to save info JSON (default: dataset_dir/info.json)
    
    Returns:
        Dictionary with dataset information
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory not found: {dataset_dir}")
    
    info = {
        "dataset_path": str(dataset_path),
        "labels": {},
        "total_files": 0,
        "supported_formats": ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm']
    }
    
    # Count files per label
    for label_folder in dataset_path.iterdir():
        if not label_folder.is_dir():
            continue
        
        label = label_folder.name
        audio_files = [
            f for f in label_folder.iterdir()
            if f.is_file() and f.suffix.lower() in info["supported_formats"]
        ]
        
        info["labels"][label] = {
            "count": len(audio_files),
            "files": [f.name for f in audio_files[:10]]  # First 10 filenames as sample
        }
        info["total_files"] += len(audio_files)
    
    # Save info
    if output_file is None:
        output_file = dataset_path / "dataset_info.json"
    else:
        output_file = Path(output_file)
    
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Dataset info saved to: {output_file}")
    print(f"\nDataset Summary:")
    print(f"  Total files: {info['total_files']}")
    print(f"  Labels: {len(info['labels'])}")
    for label, data in info["labels"].items():
        print(f"    {label}: {data['count']} files")
    
    return info


def validate_dataset_structure(dataset_dir: str) -> Dict:
    """
    Validate that a dataset has the correct structure for training.
    
    Args:
        dataset_dir: Path to dataset directory
    
    Returns:
        Dictionary with validation results
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return {
            "valid": False,
            "error": "Dataset directory not found"
        }
    
    results = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "labels": {},
        "recommendations": []
    }
    
    # Check for label folders
    label_folders = [f for f in dataset_path.iterdir() if f.is_dir()]
    
    if len(label_folders) == 0:
        results["valid"] = False
        results["errors"].append("No label folders found. Expected structure: dataset/label_name/audio_files")
        return results
    
    # Check each label folder
    min_samples = 10
    recommended_samples = 50
    
    for label_folder in label_folders:
        label = label_folder.name
        audio_files = [
            f for f in label_folder.iterdir()
            if f.is_file() and f.suffix.lower() in ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm']
        ]
        
        count = len(audio_files)
        results["labels"][label] = {
            "count": count,
            "status": "ok" if count >= min_samples else "low"
        }
        
        if count == 0:
            results["errors"].append(f"Label '{label}' has no audio files")
            results["valid"] = False
        elif count < min_samples:
            results["warnings"].append(f"Label '{label}' has only {count} samples (minimum {min_samples} recommended)")
        elif count < recommended_samples:
            results["recommendations"].append(f"Label '{label}' has {count} samples (recommended: {recommended_samples}+)")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Dataset download and preparation helpers")
    parser.add_argument("--map-donate-a-cry", type=str, help="Map Donate-a-Cry labels to system labels")
    parser.add_argument("--map-baby-crying-sounds", type=str, help="Map Baby Crying Sounds dataset labels to system labels")
    parser.add_argument("--include-laugh", action="store_true", help="Include laugh sounds as attention (for Baby Crying Sounds)")
    parser.add_argument("--output", type=str, help="Output directory for mapped dataset")
    parser.add_argument("--info", type=str, help="Create info file for dataset")
    parser.add_argument("--validate", type=str, help="Validate dataset structure")
    
    args = parser.parse_args()
    
    if args.map_baby_crying_sounds:
        map_baby_crying_sounds_labels(
            args.map_baby_crying_sounds,
            args.output,
            include_laugh=args.include_laugh
        )
    elif args.map_donate_a_cry:
        map_donate_a_cry_labels(args.map_donate_a_cry, args.output)
    elif args.info:
        create_dataset_info(args.info)
    elif args.validate:
        results = validate_dataset_structure(args.validate)
        print("\nValidation Results:")
        print(json.dumps(results, indent=2))
    else:
        parser.print_help()




