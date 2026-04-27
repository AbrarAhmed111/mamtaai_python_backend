"""
Dataset Preparation Utility
Helps prepare training datasets for baby cry classification from audio files.
"""
import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import librosa
import numpy as np

from services.audio import convert_audio_format, extract_features, preprocess_audio


def prepare_dataset_from_directory(
    dataset_dir: str,
    output_file: Optional[str] = None,
    audio_format: str = "wav",
    n_mfcc: int = 13,
    remove_noise: bool = True,
    normalize: bool = True,
    label_from_folder: bool = True,
    label_mapping: Optional[Dict[str, str]] = None,
    keep_full_features: bool = False
) -> List[Dict]:
    """
    Prepare training dataset from a directory structure.
    
    Expected directory structure:
    dataset/
        hungry/
            audio1.wav
            audio2.wav
        tired/
            audio1.wav
            audio2.wav
        ...
    
    Or flat structure with label_mapping:
    dataset/
        hungry_audio1.wav
        hungry_audio2.wav
        tired_audio1.wav
        ...
    
    Args:
        dataset_dir: Path to dataset directory
        output_file: Optional path to save prepared dataset as JSON
        audio_format: Audio file format (wav, mp3, m4a, etc.)
        n_mfcc: Number of MFCC coefficients
        remove_noise: Whether to remove noise during preprocessing
        normalize: Whether to normalize audio
        label_from_folder: If True, use folder names as labels. If False, use label_mapping
        label_mapping: Dictionary mapping filenames/patterns to labels
        keep_full_features: If True, keep large arrays (MFCC coefficients, spectrogram)
    
    Returns:
        List of training samples with 'features' and 'label' keys
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        raise ValueError(f"Dataset directory does not exist: {dataset_dir}")
    
    training_data = []
    supported_formats = ['.wav', '.mp3', '.m4a', '.ogg', '.flac', '.webm']
    
    if label_from_folder:
        # Process directory structure: each subfolder is a label
        # First, count total files for progress tracking
        total_files = 0
        label_file_counts = {}
        for label_folder in dataset_path.iterdir():
            if not label_folder.is_dir():
                continue
            label = label_folder.name
            audio_files = [
                f for f in label_folder.iterdir()
                if f.suffix.lower() in supported_formats
            ]
            label_file_counts[label] = len(audio_files)
            total_files += len(audio_files)
        
        print(f"Total audio files to process: {total_files}")
        print("=" * 60)
        
        processed_files = 0
        # Process directory structure: each subfolder is a label
        for label_folder in dataset_path.iterdir():
            if not label_folder.is_dir():
                continue
            
            label = label_folder.name
            print(f"\nProcessing label: {label} ({label_file_counts.get(label, 0)} files)")
            
            audio_files = [
                f for f in label_folder.iterdir()
                if f.suffix.lower() in supported_formats
            ]
            
            for idx, audio_file in enumerate(audio_files, 1):
                try:
                    processed_files += 1
                    progress = (processed_files / total_files) * 100
                    print(f"  [{processed_files}/{total_files}] ({progress:.1f}%) Processing: {audio_file.name}")
                    
                    sample = _process_audio_file(
                        str(audio_file),
                        label,
                        audio_format,
                        n_mfcc,
                        remove_noise,
                        normalize,
                        keep_full_features
                    )
                    if sample:
                        training_data.append(sample)
                except Exception as e:
                    print(f"  Error processing {audio_file.name}: {str(e)}")
                    continue
    else:
        # Process flat structure with label_mapping
        if not label_mapping:
            raise ValueError("label_mapping required when label_from_folder=False")
        
        audio_files = [
            f for f in dataset_path.iterdir()
            if f.is_file() and f.suffix.lower() in supported_formats
        ]
        
        total_files = len(audio_files)
        print(f"Total audio files to process: {total_files}")
        print("=" * 60)
        
        processed_files = 0
        for audio_file in audio_files:
            # Find matching label
            label = None
            for pattern, mapped_label in label_mapping.items():
                if pattern in audio_file.name:
                    label = mapped_label
                    break
            
            if not label:
                print(f"  Skipping {audio_file.name}: no label mapping found")
                continue
            
            try:
                processed_files += 1
                progress = (processed_files / total_files) * 100
                print(f"  [{processed_files}/{total_files}] ({progress:.1f}%) Processing: {audio_file.name} -> {label}")
                
                sample = _process_audio_file(
                    str(audio_file),
                    label,
                    audio_format,
                    n_mfcc,
                    remove_noise,
                    normalize,
                    keep_full_features
                )
                if sample:
                    training_data.append(sample)
            except Exception as e:
                print(f"  Error processing {audio_file.name}: {str(e)}")
                continue
    
    print(f"\nTotal samples prepared: {len(training_data)}")
    
    # Count samples per label
    label_counts = {}
    for sample in training_data:
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nSamples per label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"\nDataset saved to: {output_file}")
    
    return training_data


def prepare_dataset_from_csv(
    csv_file: str,
    audio_dir: str,
    output_file: Optional[str] = None,
    audio_format: str = "wav",
    n_mfcc: int = 13,
    remove_noise: bool = True,
    normalize: bool = True,
    filename_column: str = "filename",
    label_column: str = "label",
    keep_full_features: bool = False
) -> List[Dict]:
    """
    Prepare training dataset from a CSV file.
    
    CSV format:
    filename,label
    audio1.wav,hungry
    audio2.wav,tired
    ...
    
    Args:
        csv_file: Path to CSV file with filename and label columns
        audio_dir: Directory containing audio files
        output_file: Optional path to save prepared dataset as JSON
        audio_format: Audio file format
        n_mfcc: Number of MFCC coefficients
        remove_noise: Whether to remove noise during preprocessing
        normalize: Whether to normalize audio
        filename_column: Name of the filename column in CSV
        label_column: Name of the label column in CSV
        keep_full_features: If True, keep large arrays (MFCC coefficients, spectrogram)
    
    Returns:
        List of training samples with 'features' and 'label' keys
    """
    audio_path = Path(audio_dir)
    if not audio_path.exists():
        raise ValueError(f"Audio directory does not exist: {audio_dir}")
    
    training_data = []
    
    # Read CSV
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if filename_column not in rows[0] or label_column not in rows[0]:
        raise ValueError(f"CSV must contain '{filename_column}' and '{label_column}' columns")
    
    total_rows = len(rows)
    print(f"Processing {total_rows} entries from CSV...")
    print("=" * 60)
    
    processed_rows = 0
    for row in rows:
        filename = row[filename_column]
        label = row[label_column]
        
        audio_file = audio_path / filename
        if not audio_file.exists():
            print(f"  Skipping {filename}: file not found")
            continue
        
        try:
            processed_rows += 1
            progress = (processed_rows / total_rows) * 100
            print(f"  [{processed_rows}/{total_rows}] ({progress:.1f}%) Processing: {filename} -> {label}")
            sample = _process_audio_file(
                str(audio_file),
                label,
                audio_format,
                n_mfcc,
                remove_noise,
                normalize,
                keep_full_features
            )
            if sample:
                training_data.append(sample)
        except Exception as e:
            print(f"  Error processing {filename}: {str(e)}")
            continue
    
    print(f"\nTotal samples prepared: {len(training_data)}")
    
    # Count samples per label
    label_counts = {}
    for sample in training_data:
        label = sample['label']
        label_counts[label] = label_counts.get(label, 0) + 1
    
    print("\nSamples per label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    
    # Save to file if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(training_data, f, indent=2)
        print(f"\nDataset saved to: {output_file}")
    
    return training_data


def load_prepared_dataset(json_file: str) -> List[Dict]:
    """
    Load a previously prepared dataset from JSON file.
    
    Args:
        json_file: Path to JSON file containing training data
    
    Returns:
        List of training samples
    """
    with open(json_file, 'r') as f:
        return json.load(f)


def _process_audio_file(
    audio_file: str,
    label: str,
    audio_format: str,
    n_mfcc: int,
    remove_noise: bool,
    normalize: bool,
    keep_full_features: bool
) -> Optional[Dict]:
    """
    Process a single audio file and extract features.
    
    Args:
        audio_file: Path to audio file
        label: Label for this audio file
        audio_format: Audio file format
        n_mfcc: Number of MFCC coefficients
        remove_noise: Whether to remove noise
        normalize: Whether to normalize
    
    Returns:
        Dictionary with 'features' and 'label' keys, or None if processing fails
    """
    try:
        # Read audio file
        with open(audio_file, 'rb') as f:
            audio_bytes = f.read()
        
        # Preprocess audio
        preprocessing_result = preprocess_audio(
            audio_data=audio_bytes,
            input_format=audio_format,
            remove_noise_flag=remove_noise,
            normalize_flag=normalize,
            segment_flag=False
        )
        
        # Convert processed audio back to numpy array for feature extraction
        from services.audio import convert_audio_format
        processed_audio, sample_rate = convert_audio_format(
            preprocessing_result["processed_audio"],
            input_format="wav"
        )
        
        # Extract features
        features = extract_features(processed_audio, sample_rate, n_mfcc=n_mfcc)
        if not keep_full_features:
            features = _compact_features(features)
        
        return {
            "features": features,
            "label": label
        }
    
    except Exception as e:
        print(f"    Error: {str(e)}")
        return None


def _compact_features(features: Dict) -> Dict:
    """
    Remove large arrays from the feature set to keep JSON small and fast to write.
    """
    compact = {}

    mfcc = features.get("mfcc", {})
    if mfcc:
        compact["mfcc"] = {
            "mfcc_mean": mfcc.get("mfcc_mean", []),
            "mfcc_std": mfcc.get("mfcc_std", []),
            "num_coefficients": mfcc.get("num_coefficients", 0),
            "num_frames": mfcc.get("num_frames", 0)
        }

    spectrogram = features.get("spectrogram", {})
    if spectrogram:
        compact["spectrogram"] = {
            "magnitude_mean": spectrogram.get("magnitude_mean", 0),
            "magnitude_max": spectrogram.get("magnitude_max", 0),
            "magnitude_min": spectrogram.get("magnitude_min", 0)
        }

    if "pitch_frequency" in features:
        compact["pitch_frequency"] = features["pitch_frequency"]

    if "duration" in features:
        compact["duration"] = features["duration"]

    return compact


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Prepare dataset for baby cry classification",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare from directory structure
  python -m utils.dataset_preparation --dataset-dir /path/to/dataset --output dataset.json
  
  # Prepare from CSV file
  python -m utils.dataset_preparation --csv-file labels.csv --audio-dir /path/to/audio --output dataset.json
  
  # For Donate-a-Cry Corpus (after mapping labels)
  python -m utils.dataset_preparation --dataset-dir /path/to/donate_a_cry_mapped --output dacry_dataset.json
        """
    )
    parser.add_argument("--dataset-dir", type=str, help="Directory containing audio files")
    parser.add_argument("--csv-file", type=str, help="CSV file with filename and label columns")
    parser.add_argument("--audio-dir", type=str, help="Directory containing audio files (for CSV mode)")
    parser.add_argument("--output", type=str, default="dataset.json", help="Output JSON file")
    parser.add_argument("--audio-format", type=str, default="wav", help="Audio file format")
    parser.add_argument("--n-mfcc", type=int, default=13, help="Number of MFCC coefficients")
    parser.add_argument("--no-remove-noise", action="store_true", help="Don't remove noise")
    parser.add_argument("--no-normalize", action="store_true", help="Don't normalize audio")
    parser.add_argument(
        "--keep-full-features",
        action="store_true",
        help="Keep large arrays (MFCC coefficients, spectrogram) in output JSON"
    )
    
    args = parser.parse_args()
    
    if args.csv_file:
        # CSV mode
        if not args.audio_dir:
            parser.error("--audio-dir required when using --csv-file")
        
        prepare_dataset_from_csv(
            csv_file=args.csv_file,
            audio_dir=args.audio_dir,
            output_file=args.output,
            audio_format=args.audio_format,
            n_mfcc=args.n_mfcc,
            remove_noise=not args.no_remove_noise,
            normalize=not args.no_normalize,
            keep_full_features=args.keep_full_features
        )
    elif args.dataset_dir:
        # Directory mode
        prepare_dataset_from_directory(
            dataset_dir=args.dataset_dir,
            output_file=args.output,
            audio_format=args.audio_format,
            n_mfcc=args.n_mfcc,
            remove_noise=not args.no_remove_noise,
            normalize=not args.no_normalize,
            keep_full_features=args.keep_full_features
        )
    else:
        parser.error("Either --dataset-dir or --csv-file must be provided")





