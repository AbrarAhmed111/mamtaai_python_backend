"""
Inference wrapper for the fine-tuned wav2vec2 cry classifier.
Loaded automatically by streaming.py when the model is available locally or
downloadable from HuggingFace Hub (set WAV2VEC2_HF_REPO, e.g. "user/repo").
"""
import json
import os
import threading
from pathlib import Path

import librosa
import numpy as np

_MODEL_DIR = Path(__file__).parent.parent / "models" / "wav2vec2_cry_classifier"
_HF_REPO   = os.getenv("WAV2VEC2_HF_REPO", "")

SAMPLE_RATE = 16_000
MAX_SAMPLES = 8 * SAMPLE_RATE  # 128 000 samples = 8 seconds

_REQUIRED_FILES = ("config.json", "label_map.json", "model.safetensors")

_model     = None
_extractor = None
_label_map = None
_device    = None
_load_lock = threading.Lock()


def _local_files_complete() -> bool:
    return all((_MODEL_DIR / f).exists() for f in _REQUIRED_FILES)


def _torch_importable() -> bool:
    try:
        import torch  # noqa: F401 — torch not installed on slim deploys
        return True
    except ImportError:
        return False


def is_available() -> bool:
    # Weights come either from a completed local dir or from HF Hub at first use.
    if not _torch_importable():
        return False
    return _local_files_complete() or bool(_HF_REPO)


def _ensure_weights():
    """Download model files from HuggingFace Hub if not present locally."""
    if _local_files_complete():
        return
    if not _HF_REPO:
        raise RuntimeError(
            f"[wav2vec2] Model files missing from {_MODEL_DIR} and WAV2VEC2_HF_REPO is not set"
        )
    from huggingface_hub import snapshot_download
    print(f"[wav2vec2] Downloading weights from HF Hub repo '{_HF_REPO}' ...")
    snapshot_download(
        repo_id=_HF_REPO,
        local_dir=str(_MODEL_DIR),
        token=os.getenv("HF_TOKEN") or None,  # needed only for private repos
    )
    if not _local_files_complete():
        missing = [f for f in _REQUIRED_FILES if not (_MODEL_DIR / f).exists()]
        raise RuntimeError(f"[wav2vec2] HF download finished but files missing: {missing}")


def _load():
    global _model, _extractor, _label_map, _device

    import torch
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    _ensure_weights()

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _extractor = AutoFeatureExtractor.from_pretrained(str(_MODEL_DIR))
    _model     = Wav2Vec2ForSequenceClassification.from_pretrained(str(_MODEL_DIR))
    _model.eval()
    _model.to(_device)

    try:
        with open(_MODEL_DIR / "label_map.json") as f:
            _label_map = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        raise RuntimeError(f"[wav2vec2] Failed to load label_map.json: {e}") from e

    print(f"[wav2vec2] Loaded from {_MODEL_DIR} on {_device}")


def predict(audio: np.ndarray, sample_rate: int) -> dict:
    """
    Args:
        audio       : float32 numpy array, any sample rate
        sample_rate : original sample rate of `audio`

    Returns:
        {
            "predicted_cry_type": str,
            "confidence": float,
            "probabilities": {label: float, ...}
        }
    """
    import torch

    global _model, _extractor, _label_map, _device

    if _model is None:
        with _load_lock:
            if _model is None:  # double-checked locking
                _load()

    # Resample to 16 kHz
    if sample_rate != SAMPLE_RATE:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=SAMPLE_RATE)

    # Pad or crop
    if len(audio) > MAX_SAMPLES:
        audio = audio[:MAX_SAMPLES]
    elif len(audio) < MAX_SAMPLES:
        audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))

    inputs = _extractor(
        audio,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
        padding=False,
    )
    input_values = inputs["input_values"].to(_device)

    with torch.no_grad():
        logits = _model(input_values=input_values).logits

    probs      = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred_id    = int(probs.argmax())
    id2label   = _label_map["id2label"]
    pred_label = id2label[str(pred_id)]

    probabilities = {id2label[str(i)]: float(p) for i, p in enumerate(probs)}

    return {
        "predicted_cry_type": pred_label,
        "confidence": float(probs[pred_id]),
        "probabilities": probabilities,
    }
