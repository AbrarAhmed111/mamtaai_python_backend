"""
Inference wrapper for the fine-tuned wav2vec2 cry classifier.
Loaded automatically by classification.py when models/wav2vec2_cry_classifier/ exists.
"""
import json
from pathlib import Path

import librosa
import numpy as np
import torch

_MODEL_DIR = Path(__file__).parent.parent / "models" / "wav2vec2_cry_classifier"

SAMPLE_RATE = 16_000
MAX_SAMPLES = 8 * SAMPLE_RATE   # 128 000 — 8 seconds

_model     = None
_extractor = None
_label_map = None
_device    = None


def is_available() -> bool:
    return (_MODEL_DIR / "config.json").exists() and (_MODEL_DIR / "label_map.json").exists()


def _load():
    global _model, _extractor, _label_map, _device

    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _extractor = AutoFeatureExtractor.from_pretrained(str(_MODEL_DIR))
    _model     = Wav2Vec2ForSequenceClassification.from_pretrained(str(_MODEL_DIR))
    _model.eval()
    _model.to(_device)

    with open(_MODEL_DIR / "label_map.json") as f:
        _label_map = json.load(f)

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
    global _model, _extractor, _label_map, _device

    if _model is None:
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
