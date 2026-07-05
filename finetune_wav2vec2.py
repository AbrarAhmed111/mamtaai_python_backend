"""
Fine-tune facebook/wav2vec2-base for baby cry classification.

Target hardware : RTX 2050 — 4 GB VRAM
Strategy        : freeze CNN feature encoder, fp16, batch=4 + grad-accum=8

Run from mamtaai_python_backend/:
    python finetune_wav2vec2.py

Outputs:
    models/wav2vec2_cry_classifier/   — HuggingFace model + tokenizer
    models/wav2vec2_cry_classifier/label_map.json
"""

import json
import os
import random
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    get_linear_schedule_with_warmup,
)

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_DIR   = Path("datasets/merged_augmented")
OUTPUT_DIR    = Path("models/wav2vec2_cry_classifier")
BASE_MODEL    = "facebook/wav2vec2-base"

SAMPLE_RATE   = 16_000
MAX_SECONDS   = 8
MAX_SAMPLES   = MAX_SECONDS * SAMPLE_RATE   # 128 000

TRAIN_SPLIT   = 0.70
VAL_SPLIT     = 0.15
# test = remaining 0.15

BATCH_SIZE    = 4
GRAD_ACCUM    = 8       # effective batch = 32 (VRAM-limited, not RAM-limited)
EPOCHS        = 20
LR            = 1e-4
WARMUP_RATIO  = 0.10
PATIENCE      = 4       # early-stop after N val epochs without improvement
SEED          = 42
NUM_WORKERS   = 6       # 32 GB RAM — use more CPU workers for data loading
CACHE_IN_RAM  = True    # pre-load all 22k files into RAM after first epoch

USE_FP16      = torch.cuda.is_available()
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Dataset ───────────────────────────────────────────────────────────────────

def load_file_list(dataset_dir: Path) -> tuple[list[str], list[int], dict, dict]:
    label_dirs = sorted([d for d in dataset_dir.iterdir() if d.is_dir()])
    label2id   = {d.name: i for i, d in enumerate(label_dirs)}
    id2label   = {i: d.name for i, d in enumerate(label_dirs)}

    files, labels = [], []
    for d in label_dirs:
        for f in d.glob("*.wav"):
            files.append(str(f))
            labels.append(label2id[d.name])

    combined = list(zip(files, labels))
    random.Random(SEED).shuffle(combined)
    files, labels = zip(*combined)
    return list(files), list(labels), label2id, id2label


class CryDataset(Dataset):
    def __init__(self, files: list[str], labels: list[int], feature_extractor,
                 cache: dict | None = None):
        self.files     = files
        self.labels    = labels
        self.extractor = feature_extractor
        # Shared RAM cache: path -> preprocessed float32 array (filled lazily)
        self.cache = cache if cache is not None else {}

    def __len__(self):
        return len(self.files)

    def _load_audio(self, path: str) -> np.ndarray:
        if path in self.cache:
            return self.cache[path]
        audio, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True)
        if len(audio) > MAX_SAMPLES:
            audio = audio[:MAX_SAMPLES]
        elif len(audio) < MAX_SAMPLES:
            audio = np.pad(audio, (0, MAX_SAMPLES - len(audio)))
        if CACHE_IN_RAM:
            self.cache[path] = audio
        return audio

    def __getitem__(self, idx):
        audio = self._load_audio(self.files[idx])
        inputs = self.extractor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=False,
        )
        return {
            "input_values": inputs["input_values"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def collate_fn(batch):
    input_values = torch.stack([b["input_values"] for b in batch])
    labels       = torch.stack([b["label"]       for b in batch])
    return {"input_values": input_values, "labels": labels}


# ── Training loop ─────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_values = batch["input_values"].to(DEVICE)
            labels       = batch["labels"].to(DEVICE)
            with torch.cuda.amp.autocast(enabled=USE_FP16):
                out = model(input_values=input_values)
            loss = criterion(out.logits, labels)
            total_loss += loss.item() * len(labels)
            correct    += (out.logits.argmax(-1) == labels).sum().item()
            n          += len(labels)
    return total_loss / n, correct / n


def train():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device : {DEVICE}")
    print(f"FP16   : {USE_FP16}\n")

    # 1. Load file list
    print(f"Scanning {DATASET_DIR} …")
    files, labels, label2id, id2label = load_file_list(DATASET_DIR)
    n = len(files)
    print(f"  {n} files across {len(label2id)} classes: {list(label2id.keys())}\n")

    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)
    train_files,  train_labels  = files[:n_train],          labels[:n_train]
    val_files,    val_labels    = files[n_train:n_train+n_val], labels[n_train:n_train+n_val]
    test_files,   test_labels   = files[n_train+n_val:],    labels[n_train+n_val:]
    print(f"Split  : {len(train_files)} train / {len(val_files)} val / {len(test_files)} test\n")

    # 2. Feature extractor
    print(f"Loading feature extractor from {BASE_MODEL} …")
    extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL)

    # 3. Datasets & loaders
    # Single shared cache so all splits populate the same RAM store
    ram_cache = {} if CACHE_IN_RAM else None
    print(f"RAM cache : {'enabled (22k files ~4–6 GB RAM)' if CACHE_IN_RAM else 'disabled'}")

    train_ds = CryDataset(train_files, train_labels, extractor, cache=ram_cache)
    val_ds   = CryDataset(val_files,   val_labels,   extractor, cache=ram_cache)
    test_ds  = CryDataset(test_files,  test_labels,  extractor, cache=ram_cache)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=NUM_WORKERS,
                              pin_memory=True, persistent_workers=True)

    # 4. Model
    print(f"Loading model {BASE_MODEL} …")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(label2id),
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    # Freeze CNN feature encoder — saves ~1 GB VRAM, speeds up training
    model.freeze_feature_encoder()
    model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,}\n")

    # 5. Optimizer, scheduler, scaler
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=LR, weight_decay=0.01,
    )
    total_steps   = (len(train_loader) // GRAD_ACCUM) * EPOCHS
    warmup_steps  = int(total_steps * WARMUP_RATIO)
    scheduler     = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler        = torch.cuda.amp.GradScaler(enabled=USE_FP16)
    criterion     = nn.CrossEntropyLoss()

    # 6. Training
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc  = 0.0
    patience_cnt  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss, running_correct, running_n = 0.0, 0, 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader, 1):
            input_values = batch["input_values"].to(DEVICE)
            labels_t     = batch["labels"].to(DEVICE)

            with torch.cuda.amp.autocast(enabled=USE_FP16):
                out  = model(input_values=input_values)
                loss = criterion(out.logits, labels_t) / GRAD_ACCUM

            scaler.scale(loss).backward()

            running_loss    += loss.item() * GRAD_ACCUM * len(labels_t)
            running_correct += (out.logits.argmax(-1) == labels_t).sum().item()
            running_n       += len(labels_t)

            if step % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

        train_loss = running_loss / running_n
        train_acc  = running_correct / running_n

        val_loss, val_acc = evaluate(model, val_loader, criterion)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}  "
            f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_cnt = 0
            model.save_pretrained(OUTPUT_DIR)
            extractor.save_pretrained(OUTPUT_DIR)
            print(f"  ✓ New best val_acc={best_val_acc:.4f} — checkpoint saved")
        else:
            patience_cnt += 1
            print(f"  No improvement ({patience_cnt}/{PATIENCE})")
            if patience_cnt >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    # 7. Test evaluation with best checkpoint
    print(f"\nLoading best checkpoint for test evaluation …")
    best_model = Wav2Vec2ForSequenceClassification.from_pretrained(OUTPUT_DIR)
    best_model.to(DEVICE)
    test_loss, test_acc = evaluate(best_model, test_loader, criterion)
    print(f"\n{'='*60}")
    print(f"  Best val  accuracy : {best_val_acc:.4f}")
    print(f"  Test      accuracy : {test_acc:.4f}")
    print(f"{'='*60}")

    # 8. Save label map
    label_map = {"label2id": label2id, "id2label": {str(k): v for k, v in id2label.items()}}
    with open(OUTPUT_DIR / "label_map.json", "w") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nModel saved to: {OUTPUT_DIR}")
    print("Run the API server — it will auto-detect and use the wav2vec2 model.")


if __name__ == "__main__":
    train()
