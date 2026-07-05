"""
Standalone training script — trains RF and GradientBoosting, saves the better one.
Run from mamtaai_python_backend/:
    python train_model.py
"""
import json
from collections import Counter
from pathlib import Path
from services.classification import BabyCryClassifier, set_model, DEFAULT_CRY_TYPES

DATASET_JSON = Path("datasets/training_dataset.json")
MODEL_NAME   = "baby_cry_classifier"


def _print_results(label: str, result: dict, counts: Counter):
    m = result["metrics"]
    print(f"\n{'='*60}")
    print(f"  {label} Results")
    print(f"{'='*60}")
    print(f"  Training samples:    {m['training_samples']}")
    print(f"  Validation samples:  {m['validation_samples']}")
    print(f"  Test samples:        {m['test_samples']}")
    print(f"  Feature dimensions:  {m.get('num_features', '?')}")
    print(f"\n  Validation accuracy: {m['validation_accuracy']:.4f}")
    print(f"  Test accuracy:       {m['test_accuracy']:.4f}")
    print(f"  Test F1 (weighted):  {m['test_f1']:.4f}")
    print(f"  Cross-val mean:      {m['cross_val_mean']:.4f} (+/- {m['cross_val_std']:.4f})")
    print("\n  Per-class report:")
    report = result["classification_report"]
    for label_name in sorted(counts.keys()):
        if label_name in report:
            r = report[label_name]
            print(f"    {label_name:20}  precision={r['precision']:.3f}  recall={r['recall']:.3f}  f1={r['f1-score']:.3f}")


def main():
    print("=" * 60)
    print("MamtaAI - Baby Cry Classifier Training")
    print("=" * 60)

    if not DATASET_JSON.exists():
        print(f"ERROR: Dataset not found at {DATASET_JSON}")
        return

    print(f"\nLoading dataset: {DATASET_JSON}")
    with open(DATASET_JSON, "r") as f:
        training_data = json.load(f)

    print(f"Loaded {len(training_data)} samples")
    counts = Counter(s["label"] for s in training_data)
    print("\nSamples per label:")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:20} {cnt}")
    print(f"\nCry types from config: {DEFAULT_CRY_TYPES}")

    results = {}

    for model_type in ("random_forest", "gradient_boosting", "xgboost"):
        print(f"\n{'='*60}")
        print(f"Training {model_type.replace('_', ' ').title()}...")
        print("=" * 60)
        clf = BabyCryClassifier(model_type=model_type, cry_types=DEFAULT_CRY_TYPES)
        res = clf.train(training_data=training_data, test_size=0.2, validation_size=0.1)
        results[model_type] = (clf, res)
        _print_results(model_type.replace("_", " ").title(), res, counts)

    # Pick best by test accuracy
    best_type = max(results, key=lambda k: results[k][1]["metrics"]["test_accuracy"])
    best_clf, best_res = results[best_type]

    print(f"\n{'='*60}")
    print(f"Winner: {best_type.replace('_', ' ').title()}  "
          f"(accuracy={best_res['metrics']['test_accuracy']:.4f})")
    print("=" * 60)

    # Confusion matrix for winner
    classes = best_clf.label_encoder.classes_
    print("\nConfusion matrix (rows=actual, cols=predicted):")
    print("  " + "  ".join(f"{c[:6]:>6}" for c in classes))
    for i, row in enumerate(best_res["confusion_matrix"]):
        print(f"  {classes[i][:6]:>6}  " + "  ".join(f"{v:>6}" for v in row))

    model_path = best_clf.save(MODEL_NAME)
    set_model(best_clf, model_path)
    print(f"\nModel saved to: {model_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
