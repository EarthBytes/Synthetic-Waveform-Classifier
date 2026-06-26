from __future__ import annotations
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

from config import CLASS_LABELS, FEATURE_NAMES, TEST_SIZE, RANDOM_STATE
from models import create_logistic_regression, evaluate_classifier
from preprocessing import split_dataset
from generator import generate_dataset

def _print_class_counts(y: np.ndarray, indent: str = "  ") -> None:
    for label_id, name in CLASS_LABELS.items():
        print(f"{indent}Class {label_id} ({name}): {(y == label_id).sum()} samples")

def main() -> None:
    rng = np.random.default_rng(RANDOM_STATE)
    X, y = generate_dataset(rng=rng)

    n_engineered = len(FEATURE_NAMES)
    n_raw = X.shape[1] - n_engineered

    print(f"Generated {len(y)} seismic waveforms")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Raw samples: {n_raw} columns")
    print(f"  Engineered: {n_engineered} features")
    _print_class_counts(y)
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")

    X_train, X_test, y_train, y_test = split_dataset(X, y)

    train_pct = (1.0 - TEST_SIZE) * 100
    test_pct = TEST_SIZE * 100
    print()
    print(f"Train/test split ({train_pct:.0f}% / {test_pct:.0f}%)")
    print(f"  Training:   {len(y_train)} samples  X shape {X_train.shape}")
    _print_class_counts(y_train, indent="    ")
    print(f"  Testing:    {len(y_test)} samples  X shape {X_test.shape}")
    _print_class_counts(y_test, indent="    ")

    print()
    print("Training logistic regression")
    model = create_logistic_regression()
    model.fit(X_train, y_train)
    accuracy, y_pred = evaluate_classifier(model, X_test, y_test)
    print("  Predictions sample:", y_pred[:10])
    print(f"  Test accuracy: {accuracy:.1%}")

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS.values()))

if __name__ == "__main__":
    main()