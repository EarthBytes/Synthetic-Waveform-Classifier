from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from config import CLASS_LABELS, COMPARISON_OUTPUT, FEATURE_NAMES, N_VISUALISATION_EXAMPLES, RANDOM_STATE, TEST_SIZE, VISUALISATION_OUTPUT_DIR,

from generator import generate_dataset
from models import evaluate_classifier
from models.registry import CLI_MODELS, MODEL_ENTRIES, ModelEntry
from preprocessing import split_dataset
from sklearn.base import ClassifierMixin
from visualisation import extract_waveforms, model_output_path, plot_model_comparison, plot_waveform_predictions, select_comparison_indices, select_example_indices,

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train seismic waveform classifiers and generate visualisations.",
    )
    parser.add_argument(
        "-knn",
        action="store_true",
        help="Train and visualise KNN only.",
    )
    parser.add_argument(
        "-mlp",
        action="store_true",
        help="Train and visualise MLP only.",
    )
    parser.add_argument(
        "-random-forest",
        dest="random_forest",
        action="store_true",
        help="Train and visualise random forest only.",
    )
    parser.add_argument(
        "-logistic-regression",
        dest="logistic_regression",
        action="store_true",
        help="Train and visualise logistic regression only.",
    )
    args = parser.parse_args()

    selected_flags = [
        flag
        for flag, enabled in (
            ("knn", args.knn),
            ("mlp", args.mlp),
            ("random-forest", args.random_forest),
            ("logistic-regression", args.logistic_regression),
        )
        if enabled
    ]
    if len(selected_flags) > 1:
        parser.error("Choose at most one model flag.")

    return args

def _selected_models(args: argparse.Namespace) -> list[ModelEntry]:
    if args.knn:
        return [CLI_MODELS["knn"]]
    if args.mlp:
        return [CLI_MODELS["mlp"]]
    if args.random_forest:
        return [CLI_MODELS["random-forest"]]
    if args.logistic_regression:
        return [CLI_MODELS["logistic-regression"]]
    return list(MODEL_ENTRIES)

def _print_class_counts(y: np.ndarray, indent: str = "  ") -> None:
    for label_id, name in CLASS_LABELS.items():
        print(f"{indent}Class {label_id} ({name}): {(y == label_id).sum()} samples")

def _train_and_evaluate(
    name: str,
    model: ClassifierMixin,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> np.ndarray:
    print()
    print(f"Training {name}")
    model.fit(X_train, y_train)
    accuracy, y_pred = evaluate_classifier(model, X_test, y_test)
    print("  Predictions sample:", y_pred[:10])
    print(f"  Test accuracy: {accuracy:.1%}")
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=CLASS_LABELS.values()))
    return y_pred

def _save_model_visualisation(
    entry: ModelEntry,
    waveforms_test: np.ndarray,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    rng: np.random.Generator,
) -> Path:
    indices = select_example_indices(
        y_test,
        y_pred,
        N_VISUALISATION_EXAMPLES,
        rng,
    )
    output_path = model_output_path(output_dir, entry.output_slug)
    plot_waveform_predictions(
        waveforms_test,
        y_test,
        y_pred,
        entry.name,
        output_path,
        indices,
    )
    return output_path

def main() -> None:
    args = parse_args()
    selected_models = _selected_models(args)
    run_all_models = len(selected_models) == len(MODEL_ENTRIES)
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

    predictions: dict[str, np.ndarray] = {}
    for entry in selected_models:
        y_pred = _train_and_evaluate(
            entry.name,
            entry.factory(),
            X_train,
            y_train,
            X_test,
            y_test,
        )
        predictions[entry.name] = y_pred

    waveforms_test = extract_waveforms(X_test)
    output_dir = Path(VISUALISATION_OUTPUT_DIR)
    saved_paths: list[Path] = []

    print()
    print("Generating visualisations")
    for entry in selected_models:
        output_path = _save_model_visualisation(
            entry,
            waveforms_test,
            y_test,
            predictions[entry.name],
            output_dir,
            rng,
        )
        saved_paths.append(output_path)
        print(f"  Saved {entry.name} plot to {output_path}")

    if run_all_models:
        comparison_indices = select_comparison_indices(
            y_test,
            predictions,
            N_VISUALISATION_EXAMPLES,
            rng,
        )
        comparison_path = Path(COMPARISON_OUTPUT)
        plot_model_comparison(
            waveforms_test,
            y_test,
            predictions,
            [entry.name for entry in MODEL_ENTRIES],
            comparison_path,
            comparison_indices,
        )
        saved_paths.append(comparison_path)
        print(f"  Saved comparison plot to {comparison_path}")

    print()
    print(f"Done. {len(saved_paths)} plot(s) saved.")

if __name__ == "__main__":
    main()