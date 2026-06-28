from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from config import CLASS_LABELS, SAMPLE_RATE, VISUALISATION_MAX_ERRORS, WAVEFORM_LENGTH

def extract_waveforms(X: NDArray[np.float64]) -> NDArray[np.float64]:
    return X[:, :WAVEFORM_LENGTH]

def model_output_path(output_dir: Path, output_slug: str) -> Path:
    return output_dir / f"{output_slug}_predictions.png"

def _time_axis() -> NDArray[np.float64]:
    return np.linspace(0, WAVEFORM_LENGTH / SAMPLE_RATE, WAVEFORM_LENGTH, endpoint=False)

def _pick_from_pool(
    pool: NDArray[np.int64],
    count: int,
    chosen: list[int],
    rng: np.random.Generator,
) -> list[int]:
    available = pool[~np.isin(pool, chosen)]
    if len(available) == 0 or count <= 0:
        return []
    take = min(count, len(available))
    picks = rng.choice(available, size=take, replace=False)
    return picks.tolist()

def _pick_balanced_by_class(
    indices: NDArray[np.int64],
    y_true: NDArray[np.int64],
    count: int,
    chosen: list[int],
    rng: np.random.Generator,
) -> list[int]:
    if count <= 0 or len(indices) == 0:
        return []

    per_class = count // len(CLASS_LABELS)
    remainder = count % len(CLASS_LABELS)
    picks: list[int] = []

    for label_idx, label in enumerate(CLASS_LABELS):
        label_count = per_class + (1 if label_idx < remainder else 0)
        label_pool = indices[y_true[indices] == label]
        picks.extend(_pick_from_pool(label_pool, label_count, chosen + picks, rng))

    if len(picks) < count:
        extra = _pick_from_pool(indices, count - len(picks), chosen + picks, rng)
        picks.extend(extra)

    return picks[:count]

def select_example_indices(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    n_examples: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    misclassified = np.where(y_true != y_pred)[0]
    correct = np.where(y_true == y_pred)[0]
    rng.shuffle(misclassified)

    if len(misclassified) == 0:
        chosen = _pick_balanced_by_class(correct, y_true, n_examples, [], rng)
        return np.array(chosen[:n_examples], dtype=np.int64)

    n_wrong = min(len(misclassified), VISUALISATION_MAX_ERRORS)
    n_right = n_examples - n_wrong

    chosen = misclassified[:n_wrong].tolist()
    chosen.extend(_pick_balanced_by_class(correct, y_true, n_right, chosen, rng))

    return np.array(chosen[:n_examples], dtype=np.int64)

def select_comparison_indices(
    y_true: NDArray[np.int64],
    predictions: dict[str, NDArray[np.int64]],
    n_examples: int,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    any_wrong = np.zeros(len(y_true), dtype=bool)
    for y_pred in predictions.values():
        any_wrong |= y_true != y_pred

    per_class = n_examples // len(CLASS_LABELS)
    chosen: list[int] = []

    for label in CLASS_LABELS:
        class_indices = np.where(y_true == label)[0]
        rng.shuffle(class_indices)

        interesting = class_indices[any_wrong[class_indices]]
        other = class_indices[~any_wrong[class_indices]]
        rng.shuffle(interesting)
        rng.shuffle(other)

        class_picks: list[int] = []
        if len(interesting) > 0:
            class_picks.append(int(interesting[0]))

        remaining = per_class - len(class_picks)
        if remaining > 0:
            class_picks.extend(_pick_from_pool(other, remaining, class_picks, rng))

        if len(class_picks) < per_class:
            pool = np.setdiff1d(class_indices, class_picks, assume_unique=False)
            class_picks.extend(_pick_from_pool(pool, per_class - len(class_picks), class_picks, rng))

        chosen.extend(class_picks[:per_class])

    return np.array(chosen[:n_examples], dtype=np.int64)

def plot_waveform_predictions(
    waveforms: NDArray[np.float64],
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    model_name: str,
    output_path: Path,
    indices: NDArray[np.int64],
) -> None:
    n_examples = len(indices)
    n_cols = min(3, n_examples)
    n_rows = (n_examples + n_cols - 1) // n_cols
    time = _time_axis()

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2.5 * n_rows), squeeze=False)
    fig.suptitle(f"Test waveforms — {model_name}", fontsize=14)

    for plot_idx, sample_idx in enumerate(indices):
        row, col = divmod(plot_idx, n_cols)
        ax = axes[row][col]
        correct = y_true[sample_idx] == y_pred[sample_idx]
        true_label = CLASS_LABELS[int(y_true[sample_idx])]
        pred_label = CLASS_LABELS[int(y_pred[sample_idx])]
        status = "correct" if correct else "misclassified"
        colour = "#2a9d8f" if correct else "#e76f51"

        ax.plot(time, waveforms[sample_idx], color="#264653", linewidth=0.9)
        ax.set_title(
            f"True: {true_label}\nPredicted: {pred_label} ({status})",
            fontsize=9,
            color=colour,
        )
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)

    for plot_idx in range(n_examples, n_rows * n_cols):
        row, col = divmod(plot_idx, n_cols)
        axes[row][col].axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_model_comparison(
    waveforms: NDArray[np.float64],
    y_true: NDArray[np.int64],
    predictions: dict[str, NDArray[np.int64]],
    model_names: list[str],
    output_path: Path,
    indices: NDArray[np.int64],
) -> None:
    n_examples = len(indices)
    time = _time_axis()

    fig, axes = plt.subplots(
        n_examples,
        1,
        figsize=(11, 2.6 * n_examples),
        squeeze=False,
    )
    fig.suptitle("Model comparison on test waveforms", fontsize=14)

    for row, sample_idx in enumerate(indices):
        ax = axes[row][0]
        ax.plot(time, waveforms[sample_idx], color="#264653", linewidth=0.9)
        true_label = CLASS_LABELS[int(y_true[sample_idx])]

        lines = [f"True label: {true_label}"]
        for name in model_names:
            pred = int(predictions[name][sample_idx])
            pred_label = CLASS_LABELS[pred]
            status = "correct" if pred == y_true[sample_idx] else "misclassified"
            lines.append(f"{name}: {pred_label} ({status})")

        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        if row == n_examples - 1:
            ax.set_xlabel("Time (s)")
        else:
            ax.set_xticklabels([])

        ax.text(
            1.02,
            0.5,
            "\n".join(lines),
            transform=ax.transAxes,
            va="center",
            ha="left",
            fontsize=9,
            family="monospace",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(right=0.72)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
