from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from sklearn.base import ClassifierMixin
from models.knn import create_knn
from models.logistic_regression import create_logistic_regression
from models.random_forest import create_random_forest
from models.mlp import create_mlp

ModelFactory = Callable[[], ClassifierMixin]

@dataclass(frozen=True)
class ModelEntry:
    name: str
    cli_flag: str
    output_slug: str
    factory: ModelFactory

MODEL_ENTRIES: tuple[ModelEntry, ...] = (
    ModelEntry(
        "logistic regression",
        "logistic-regression",
        "logistic_regression",
        create_logistic_regression,
    ),
    ModelEntry("KNN", "knn", "knn", create_knn),
    ModelEntry("random forest", "random-forest", "random_forest", create_random_forest),
    ModelEntry("MLP", "mlp", "mlp", create_mlp),
)

MODELS: list[tuple[str, ModelFactory]] = [
    (entry.name, entry.factory) for entry in MODEL_ENTRIES
]

CLI_MODELS: dict[str, ModelEntry] = {
    entry.cli_flag: entry for entry in MODEL_ENTRIES
}
