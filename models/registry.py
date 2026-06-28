from __future__ import annotations
from collections.abc import Callable
from sklearn.pipeline import Pipeline
from models.knn import create_knn
from models.logistic_regression import create_logistic_regression

ModelFactory = Callable[[], Pipeline]

MODELS: list[tuple[str, ModelFactory]] = [
    ("logistic regression", create_logistic_regression),
    ("KNN", create_knn),
]
