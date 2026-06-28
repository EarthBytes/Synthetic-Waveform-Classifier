from __future__ import annotations
from collections.abc import Callable
from sklearn.base import ClassifierMixin
from models.knn import create_knn
from models.logistic_regression import create_logistic_regression
from models.random_forest import create_random_forest

ModelFactory = Callable[[], ClassifierMixin]

MODELS: list[tuple[str, ModelFactory]] = [
    ("logistic regression", create_logistic_regression),
    ("KNN", create_knn),
    ("random forest", create_random_forest),
]
