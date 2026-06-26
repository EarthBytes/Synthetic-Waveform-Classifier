from __future__ import annotations

from numpy.typing import NDArray
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import accuracy_score

def evaluate_classifier(
    model: ClassifierMixin,
    X_test: NDArray[np.float64],
    y_test: NDArray[np.int64],
) -> tuple[float, NDArray[np.int64]]:
    y_pred = model.predict(X_test)
    return float(accuracy_score(y_test, y_pred)), np.asarray(y_pred, dtype=np.int64)
