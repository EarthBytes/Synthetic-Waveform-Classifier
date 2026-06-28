from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from config import RANDOM_STATE
from models.pipelines import scaled_pipeline

def create_logistic_regression() -> Pipeline:
    return scaled_pipeline(
        LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
    )
