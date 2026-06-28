from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

from config import KNN_N_NEIGHBORS
from models.pipelines import scaled_pipeline

def create_knn() -> Pipeline:
    return scaled_pipeline(KNeighborsClassifier(n_neighbors=KNN_N_NEIGHBORS))
