from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

from config import RANDOM_STATE, RF_N_ESTIMATORS

def create_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
