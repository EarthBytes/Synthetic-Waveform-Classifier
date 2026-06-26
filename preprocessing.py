from __future__ import annotations

from numpy.typing import NDArray
import numpy as np
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE

def split_dataset(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.int64],
]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
