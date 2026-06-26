from __future__ import annotations
import numpy as np

from config import CLASS_LABELS, FEATURE_NAMES
from generator import generate_dataset

def main() -> None:
    rng = np.random.default_rng(42)
    X, y = generate_dataset(rng=rng)

    n_engineered = len(FEATURE_NAMES)
    n_raw = X.shape[1] - n_engineered

    print(f"Generated {len(y)} seismic waveforms")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Raw samples: {n_raw} columns")
    print(f"  Engineered: {n_engineered} features")
    print(f"  Class 0 ({CLASS_LABELS[0]}): {(y == 0).sum()} samples")
    print(f"  Class 1 ({CLASS_LABELS[1]}): {(y == 1).sum()} samples")
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")

if __name__ == "__main__":
    main()