import numpy as np

# Waveform parameters
N_SAMPLES = 100
WAVEFORM_LENGTH = 200
SAMPLE_RATE = 100  # Hz 

# Class A (volcanic tremors) with higher frequency
CLASS_A_FREQUENCY = 10.0  # Hz

# Class B (fault slip events) with lower frequency
CLASS_B_FREQUENCY = 2.0  # Hz

AMPLITUDE = 1.0
NOISE_STD = 0.15 # Gaussian noise standard deviation

CLASS_LABELS = {0: "volcanic_tremor", 1: "fault_slip"}

def generate_waveform(
    frequency: float,
    length: int = WAVEFORM_LENGTH,
    sample_rate: float = SAMPLE_RATE,
    amplitude: float = AMPLITUDE,
    noise_std: float = NOISE_STD,
) -> np.ndarray:
    # Generate a single sine-wave seismic signal with Gaussian noise
    t = np.linspace(0, length / sample_rate, length, endpoint=False)
    signal = amplitude * np.sin(2 * np.pi * frequency * t)
    noise = np.random.normal(0, noise_std, length)
    return signal + noise

def generate_dataset(n_samples: int = N_SAMPLES) -> tuple[np.ndarray, np.ndarray]:
    # Generate synthetic seismic waveforms for two geological event classes.

    # X : ndarray of shape (n_samples, WAVEFORM_LENGTH) 
    # Feature vectors with one row per waveform.
    
    # y : ndarray of shape (n_samples,)
    # Integer labels of 0 = volcanic tremor and 1 = fault slip.

    n_per_class = n_samples // 2
    X = []
    y = []

    for _ in range(n_per_class):
        X.append(generate_waveform(CLASS_A_FREQUENCY))
        y.append(0)

    for _ in range(n_per_class):
        X.append(generate_waveform(CLASS_B_FREQUENCY))
        y.append(1)

    return np.array(X), np.array(y)

def main() -> None:
    np.random.seed(42)
    X, y = generate_dataset()

    print(f"Generated {len(y)} seismic waveforms")
    print(f"  X shape: {X.shape}  (samples × time points)")
    print(f"  y shape: {y.shape}")
    print(f"  Class 0 ({CLASS_LABELS[0]}): {(y == 0).sum()} samples")
    print(f"  Class 1 ({CLASS_LABELS[1]}): {(y == 1).sum()} samples")
    print(f"  Feature vector length: {X.shape[1]}")
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")

if __name__ == "__main__":
    main()