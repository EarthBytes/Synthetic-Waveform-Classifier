from __future__ import annotations
import numpy as np
from numpy.typing import NDArray

# Waveform parameters
N_SAMPLES = 100
WAVEFORM_LENGTH = 200
SAMPLE_RATE = 100  # Hz 
NOISE_STD = 0.12

CLASS_LABELS = {0: "volcanic_tremor", 1: "fault_slip"}

# Volcanic tremor
VOLCANIC_FREQ_RANGE = (1.0, 8.0) 
VOLCANIC_N_COMPONENTS = (3, 6)
VOLCANIC_MOD_FREQ = 0.4 

# Fault slip
FAULT_N_BURSTS = (2, 5)
FAULT_BURST_FREQ_RANGE = (2.0, 25.0)  
FAULT_BURST_WIDTH = 0.04 

def _time_axis(length: int = WAVEFORM_LENGTH, sample_rate: float = SAMPLE_RATE) -> NDArray[np.float64]:
    return np.linspace(0, length / sample_rate, length, endpoint=False)

# Ricker wavelet
def _ricker(t: NDArray[np.float64], frequency: float) -> NDArray[np.float64]:
    p = (np.pi * frequency * t) ** 2
    return (1.0 - 2.0 * p) * np.exp(-p)

def _add_noise(
    signal: NDArray[np.float64],
    noise_std: float = NOISE_STD,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    rng = rng or np.random.default_rng()
    return signal + rng.normal(0.0, noise_std, signal.shape)

def generate_volcanic_tremor(
    length: int = WAVEFORM_LENGTH,
    sample_rate: float = SAMPLE_RATE,
    noise_std: float = NOISE_STD,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    rng = rng or np.random.default_rng()
    t = _time_axis(length, sample_rate)
    signal = np.zeros(length)

    n_components = rng.integers(VOLCANIC_N_COMPONENTS[0], VOLCANIC_N_COMPONENTS[1] + 1)
    for _ in range(n_components):
        freq = rng.uniform(*VOLCANIC_FREQ_RANGE)
        phase = rng.uniform(0.0, 2.0 * np.pi)
        amplitude = rng.uniform(0.4, 1.0)
        signal += amplitude * np.sin(2.0 * np.pi * freq * t + phase)

    signal /= n_components

    mod_depth = rng.uniform(0.15, 0.35)
    mod_phase = rng.uniform(0.0, 2.0 * np.pi)
    envelope = 1.0 + mod_depth * np.sin(2.0 * np.pi * VOLCANIC_MOD_FREQ * t + mod_phase)
    signal *= envelope

    return _add_noise(signal, noise_std, rng)


def generate_fault_slip(
    length: int = WAVEFORM_LENGTH,
    sample_rate: float = SAMPLE_RATE,
    noise_std: float = NOISE_STD,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    rng = rng or np.random.default_rng()
    t = _time_axis(length, sample_rate)
    duration = length / sample_rate
    signal = np.zeros(length)

    n_bursts = rng.integers(FAULT_N_BURSTS[0], FAULT_N_BURSTS[1] + 1)
    min_gap = FAULT_BURST_WIDTH * 2.0
    burst_centers: list[float] = []

    for _ in range(n_bursts):
        if not burst_centers:
            center = rng.uniform(FAULT_BURST_WIDTH, duration - FAULT_BURST_WIDTH)
        else:
            earliest = burst_centers[-1] + min_gap
            if earliest >= duration - FAULT_BURST_WIDTH:
                break
            center = rng.uniform(earliest, duration - FAULT_BURST_WIDTH)
        burst_centers.append(center)

    for center in burst_centers:
        freq = rng.uniform(*FAULT_BURST_FREQ_RANGE)
        amplitude = rng.uniform(0.8, 1.6)
        burst_t = t - center
        burst = _ricker(burst_t, freq)
        peak = np.max(np.abs(burst))
        if peak > 0:
            burst /= peak
        signal += amplitude * burst

    return _add_noise(signal, noise_std, rng)

def _dominant_frequency(
    magnitudes: NDArray[np.float64],
    freqs: NDArray[np.float64],
) -> float:
    positive = magnitudes[1 : len(magnitudes) // 2]
    positive_freqs = freqs[1 : len(freqs) // 2]
    if positive.sum() == 0:
        return 0.0
    return float(positive_freqs[np.argmax(positive)])

def _spectral_spread(
    magnitudes: NDArray[np.float64],
    freqs: NDArray[np.float64],
    centroid: float,
) -> float:
    positive = magnitudes[1 : len(magnitudes) // 2]
    positive_freqs = freqs[1 : len(freqs) // 2]
    total = positive.sum()
    if total == 0:
        return 0.0
    return float(np.sqrt(np.sum(((positive_freqs - centroid) ** 2) * positive) / total))

def _short_time_energy_features(
    waveform: NDArray[np.float64],
    sample_rate: float,
    frame_ms: float = 20.0,
) -> tuple[float, float, float]:
    frame_len = max(4, int(sample_rate * frame_ms / 1000.0))
    n_frames = max(1, (len(waveform) - frame_len) // frame_len + 1)
    energies = np.array(
        [np.sum(waveform[i : i + frame_len] ** 2) for i in range(0, n_frames * frame_len, frame_len)],
        dtype=np.float64,
    )
    mean_energy = float(energies.mean())
    max_energy = float(energies.max())
    peak_ratio = max_energy / (mean_energy + 1e-12)
    return max_energy, mean_energy, peak_ratio

def extract_features(
    waveform: NDArray[np.float64],
    sample_rate: float = SAMPLE_RATE,
) -> NDArray[np.float64]:
    magnitudes = np.abs(np.fft.rfft(waveform))
    freqs = np.fft.rfftfreq(len(waveform), d=1.0 / sample_rate)
    total_mag = magnitudes.sum() + 1e-12

    centroid = float(np.sum(freqs * magnitudes) / total_mag)
    bandwidth = _spectral_spread(magnitudes, freqs, centroid)
    dominant_freq = _dominant_frequency(magnitudes, freqs)
    spectral_entropy = float(
        -np.sum((magnitudes / total_mag) * np.log(magnitudes / total_mag + 1e-12))
    )

    std = waveform.std()
    if std < 1e-12:
        kurtosis = 0.0
    else:
        kurtosis = float(np.mean(((waveform - waveform.mean()) / std) ** 4) - 3.0)

    max_energy, mean_energy, peak_ratio = _short_time_energy_features(waveform, sample_rate)

    return np.array(
        [
            dominant_freq,
            centroid,
            bandwidth,
            spectral_entropy,
            kurtosis,
            max_energy,
            mean_energy,
            peak_ratio,
        ],
        dtype=np.float64,
    )

FEATURE_NAMES = [
    "dominant_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "kurtosis",
    "max_short_time_energy",
    "mean_short_time_energy",
    "energy_peak_ratio",
]

def generate_dataset(
    n_samples: int = N_SAMPLES,
    include_raw: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    rng = rng or np.random.default_rng()
    
    n_per_class = n_samples // 2
    pairs: list[tuple[NDArray[np.float64], int]] = []
    for generator, label in ((generate_volcanic_tremor, 0), (generate_fault_slip, 1)):
        for _ in range(n_per_class):
            pairs.append((generator(rng=rng), label))

    indices = rng.permutation(n_samples)
    pairs = [pairs[i] for i in indices]
    waveforms = [p[0] for p in pairs]
    labels = [p[1] for p in pairs]

    engineered = np.vstack([extract_features(w) for w in waveforms])
    if include_raw:
        raw = np.vstack(waveforms)
        X = np.hstack([raw, engineered])
    else:
        X = engineered

    return X, np.array(labels, dtype=np.int64)

def main() -> None:
    rng = np.random.default_rng(42)
    X, y = generate_dataset(rng=rng)

    n_engineered = len(FEATURE_NAMES)
    n_raw = WAVEFORM_LENGTH if X.shape[1] > n_engineered else 0

    print(f"Generated {len(y)} seismic waveforms")
    print(f"  X shape: {X.shape}")
    print(f"  y shape: {y.shape}")
    print(f"  Class 0 ({CLASS_LABELS[0]}): {(y == 0).sum()} samples")
    print(f"  Class 1 ({CLASS_LABELS[1]}): {(y == 1).sum()} samples")
    print(f"  Value range: [{X.min():.3f}, {X.max():.3f}]")

if __name__ == "__main__":
    main()