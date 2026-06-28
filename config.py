# Train/test split
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Waveform parameters
N_PER_CLASS = 500
WAVEFORM_LENGTH = 200
SAMPLE_RATE = 100  # Hz
NOISE_STD_RANGE = (0.06, 0.20)

CLASS_LABELS = {
    0: "volcanic_tremor",
    1: "fault_slip",
}

# Volcanic tremor
VOLCANIC_FREQ_RANGE = (1.0, 8.0)
VOLCANIC_N_COMPONENTS = (3, 6)
VOLCANIC_MOD_FREQ = 0.4

# Fault slip
FAULT_N_BURSTS = (2, 5)
FAULT_BURST_FREQ_RANGE = (2.0, 25.0)
FAULT_BURST_WIDTH = 0.04

# KNN
KNN_N_NEIGHBORS = 5

# Random Forest
RF_N_ESTIMATORS = 100

FEATURE_NAMES = [
    "dominant_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_entropy",
    "excess_kurtosis",
    "max_short_time_energy",
    "mean_short_time_energy",
    "energy_peak_ratio",
]