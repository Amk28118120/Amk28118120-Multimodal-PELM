import numpy as np
RANDOM_SEED = 42

# =============================================================================
# DATASET SELECTION
# =============================================================================
DATASET_TYPE = "fsdd" # "mnist" | "fsdd" | "abalone" | "mushroom" 

# Dictionary mapping datasets to their checkpoint intervals
CKPT_INT_MAP = {
    "fsdd": 100,
    "mnist": 1000,
    "abalone": 230,
    "mushroom": 4, 
}
if DATASET_TYPE not in CKPT_INT_MAP:
    raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")

CKPT_INT = CKPT_INT_MAP[DATASET_TYPE]


# =============================================================================
# DATASET SIZE - EDIT HERE FOR TESTING vs FULL EXPERIMENT
# =============================================================================

# # abalone has 4177 total, 80/20 split:
# N_TRAIN = 3480
# N_TEST  = 696

Full MNIST paper replication (4-5 hours):
N_TRAIN = 60000
N_TEST = 10000

# # #FSDD full dataset (all 3000 samples, ~1800 train / 200 test):
# N_TRAIN = 2700
# N_TEST  = 300

# # # mushroom full dataset (all 8124 samples, ~6500 train / 1624 test):
# N_TRAIN = 4124
# N_TEST  = 1520

# Dataset sizes (for validation)
DATASET_SIZES = {
    "mnist": 70000,   # MNIST: 60k train + 10k test available
    "fsdd": 2700,     # FSDD: 6 speakers × 10 digits × 50 = 3000
    "abalone": 4177,  # Abalone: 4177 samples total
    "mushroom": 8124, # Mushroom: 8124 samples total
}
TASK_TYPE = "regression" if DATASET_TYPE == "abalone" else "classification"

MNIST_INPUT_SIZE = 28  


M_FEATURES = 4096

ENCODING_METHOD = "fourier_embedding" # Options: "noise_embedding" or "fourier_embedding" 

NOISE_CORR_LENGTH = 5  # Noise correlation length in pixels

FOURIER_FREQUENCIES = 2000  # Number of Fourier frequencies

PHASE_RANGE_INPUT = np.pi      # Input data encoded in [0, π]
PHASE_RANGE_EMBEDDING = np.pi  # Embedding matrix in [0, π]

LAMBDA_REG = 0.001833 # Regularization parameter 

Zoom_Factor = 18  # mnist

# SLM specifications
SLM_WIDTH = 1920
SLM_HEIGHT = 1200
SLM_PIXEL_PITCH = 8e-6  # meters
SLM_PHASE_LEVELS = 210

# Camera specifications
CAM_WIDTH = 1280
CAM_HEIGHT = 1024
CAM_BIT_DEPTH = 8

CAM_BIN_SIZE = 10  # Size of spatial bins for feature extraction (pixels)
CAM_EXPOSURE_US = 10000  # us

# Optical setup
LASER_WAVELENGTH = 532e-9    # meters (green laser)
LENS_FOCAL_LENGTH = 150e-3   # meters (150mm focal length)

# Timing parameters
LC_SETTLING_TIME = 0.10       # seconds - liquid crystal settling time
NUM_FRAMES_TO_AVERAGE = 3     # Number of frames to average (reduces vibration noise)

# Visualization settings
SAVE_EXAMPLE_IMAGES = True    # Save embedding, encoding, feature extraction examples
NUM_EXAMPLES_TO_SAVE = 3     # Number of example visualizations to save

# Configuration validation
if N_TRAIN / M_FEATURES < 0.5:
    print(f"WARNING: N/M ratio is {N_TRAIN/M_FEATURES:.2f}, which is very low.")
    print(f"Paper recommends N/M ≈ 20. Consider increasing N_TRAIN or decreasing M_FEATURES.")

Fold_ID = 1 
# =============================================================================
# AUDIO EXPERIMENT CONFIG
# =============================================================================
# ── Signal parameters ────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 8000
#AUDIO_DURATION = 1.0
AUDIO_N_MELS = 128     # 128 is the SOTA standard for speech/audio features
AUDIO_N_FFT = 256      # 32ms window is still perfect for phonemes
AUDIO_HOP_LENGTH = 64  # Halved to double the time resolution
AUDIO_FILL_H = 0.5  # fraction of SLM_HEIGHT used by the mel spectrogram
AUDIO_FILL_W = 0.5  # fraction of SLM_WIDTH  used by the mel spectrogram




