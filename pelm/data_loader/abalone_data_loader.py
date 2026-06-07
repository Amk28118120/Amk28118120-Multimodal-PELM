"""
Abalone data loader for hardware-based PELM experiments.
Converts 8 tabular features → uint8 28×28 image (same format as MNIST)
so the entire hardware pipeline (encode_input, SLM zoom) is unchanged.
"""
import numpy as np
from config import RANDOM_SEED, N_TRAIN, N_TEST

# These globals are populated by get_abalone() and imported by main.py
# to compute RMSD against the true float ring counts.
ABALONE_Y_FLOAT_TRAIN = None
ABALONE_Y_FLOAT_TEST  = None

# Column names for reference:
# Sex(0) Length(1) Diameter(2) Height(3)
# WholeWeight(4) ShuckedWeight(5) VisceraWeight(6) ShellWeight(7) Rings(8)
ABALONE_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
)

def _load_raw():
    """Download (or read cached) abalone CSV and return X float64, y int arrays."""
    import os, urllib.request
    cache = "./data/abalone.data"
    os.makedirs("./data", exist_ok=True)
    if not os.path.exists(cache):
        print("[Data] Downloading abalone dataset...")
        urllib.request.urlretrieve(ABALONE_URL, cache)
        print("[Data] Download complete.")

    rows = []
    with open(cache, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            sex = parts[0].strip().upper()
            # Ordinal encode Sex: F->0, I->1, M->2
            sex_enc = {"F": 0, "I": 1, "M": 2}[sex]
            feats = [float(sex_enc)] + [float(p) for p in parts[1:8]]
            rings = int(parts[8])
            rows.append(feats + [rings])

    arr  = np.array(rows, dtype=np.float64)
    X    = arr[:, :8]   # shape (4177, 8)
    y    = arr[:,  8].astype(np.int32)  # ring counts 1–29
    return X, y


def _minmax_scale_train_test(X_train, X_test):
    """
    Fit min-max scaler on train split only → scale both splits to [0, 255].
    Test values outside the train range are clipped (not leaked).

    Returns
    -------
    X_train_sc, X_test_sc : uint8 arrays of shape (N, 8)
    """
    feat_min = X_train.min(axis=0)          # shape (8,)
    feat_max = X_train.max(axis=0)          # shape (8,)
    rng = feat_max - feat_min
    rng[rng == 0] = 1.0                     # avoid division by zero (constant col)

    X_train_sc = np.clip((X_train - feat_min) / rng * 255.0, 0, 255).astype(np.uint8)
    X_test_sc  = np.clip((X_test  - feat_min) / rng * 255.0, 0, 255).astype(np.uint8)
    return X_train_sc, X_test_sc


def _tile_to_28x28(X_scaled):
    """
    Tile each 8-feature row into a uint8 (28, 28) image.

    Pattern: [f0,f1,f2,f3,f4,f5,f6,f7, f0,f1,...] repeated across 784 pixels.
    This fills every SLM pixel, avoiding dead zones.

    Returns
    -------
    images : uint8 ndarray  shape (N, 28, 28)
    """
    N = X_scaled.shape[0]
    L = 784  # 28 × 28

    # tile each row to length 784, then truncate, then reshape
    repeats = (L // 8) + 1                          # 99 full repeats → 792 values
    tiled   = np.tile(X_scaled, (1, repeats))[:, :L]  # (N, 784)
    return tiled.reshape(N, 28, 28).astype(np.uint8)


def get_abalone():
    """
    Load Abalone dataset and return train/test splits as uint8 28×28 images.

    Encoding
    --------
    - Sex column ordinal-encoded (F=0, I=1, M=2)
    - All 8 features min-max scaled to [0,255] (fit on train only)
    - Each sample tiled from shape (8,) → (28,28) uint8

    The dtype=uint8 + shape (28,28) makes encode_input() treat these
    exactly like MNIST images, including the 18× SLM zoom.

    Returns
    -------
    (X_train, y_train), (X_test, y_test)
        X : uint8 ndarray  shape (N, 28, 28)
        y : int32 ndarray  shape (N,)   ring counts (integer labels)

    Side effects
    ------------
    Sets module globals ABALONE_Y_FLOAT_TRAIN / ABALONE_Y_FLOAT_TEST
    (float ring values before int-cast) so main.py can compute RMSD.
    """
    global ABALONE_Y_FLOAT_TRAIN, ABALONE_Y_FLOAT_TEST

    print("[Data] Loading Abalone...")
    X, y = _load_raw()

    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(len(X))

    train_idx = idx[:N_TRAIN]
    test_idx  = idx[N_TRAIN: N_TRAIN + N_TEST]

    X_train_raw = X[train_idx]
    X_test_raw  = X[test_idx]
    y_train     = y[train_idx]
    y_test      = y[test_idx]

    # Store float ring values for RMSD computation in main.py
    ABALONE_Y_FLOAT_TRAIN = y_train.astype(np.float64)
    ABALONE_Y_FLOAT_TEST  = y_test.astype(np.float64)

    # Scale features and tile to 28×28
    X_train_sc, X_test_sc = _minmax_scale_train_test(X_train_raw, X_test_raw)
    X_train_img = _tile_to_28x28(X_train_sc)
    X_test_img  = _tile_to_28x28(X_test_sc)

    # Remap ring labels to 0-based class indices
    # rings range 1–29, subtract 1 → 0–28  (29 classes)
    # keep raw ring values
    y_train_out = y_train.astype(np.float64)
    y_test_out  = y_test.astype(np.float64)

    return (X_train_img, y_train_out), (X_test_img, y_test_out)

    n_classes = int(y_train_cls.max()) + 1
    print(f"[Data] Abalone: {N_TRAIN} train, {N_TEST} test | "
          f"Ring classes: {n_classes} (0–{n_classes-1})")

    return (X_train_img, y_train_cls), (X_test_img, y_test_cls)