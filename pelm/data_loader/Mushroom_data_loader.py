"""
Mushroom loader for hardware-based PELM experiments.
Converts 22 tabular features → uint8 28×28 image (same as abalone)
"""
import numpy as np
from config import RANDOM_SEED, N_TRAIN, N_TEST

MUSHROOM_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data"
)

def _load_raw():
    """Download (or read cached) mushroom dataset → X (N,22), y (N,)"""
    import os, urllib.request

    cache = "./data/mushroom.data"
    os.makedirs("./data", exist_ok=True)

    if not os.path.exists(cache):
        print("[Data] Downloading Mushroom dataset...")
        urllib.request.urlretrieve(MUSHROOM_URL, cache)
        print("[Data] Download complete.")

    rows = []
    with open(cache, "r") as f:
        for line in f:
            parts = line.strip().split(",")
            if not parts or "?" in parts:
                continue  # skip missing values (simple + safe)

            label = parts[0]          # 'e' or 'p'
            features = parts[1:]

            # encode categorical → integer (like abalone sex encoding)
            feats_enc = [ord(c) for c in features]  # simple stable encoding

            y_val = 1 if label == 'p' else 0

            rows.append(feats_enc + [y_val])

    arr = np.array(rows, dtype=np.float64)

    X = arr[:, :22]     # (N, 22)
    y = arr[:, 22].astype(np.int32)

    return X, y


def _minmax_scale_train_test(X_train, X_test):
    """Same as abalone"""
    feat_min = X_train.min(axis=0)
    feat_max = X_train.max(axis=0)

    rng = feat_max - feat_min
    rng[rng == 0] = 1.0

    X_train_sc = np.clip((X_train - feat_min) / rng * 255.0, 0, 255).astype(np.uint8)
    X_test_sc  = np.clip((X_test  - feat_min) / rng * 255.0, 0, 255).astype(np.uint8)

    return X_train_sc, X_test_sc


def _tile_to_28x28(X_scaled):
    """Tile 22 features → 784 pixels (same idea as abalone)"""
    N = X_scaled.shape[0]
    L = 784

    repeats = (L // 22) + 1
    tiled = np.tile(X_scaled, (1, repeats))[:, :L]

    return tiled.reshape(N, 28, 28).astype(np.uint8)


def get_mushroom():
    """Main loader — mirrors get_abalone() exactly"""

    print("[Data] Loading Mushroom...")

    X, y = _load_raw()

    np.random.seed(RANDOM_SEED)
    idx = np.random.permutation(len(X))

    train_idx = idx[:N_TRAIN]
    test_idx  = idx[N_TRAIN: N_TRAIN + N_TEST]

    X_train_raw = X[train_idx]
    X_test_raw  = X[test_idx]
    y_train     = y[train_idx]
    y_test      = y[test_idx]

    # scale + tile (same as abalone)
    X_train_sc, X_test_sc = _minmax_scale_train_test(X_train_raw, X_test_raw)

    X_train_img = _tile_to_28x28(X_train_sc)
    X_test_img  = _tile_to_28x28(X_test_sc)

    # classification labels (NO regression here)
    y_train_out = y_train.astype(np.int32)
    y_test_out  = y_test.astype(np.int32)

    print(f"[Data] Mushroom: {N_TRAIN} train, {N_TEST} test")

    return (X_train_img, y_train_out), (X_test_img, y_test_out)