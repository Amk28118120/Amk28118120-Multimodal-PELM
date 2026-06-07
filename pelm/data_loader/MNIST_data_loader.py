"""
MNIST loader for hardware-based PELM experiments.
"""
import numpy as np
from torchvision import datasets
from config import *

def get_mnist():
    print("[Data] Loading MNIST...")
    
    train_d = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=None  # explicit: raw uint8
    )
    
    test_d = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=None
    )

    # Convert to numpy
    X_train_full = train_d.data.numpy()
    y_train_full = train_d.targets.numpy()

    X_test_full = test_d.data.numpy()
    y_test_full = test_d.targets.numpy()

    # -------------------------------------------------
    # random permutation before slicing
    # -------------------------------------------------
    np.random.seed(RANDOM_SEED)

    train_idx = np.random.permutation(len(X_train_full))[:N_TRAIN]
    test_idx  = np.random.permutation(len(X_test_full))[:N_TEST]

    X_train = X_train_full[train_idx]
    y_train = y_train_full[train_idx]

    X_test = X_test_full[test_idx]
    y_test = y_test_full[test_idx]
    # -------------------------------------------------

    print(f"[Data] Loaded {N_TRAIN} train, {N_TEST} test samples")
    
    return (X_train, y_train), (X_test, y_test)
