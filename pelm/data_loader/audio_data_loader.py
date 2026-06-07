"""
Clean audio data loader for PELM.
Supports:
- FSDD (simple random split)
"""

import os
import numpy as np
import librosa
from pathlib import Path
from config import *
import csv
import shutil
import hashlib
import subprocess
import urllib.request
import zipfile
import cv2 

_FSDD_DIR = "./data/fsdd/recordings"


# ============================================================
# HANN TAPER
# ============================================================

def _apply_hann_taper(mel: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """
    Apply a 2D Hann-window taper to mel spectrogram edges.
    Smooths boundaries to zero → prevents sinc ringing in SLM Fourier plane.
    alpha: fraction of each axis to taper (0.1 = 10% on each side)
    """
    rows, cols = mel.shape

    def _hann_ramp(length, fade_len):
        window = np.ones(length)
        ramp = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
        window[:fade_len]  = ramp
        window[-fade_len:] = ramp[::-1]
        return window

    row_win = _hann_ramp(rows, int(rows * alpha))
    col_win = _hann_ramp(cols, int(cols * alpha))

    taper = np.outer(row_win, col_win)
    return (mel * taper).astype(np.float32)

# ============================================================
# MEL SPECTROGRAM
# ============================================================

def _get_mel_spec(wav_path: str) -> np.ndarray:
    
    # ==============================
    # LOAD AUDIO
    # ==============================
    y, sr = librosa.load(wav_path, sr=AUDIO_SAMPLE_RATE)

    #target_len = int(AUDIO_SAMPLE_RATE * AUDIO_DURATION)

    # # ==============================
    # # FIX LENGTH (VERY IMPORTANT)
    # # ==============================
    # if len(y) < target_len:
    #     y = np.pad(y, (0, target_len - len(y)))
    # else:
    #     start = (len(y) - target_len) // 2

    #     y = y[start:start + target_len]

    # ==============================
    # NORMALIZE AUDIO
    # ==============================
    #y = y / (np.max(np.abs(y)) + 1e-8)
    y = y / (np.max(np.abs(y)) + 1e-8)
    #y = y * 2.0
    #y = np.clip(y, -1.0, 1.0)
    

    # ==============================
    # MEL SPECTROGRAM
    # ==============================
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=AUDIO_N_MELS,
        n_fft=AUDIO_N_FFT,
        hop_length=AUDIO_HOP_LENGTH,
        power=2.0
    )

    # ==============================
    # LOG SCALE (CRITICAL)
    # ==============================
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # ==============================
    # DYNAMIC RANGE CLIPPING
    # ==============================
    log_mel = np.clip(log_mel, -80, 0)

    # ==============================
    # NORMALIZATION → [0,1]
    # ==============================
    log_mel = (log_mel + 80) / 80
    
    # ==============================
    # HANN TAPER (kills sinc boundary artifacts)
    # ==============================
    log_mel = _apply_hann_taper(log_mel, alpha=0.1)

    # ==============================
    # RESIZE TO MATCH PIPELINE
    # ==============================
    log_mel = cv2.resize(log_mel, (128, 128), interpolation=cv2.INTER_CUBIC)
    
    #log_mel = cv2.resize(log_mel, (128, 128), interpolation=cv2.INTER_AREA)

    
    #log_mel = cv2.resize(log_mel, (128, 128), interpolation=cv2.INTER_LANCZOS4)

    # ==============================
    # FINAL CLEANUP
    # ==============================
    log_mel = np.clip(log_mel, 0.0, 1.0)

    return log_mel.astype(np.float32)


# ============================================================
# DOWNLOADERS
# ============================================================

def _maybe_download_fsdd(data_dir: str = _FSDD_DIR):
    os.makedirs(data_dir, exist_ok=True)

    if len(list(Path(data_dir).glob("*.wav"))) > 2999:
        print(f"[FSDD] Already exists")
        return

    print("[FSDD] Downloading...")

    speakers = ["george", "jackson", "lucas", "nicolas", "theo", "yweweler"]
    base_url = "https://github.com/Jakobovski/free-spoken-digit-dataset/raw/master/recordings"

    for digit in range(10):
        for speaker in speakers:
            for idx in range(50):
                fname = f"{digit}_{speaker}_{idx}.wav"
                fpath = os.path.join(data_dir, fname)

                if os.path.exists(fpath):
                    continue

                url = f"{base_url}/{fname}"
                try:
                    urllib.request.urlretrieve(url, fpath)
                except Exception as e:
                    print(f"[FSDD] Failed to download {fname}: {e}")

    print("[FSDD] Done")

# ============================================================
# FSDD
# ============================================================

def get_fsdd():
    _maybe_download_fsdd(_FSDD_DIR)
    
    data_dir = Path(_FSDD_DIR)
    
    X_train, y_train = [], []
    X_test, y_test = [], []

    print("[FSDD] Processing and splitting data...")

    # FSDD has no subfolders, read directly from the directory
    for fpath in data_dir.glob("*.wav"):
        fname = fpath.name
        
        # FSDD format: {digit}_{speaker}_{index}.wav
        parts = fname.replace(".wav", "").split("_")
        if len(parts) != 3:
            continue
            
        digit = int(parts[0])
        idx = int(parts[2])

        try:
            mel = _get_mel_spec(str(fpath))
            
            # Official split: Indexes 0-4 for test (10%), 5-49 for train (90%)
            if idx <= 4:
                X_test.append(mel)
                y_test.append(digit)
            else:
                X_train.append(mel)
                y_train.append(digit)
        except Exception as e:
            print(f"[FSDD] Failed to process {fname}: {e}")

    X_train = np.stack(X_train).astype(np.float32)
    y_train = np.array(y_train)
    X_test = np.stack(X_test).astype(np.float32)
    y_test = np.array(y_test)

    # Shuffle the sets so the model doesn't train sequentially by digit/speaker
    np.random.seed(RANDOM_SEED)
    
    train_perm = np.random.permutation(len(X_train))
    X_train, y_train = X_train[train_perm], y_train[train_perm]
    
    test_perm = np.random.permutation(len(X_test))
    X_test, y_test = X_test[test_perm], y_test[test_perm]

    # Subset based on config.py (useful for your "Quick test" mode)
    X_train, y_train = X_train[:N_TRAIN], y_train[:N_TRAIN]
    X_test, y_test = X_test[:N_TEST], y_test[:N_TEST]

    print(f"[FSDD] Train {X_train.shape} | Test {X_test.shape}")

    return (X_train, y_train), (X_test, y_test)