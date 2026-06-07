"""
Deepfake Face data loader for hardware-based PELM experiments.
Reads from a manually downloaded local directory.
1. Loads ALL data from 'Real' and 'Fake' folders.
2. Randomizes Real and Fake classes INDEPENDENTLY.
3. Splits both classes into an 80/20 Train/Test pool (Stratified Split).
4. Samples exactly N_TRAIN and N_TEST (balanced 50/50 per class) from the pools.
"""
import os
import cv2
import numpy as np
from config import RANDOM_SEED, N_TRAIN, N_TEST

# MATCHES RAW DATASET TO AVOID DIGITAL BLURRING BEFORE EDGE DETECTION
IMG_SIZE = 256 #try with 1:1 mappign instead of upscaling

# Point this to exactly where you placed the unzipped folder
DATA_DIR = "./data/Final Dataset"

def _load_images_from_folder(folder_path, label):
    """Loads ALL images from the folder, applies Sobel edge detection, and shuffles them."""
    images = []
    labels = []
    
    if not os.path.exists(folder_path):
        print(f"[ERROR] Could not find folder: {folder_path}")
        return np.array(images), np.array(labels)
        
    for f in os.listdir(folder_path):
        if not f.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
            
        img_path = os.path.join(folder_path, f)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        if img is not None:
            # =================================================================
            # 1. SOBEL FIRST (On the raw, unaltered original pixels)
            # =================================================================
            sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = cv2.magnitude(sobelx, sobely)
            
            # Absolute clipping to preserve the true severity of deepfake artifacts
            img_edges_raw = np.clip(magnitude, 0, 255).astype(np.uint8)
            
            # =================================================================
            # 2. NOW RESIZE (Scale the extracted edges as a safety net)
            # =================================================================
            img_final = cv2.resize(img_edges_raw, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            
            images.append(img_final)
            labels.append(label)
            
    X = np.array(images)
    y = np.array(labels, dtype=np.int32)
    
    # Shuffle immediately upon loading
    np.random.seed(RANDOM_SEED + label) 
    shuffled_indices = np.random.permutation(len(X))
    
    return X[shuffled_indices], y[shuffled_indices]

def get_deepfake():
    print("[Data] Loading local Deepfake Dataset and applying Sobel filter...")
    
    # Matching the exact folder names from your screenshot
    real_path = os.path.join(DATA_DIR, "Real")
    fake_path = os.path.join(DATA_DIR, "Fake")
    
    # 1. LOAD AND SHUFFLE INDEPENDENTLY
    X_real, y_real = _load_images_from_folder(real_path, 0)
    X_fake, y_fake = _load_images_from_folder(fake_path, 1)
    
    total_real = len(X_real)
    total_fake = len(X_fake)
    print(f"[Data] Found {total_real} Real images and {total_fake} Fake images.")
    
    if total_real == 0 or total_fake == 0:
        raise ValueError("Missing images. Ensure 'Real' and 'Fake' folders exist inside './data/Final Dataset/' and contain images.")

    # 2. STRATIFIED 80/20 SPLIT FOR THE POOLS
    real_split_idx = int(total_real * 0.8)
    fake_split_idx = int(total_fake * 0.8)
    
    X_real_train_pool, y_real_train_pool = X_real[:real_split_idx], y_real[:real_split_idx]
    X_real_test_pool, y_real_test_pool   = X_real[real_split_idx:], y_real[real_split_idx:]
    
    X_fake_train_pool, y_fake_train_pool = X_fake[:fake_split_idx], y_fake[:fake_split_idx]
    X_fake_test_pool, y_fake_test_pool   = X_fake[fake_split_idx:], y_fake[fake_split_idx:]
    
    # 3. SAMPLE EXACTLY (N_TRAIN / 2) AND (N_TEST / 2)
    n_train_per_class = N_TRAIN // 2
    n_test_per_class = N_TEST // 2
    
    if n_train_per_class > len(X_real_train_pool) or n_train_per_class > len(X_fake_train_pool):
        print(f"[WARN] N_TRAIN/2 ({n_train_per_class}) exceeds the balanced training pool. Capping.")
        n_train_per_class = min(len(X_real_train_pool), len(X_fake_train_pool))
        
    if n_test_per_class > len(X_real_test_pool) or n_test_per_class > len(X_fake_test_pool):
        print(f"[WARN] N_TEST/2 ({n_test_per_class}) exceeds the balanced testing pool. Capping.")
        n_test_per_class = min(len(X_real_test_pool), len(X_fake_test_pool))

    X_train = np.concatenate([X_real_train_pool[:n_train_per_class], X_fake_train_pool[:n_train_per_class]])
    y_train = np.concatenate([y_real_train_pool[:n_train_per_class], y_fake_train_pool[:n_train_per_class]])
    
    X_test = np.concatenate([X_real_test_pool[:n_test_per_class], X_fake_test_pool[:n_test_per_class]])
    y_test = np.concatenate([y_real_test_pool[:n_test_per_class], y_fake_test_pool[:n_test_per_class]])

    # 4. FINAL SHUFFLE
    np.random.seed(RANDOM_SEED)
    train_shuff_idx = np.random.permutation(len(X_train))
    test_shuff_idx = np.random.permutation(len(X_test))
    
    X_train, y_train = X_train[train_shuff_idx], y_train[train_shuff_idx]
    X_test, y_test = X_test[test_shuff_idx], y_test[test_shuff_idx]

    print(f"[Data] Final Deepfake Selection: {len(X_train)} Train | {len(X_test)} Test")
    
    return (X_train, y_train), (X_test, y_test)