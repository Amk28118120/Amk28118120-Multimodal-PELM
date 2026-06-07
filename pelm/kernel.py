"""
experimental_kernel.py
=======================
Experimentally identify the PELM optical kernel across all embeddings
and compare to exact, double-centered theoretical predictions.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats    import binned_statistic, pearsonr
from scipy.optimize import minimize_scalar

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# =============================================================================
# 1. TOGGLE DATASET MODE HERE
# =============================================================================

# Options: "mnist", "fsdd", "abalone", "mushroom"
TARGET_DATASET = "mushroom"
EMBEDDINGS     = ["noise", "fourier"]

# =============================================================================
# 2. EXPERIMENT CONFIGURATION
# =============================================================================

DATA_SPLIT = "train"
SPLIT_SEED = 42

N_SAMPLES = 500       # Number of samples to process 
N_PAIRS   = 10000     # Number of random pairs for kernel estimation 
N_BINS    = 30        # Statistical binning
BASE_OUT_DIR = "kernel_analysis"

REMOVE_TOP_MODES          = 0
REMOVE_DC_BACKGROUND      = True   # Stage 1: axis=1 (per-sample DC removal)
CENTER_EMPIRICAL_FEATURES = True   # Stage 2: axis=0 (per-feature dataset centering)


def get_npz_paths(dataset, base_dir="/Users/anushkakumari/tcspc/pelm/npz_files"):
    """Returns the dictionary of file paths based on the chosen dataset."""
    return {
        emb: f"{base_dir}/{dataset}_{DATA_SPLIT}_fold1_{emb}.npz" for emb in EMBEDDINGS
    }

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — EXACT THEORETICAL MATRICES (CHECK 4 COMPLIANT)
# ─────────────────────────────────────────────────────────────────────────────

def center_matrix(K):
    """Rigorous double-centering of a kernel matrix."""
    row_mean = K.mean(axis=1, keepdims=True)
    col_mean = K.mean(axis=0, keepdims=True)
    return K - row_mean - col_mean + K.mean()

def K_gaussian_mat(X):
    """K_Gaussian(theta) = 1 + cos^2(theta) for complex Gaussian matrices."""
    C = np.clip(X @ X.T, -1.0, 1.0)
    return 1.0 + C**2

def K_phase_mat(X):
    """Exact K_phase without CLT approximation."""
    C = np.clip(X @ X.T, -1.0, 1.0)
    X_sq = X**2
    return 1.0 + C**2 - (X_sq @ X_sq.T) / X.shape[1]

def K2_mat(X):
    """Arc-cosine K2 kernel (Cho & Saul)."""
    C = np.clip(X @ X.T, -1.0, 1.0)
    theta = np.arccos(C)
    return (np.sin(theta) + (np.pi - theta) * np.cos(theta)) / np.pi

def K1_mat(X):
    """Arc-cosine K1 kernel (Cho & Saul)."""
    C = np.clip(X @ X.T, -1.0, 1.0)
    theta = np.arccos(C)
    return (np.pi - theta) / np.pi

def RBF_mat(X, gamma):
    """Standard RBF Kernel."""
    C = np.clip(X @ X.T, -1.0, 1.0)
    theta = np.arccos(C)
    return np.exp(-gamma * theta**2)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_optical_features(npz_path, split):
    data = np.load(npz_path, allow_pickle=False)
    keys = list(data.keys())
    
    h_key = f"H_{split}"
    y_key = f"y_{split}"

    if h_key not in keys:
        raise KeyError(f"Cannot find {h_key} in {npz_path}.")

    H = data[h_key].astype(np.float64)
    last_idx = int(data["last_idx"]) if "last_idx" in keys else len(H)
    H = H[:last_idx]
    
    y = data[y_key].astype(np.int64)[:last_idx] if y_key in keys else None
    return H, y

def load_original_data(dataset, n_samples, split, seed):
    dataset = dataset.lower()

    if dataset == "mnist":
        from data_loader.MNIST_data_loader import get_mnist
        train_data, test_data = get_mnist()
    elif dataset in ("fsdd", "audio"):
        from data_loader.audio_data_loader import get_fsdd
        train_data, test_data = get_fsdd() 
    elif dataset == "abalone":
        from data_loader.abalone_data_loader import get_abalone
        train_data, test_data = get_abalone()
    elif dataset == "mushroom":
        from data_loader.Mushroom_data_loader import get_mushroom
        train_data, test_data = get_mushroom()
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")

    X, y = train_data if split == "train" else test_data
    X = np.array(X[:n_samples], dtype=np.float64).reshape(min(n_samples, len(X)), -1)
    y = np.array(y[:n_samples])
    print(f"  [Data] Loaded Original {dataset.upper()} ({split}): X={X.shape}, y={y.shape}")
    return X, y

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_features(H, remove_common_mode=True, normalize=True):
    H = H.astype(np.float64).copy()
    if remove_common_mode:
        H -= H.mean(axis=1, keepdims=True)
    if normalize:
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        H /= (norms + 1e-12)
    return H

def preprocess_inputs(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (norms + 1e-12)

def remove_top_singular_modes(H, n_remove=1):
    print("  [Preproc] Removing dominant singular modes...")
    U, S, Vt = np.linalg.svd(H, full_matrices=False)
    total_energy = np.sum(S**2)
    H_clean = H.copy()

    for k in range(n_remove):
        frac = (S[k]**2) / total_energy
        component = np.outer(U[:, k] * S[k], Vt[k])
        H_clean -= component

    norms = np.linalg.norm(H_clean, axis=1, keepdims=True)
    H_clean /= (norms + 1e-12)
    return H_clean

def center_kernel_features(H):
    H = H - H.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(H, axis=1, keepdims=True)
    H /= (norms + 1e-12)
    return H

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — EMPIRICAL KERNEL COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_empirical_kernel(H_norm, X_norm, n_pairs=10000, seed=42):
    rng = np.random.default_rng(seed)
    N   = len(H_norm)

    idx_i = rng.integers(0, N, int(n_pairs * 1.1))
    idx_j = rng.integers(0, N, int(n_pairs * 1.1))
    mask  = idx_i != idx_j
    idx_i, idx_j = idx_i[mask][:n_pairs], idx_j[mask][:n_pairs]

    cos_theta = np.clip(np.einsum('ij,ij->i', X_norm[idx_i], X_norm[idx_j]), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    K_emp = np.einsum('ij,ij->i', H_norm[idx_i], H_norm[idx_j])

    return theta, K_emp, idx_i, idx_j

def bin_kernel(theta, K_emp, n_bins=30):
    MIN_COUNT = 10
    edges = np.linspace(0.0, np.pi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    mean_K, _, _ = binned_statistic(theta, K_emp, statistic="mean",  bins=edges)
    std_K,  _, _ = binned_statistic(theta, K_emp, statistic="std",   bins=edges)
    count,  _, _ = binned_statistic(theta, K_emp, statistic="count", bins=edges)

    valid = count >= MIN_COUNT
    std_K = np.where(valid, std_K, np.nan)
    return centers, edges, mean_K, std_K, count, valid

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — EXACT DOUBLE-CENTERED FIT
# ─────────────────────────────────────────────────────────────────────────────

def fit_and_bin_theoretical_kernels(X_norm, K_emp, theta, idx_i, idx_j, edges):
    def _process(K_mat):
        K_cen = center_matrix(K_mat)
        K_pairs = K_cen[idx_i, idx_j]
        
        # Least squares scaling factor alpha
        alpha = np.dot(K_emp, K_pairs) / (np.dot(K_pairs, K_pairs) + 1e-12)
        K_fit = alpha * K_pairs
        
        r, _ = pearsonr(K_emp, K_fit)
        ss_res = np.sum((K_emp - K_fit)**2)
        ss_tot = np.sum((K_emp - K_emp.mean())**2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        rmse = np.sqrt(ss_res / len(K_emp))
        
        binned_mean, _, _ = binned_statistic(theta, K_fit, "mean", bins=edges)
        return {"alpha": alpha, "r": r, "r2": r2, "rmse": rmse, "binned": binned_mean, "raw_fit": K_fit}

    results = {}
    results["K_gaussian"] = _process(K_gaussian_mat(X_norm))
    results["K_phase"]    = _process(K_phase_mat(X_norm))
    results["K2"]         = _process(K2_mat(X_norm))
    results["K1"]         = _process(K1_mat(X_norm))

    def rbf_opt(gamma):
        K_cen = center_matrix(RBF_mat(X_norm, gamma))
        K_pairs = K_cen[idx_i, idx_j]
        alpha = np.dot(K_emp, K_pairs) / (np.dot(K_pairs, K_pairs) + 1e-12)
        return np.sum((K_emp - alpha * K_pairs)**2)
    
    res = minimize_scalar(rbf_opt, bounds=(0.1, 5.0), method='bounded')
    best_gamma = res.x
    rbf_res = _process(RBF_mat(X_norm, best_gamma))
    rbf_res["gamma"] = best_gamma
    results["RBF"] = rbf_res

    return results

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 8 — PLOTTING (JOURNAL-READY VERSION)
# ─────────────────────────────────────────────────────────────────────────────

def plot_kernel_comparison(theta_bins, K_mean, K_std, count, valid, fits, dataset, tag, out_dir):
    plt.style.use('seaborn-v0_8-paper')
    
    theta_v = theta_bins[valid]
    K_v     = K_mean[valid]
    K_std_v = K_std[valid]
    cnt_v   = count[valid]
    
    K2_v         = fits["K2"]["binned"][valid]
    K_gauss_v    = fits["K_gaussian"]["binned"][valid]
    K_phase_v    = fits["K_phase"]["binned"][valid]
    K1_v         = fits["K1"]["binned"][valid]
    Krbf_v       = fits["RBF"]["binned"][valid]
    residual     = K_v - K2_v

    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, wspace=0.25)

    # Panel 1: Empirical vs Theoretical
    ax1 = fig.add_subplot(gs[0])
    ax1.errorbar(np.degrees(theta_v), K_v, yerr=K_std_v / np.sqrt(np.maximum(cnt_v, 1)),
                fmt="o", color="black", ms=4, capsize=2, linewidth=0.8, label="Empirical PELM kernel", zorder=5)
    
    ax1.plot(np.degrees(theta_v), K_phase_v, "m-",  lw=2, label=f"Exact K_phase (r={fits['K_phase']['r']:.2f})")
    ax1.plot(np.degrees(theta_v), K_gauss_v, "c-",  lw=1.5, label=f"K_Gaussian (r={fits['K_gaussian']['r']:.2f})")
    ax1.plot(np.degrees(theta_v), K2_v, "b--",  lw=1.5, label=f"Arc-cosine K₂ (r={fits['K2']['r']:.2f})")
    ax1.plot(np.degrees(theta_v), K1_v, "g-.", lw=1, label=f"Arc-cosine K₁ (r={fits['K1']['r']:.2f})")
    ax1.plot(np.degrees(theta_v), Krbf_v, "r:", lw=1.5, label=f"RBF (r={fits['RBF']['r']:.2f}, γ={fits['RBF']['gamma']:.2f})")

    ax1.axhline(0, color="gray", lw=0.5, ls="--", alpha=0.5)
    ax1.set_xlabel("Input angle θ (°)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Kernel K(θ)", fontsize=12, fontweight="bold")
    ax1.set_title(f"{dataset.upper()} | {tag}", fontsize=13, fontweight="bold")
    
    y_max = np.nanmax(K_v + K_std_v / np.sqrt(np.maximum(cnt_v, 1)))
    y_min = np.nanmin(K_v - K_std_v / np.sqrt(np.maximum(cnt_v, 1)))
    ax1.set_ylim(y_min * 1.2, y_max * 1.8) 

    ax1.legend(fontsize=9, loc="upper right", frameon=True, edgecolor="black")
    ax1.set_xlim(0, 90)
    ax1.grid(True, alpha=0.2)

    # Panel 2: Residuals
    ax2 = fig.add_subplot(gs[1])
    ax2.bar(np.degrees(theta_v), residual, width=np.degrees(theta_bins[1]-theta_bins[0])*0.8, 
            color="#e74c3c", alpha=0.6)
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("Input angle θ (°)", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Residual (K_emp − K₂)", fontsize=12, fontweight="bold")
    ax2.set_title(f"Residuals (RMSE={fits['K2']['rmse']:.4f})", fontsize=13, fontweight="bold")
    ax2.set_xlim(0, 90)
    ax2.grid(True, alpha=0.2)

    # Panel 3: Goodness of Fit
    ax3 = fig.add_subplot(gs[2])
    kernels = ["Phase", "Gauss", "K₂", "K₁", "RBF"]
    r_vals = [fits["K_phase"]["r"], fits["K_gaussian"]["r"], fits["K2"]["r"], fits["K1"]["r"], fits["RBF"]["r"]]
    
    x = np.arange(len(kernels))
    ax3.bar(x, r_vals, width=0.6, color="#2980b9", alpha=0.7)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Pearson Correlation (r)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x); ax3.set_xticklabels(kernels, fontsize=11)
    ax3.set_title("Fit Quality", fontsize=13, fontweight="bold")
    ax3.grid(True, alpha=0.2, axis="y")

    fname = os.path.join(out_dir, f"{tag[0].lower()}_kernel_comparison_{dataset.lower()}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    return fname

def plot_class_kernel_matrix(H_norm, y, dataset, tag, out_dir):
    classes = np.unique(y)
    n_cls   = len(classes)
    K_mat   = np.zeros((n_cls, n_cls))

    for i, c1 in enumerate(classes):
        for j, c2 in enumerate(classes):
            m1, m2 = (y == c1), (y == c2)
            K_mat[i, j] = (H_norm[m1] @ H_norm[m2].T).mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(K_mat, cmap="RdYlGn", vmin=-0.2, vmax=1.0, aspect="auto")
    ax.set_xticks(range(n_cls)); ax.set_xticklabels(classes)
    ax.set_yticks(range(n_cls)); ax.set_yticklabels(classes)
    ax.set_xlabel("Class", fontsize=15, fontweight="bold")
    ax.set_ylabel("Class", fontsize=15, fontweight="bold")
    ax.set_title(f"Class-Averaged Kernel Matrix\n{dataset.upper()} | {tag}", fontsize=15)
    plt.colorbar(im, ax=ax)

    for i in range(n_cls):
        for j in range(n_cls):
            color = "white" if abs(K_mat[i, j]) > 0.5 else "black"
            ax.text(j, i, f"{K_mat[i,j]:.2f}", ha="center", va="center", fontsize=12, color=color)

    fname = os.path.join(out_dir, f"{tag[0].lower()}_kernel_matrix_{dataset.lower()}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    return fname

def plot_kernel_scatter(theta, K_emp, fits, dataset, tag, out_dir):
    plt.style.use('seaborn-v0_8-paper')
    theta_d = np.degrees(theta)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    idx = np.random.choice(len(theta), min(3000, len(theta)), replace=False)
    ax.scatter(theta_d[idx], K_emp[idx], alpha=0.15, s=8, color="steelblue", label="Individual pairs", rasterized=True)

    edges   = np.linspace(0, np.pi, 31)
    centers = 0.5 * (edges[:-1] + edges[1:])
    mean_K, _, _ = binned_statistic(theta, K_emp, "mean", bins=edges)
    ax.plot(np.degrees(centers), mean_K, "ko-", ms=5, lw=2, label="Binned mean")

    # Plot exact pairs for K_phase
    ax.scatter(theta_d[idx], fits["K_phase"]["raw_fit"][idx], alpha=0.15, s=8, color="magenta", label="K_phase Pairs", rasterized=True)
    
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Input angle θ (degrees)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Optical kernel K_emp(θ)",  fontsize=15, fontweight="bold")
    ax.set_title(f"Raw Pair Scatter — PELM Kernel\n{dataset.upper()} | {tag}", fontsize=15)
    ax.legend(fontsize=12)
    ax.set_xlim(0, 180)
    ax.grid(True, alpha=0.3)
    
    fname = os.path.join(out_dir, f"{tag[0].lower()}_kernel_scatter_{dataset.lower()}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 9 — MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 65)
    print("  PELM Experimental Kernel Identification (Exact Centering)")
    print(f"  Target Dataset: {TARGET_DATASET.upper()}")
    print("=" * 65)

    npz_paths = get_npz_paths(TARGET_DATASET)

    # 1. Load original data ONCE for all embeddings
    try:
        X_raw, y_orig = load_original_data(TARGET_DATASET, N_SAMPLES, DATA_SPLIT, SPLIT_SEED)
        X_norm = preprocess_inputs(X_raw)
    except Exception as e:
        print(f"\n[ERROR] Failed to load original data for {TARGET_DATASET}: {e}")
        return

    # 2. Iterate through Embeddings
    for emb, npz_file in npz_paths.items():
        if not os.path.exists(npz_file):
            print(f"\n[SKIP] {emb.upper()} — NPZ file not found: {npz_file}")
            continue

        print(f"\n{'─' * 65}")
        print(f"  Processing Embedding: {emb.upper()}")
        print(f"{'─' * 65}")

        # Setup output directory
        out_dir = os.path.join(BASE_OUT_DIR, TARGET_DATASET, emb)
        os.makedirs(out_dir, exist_ok=True)
        tag = f"{emb}"

        # Load optical features
        try:
            H_raw, y_npz = load_optical_features(npz_file, DATA_SPLIT)
            H_raw = H_raw[:N_SAMPLES]
            y_npz = y_npz[:N_SAMPLES] if y_npz is not None else None
            y = y_npz if y_npz is not None else y_orig.astype(np.int64)
        except Exception as e:
            print(f"  [ERROR] Failed to load features for {emb}: {e}")
            continue

        # Preprocess features
        H_norm = preprocess_features(H_raw, remove_common_mode=REMOVE_DC_BACKGROUND, normalize=True)
        if REMOVE_TOP_MODES > 0:
            H_norm = remove_top_singular_modes(H_norm, n_remove=REMOVE_TOP_MODES)
        if CENTER_EMPIRICAL_FEATURES:
            H_norm = center_kernel_features(H_norm)

        # Compute empirical kernel
        print(f"  [Step 3] Computing empirical kernel ({N_PAIRS:,} pairs)...")
        theta, K_emp, idx_i, idx_j = compute_empirical_kernel(H_norm, X_norm, n_pairs=N_PAIRS)

        # Bin by angle
        print("  [Step 4] Binning by input angle...")
        theta_bins, edges, K_mean, K_std, count, valid = bin_kernel(theta, K_emp, n_bins=N_BINS)

        # Fit theoretical kernels
        print("  [Step 5] Fitting theoretical kernels via Double-Centering...")
        fits = fit_and_bin_theoretical_kernels(X_norm, K_emp, theta, idx_i, idx_j, edges)

        best_name = max(fits, key=lambda k: fits[k]["r"])

        # Plotting
        print("  [Step 6] Generating plots...")
        plot_kernel_comparison(theta_bins, K_mean, K_std, count, valid, fits, TARGET_DATASET, tag, out_dir)
        plot_kernel_scatter(theta, K_emp, fits, TARGET_DATASET, tag, out_dir)
        if y is not None:
            plot_class_kernel_matrix(H_norm, y, TARGET_DATASET, tag, out_dir)

        # Print local summary
        print(f"\n  RESULTS SUMMARY — {emb.upper()}")
        print(f"  Best fitting kernel    : {best_name.upper()}")
        print(f"  Pearson r vs K_phase   : {fits['K_phase']['r']:.4f}")
        print(f"  Pearson r vs K2        : {fits['K2']['r']:.4f}")
        print(f"  RMSE vs K2             : {fits['K2']['rmse']:.4f}")
        print(f"  Plots saved to         : {os.path.abspath(out_dir)}/")

    print(f"\n{'=' * 65}")
    print("  ALL EMBEDDINGS PROCESSED")
    print(f"{'=' * 65}\n")

if __name__ == "__main__":
    main()
