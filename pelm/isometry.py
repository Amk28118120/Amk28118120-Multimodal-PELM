"""
isometry.py
=======================
Standalone script to plot distance preservation (isometry) for Check A.
Creates a 2x2 grid comparing MNIST, FSDD, and Abalone, plus a combined view,
for a selected embedding type.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ══════════════════════════════════════════════════════════════════════════════
#  ██████  TOGGLE EMBEDDING MODE HERE  ██████
# ══════════════════════════════════════════════════════════════════════════════
# Options: "fourier", "noise"
TARGET_EMBEDDING = "noise"

# Number of samples (50 samples = 1225 pairs, satisfying Check A's ~1000 pair requirement)
DIST_SAMPLES = 50
NPZ_DIR = "npz_files"
OUT_DIR = "kernel_analysis"

DATASETS = ["mnist", "fsdd", "abalone"]

DATA_LOADERS = {
    "mnist":   "data_loader.MNIST_data_loader.get_mnist",
    "fsdd":    "data_loader.audio_data_loader.get_fsdd",
    "abalone": "data_loader.abalone_data_loader.get_abalone",
}

COLORS  = {"mnist": "dodgerblue", "fsdd": "crimson", "abalone": "mediumseagreen"}
MARKERS = {"mnist": "o", "fsdd": "s", "abalone": "^"}
TITLES  = {"mnist": "MNIST", "fsdd": "FSDD (Audio)", "abalone": "Abalone"}

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def load_raw_loader(spec):
    if spec is None: return None
    mod_name, fn_name = spec.rsplit(".", 1)
    import importlib
    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, fn_name)
    except Exception as e:
        print(f"  [warn] Could not import {spec}: {e}")
        return None

def preprocess(H, is_raw):
    """
    Applies per-sample DC removal and L2 hyperspherical normalization 
    to exactly match Equation 9 of the PELM manuscript.
    """
    H = H.copy()
    if is_raw:
        H -= H.mean(axis=1, keepdims=True)
    H /= (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    return H

def load_npz(dataset, emb):
    path = f"{NPZ_DIR}/{dataset}_train_fold1_{emb}.npz"
    if not os.path.exists(path):
        print(f"  [Error] Cannot find {path}")
        return None
    
    data = np.load(path, allow_pickle=False)
    H = data["H_train"].astype(np.float64)
    
    # Process features to match manuscript Eq. 9
    return preprocess(H, is_raw=True)

def z_score(v): 
    return (v - v.mean()) / (v.std() + 1e-12)

# ─────────────────────────────────────────────────────────────────
#  MAIN EXECUTION
# ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*70)
    print(f"  CHECK A: DISTANCE PRESERVATION (Isometry)")
    print(f"  Embedding Mode : {TARGET_EMBEDDING.upper()}")
    print("="*70)

    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Store processed data for the combined plot
    plot_data = {}

    for ds in DATASETS:
        print(f"\n  Processing {ds.upper()}...")
        
        # 1. Load Raw Data (Digital)
        loader_spec = DATA_LOADERS.get(ds)
        loader_fn   = load_raw_loader(loader_spec)
        
        if not loader_fn:
            print(f"  [Skip] Could not load raw data for {ds}")
            continue
            
        (X_tr, _), _ = loader_fn()
        X_raw = X_tr[:DIST_SAMPLES].astype(np.float64).reshape(DIST_SAMPLES, -1)
        
        # 2. Load Optical Data (Reservoir)
        H_opt = load_npz(ds, TARGET_EMBEDDING)
        if H_opt is None: continue
        H_opt = H_opt[:DIST_SAMPLES]
        
        # 3. Calculate Distances
        dx = pdist(X_raw, "sqeuclidean")
        dh = pdist(H_opt, "sqeuclidean")
        
        dx_n = z_score(dx)
        dh_n = z_score(dh)
        
        rp, _ = pearsonr(dx_n, dh_n)
        rs, _ = spearmanr(dx_n, dh_n)
        
        print(f"  Pearson r = {rp:.4f}  |  Spearman ρ = {rs:.4f}")
        
        plot_data[ds] = {
            "dx": dx_n, "dh": dh_n, "rp": rp, "rs": rs
        }

    if not plot_data:
        print("\n  [Error] No data successfully processed. Check your paths.")
        return

    # 4. Generate 2x2 Grid Plot
    print("\n  Generating 2x2 Grid Plot...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    # Plot individual datasets
    for i, ds in enumerate(DATASETS):
        if ds not in plot_data: continue
        ax = axes[i]
        d = plot_data[ds]
        
        ax.scatter(d["dx"], d["dh"], color=COLORS[ds], marker=MARKERS[ds], alpha=0.5, s=30, edgecolors="none")
        ax.plot([-3, 3], [-3, 3], "k--", alpha=0.7, label="Ideal 1:1")
        
        ax.set_title(f"{TITLES[ds]}\nPearson r={d['rp']:.3f}  Spearman ρ={d['rs']:.3f}", fontsize=14, fontweight="bold")
        ax.set_xlabel("Original Space Dist ||x-y||² (Z-score)", fontsize=12)
        ax.set_ylabel("Optical Space Dist ||r(x)-r(y)||² (Z-score)", fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, ls=":", alpha=0.6)

    # Plot Combined view in the 4th quadrant
    ax_comb = axes[3]
    for ds, d in plot_data.items():
        ax_comb.scatter(d["dx"], d["dh"], color=COLORS[ds], marker=MARKERS[ds], 
                        alpha=0.5, s=30, edgecolors="none", 
                        label=f"{TITLES[ds]} (r={d['rp']:.2f})")
        
    ax_comb.plot([-3, 3], [-3, 3], "k--", alpha=0.7, label="Ideal 1:1")
    ax_comb.set_title(f"Combined Modalities ({TARGET_EMBEDDING.upper()})", fontsize=14, fontweight="bold")
    ax_comb.set_xlabel("Original Space Dist ||x-y||² (Z-score)", fontsize=12)
    ax_comb.set_ylabel("Optical Space Dist ||r(x)-r(y)||² (Z-score)", fontsize=12)
    ax_comb.legend(fontsize=11, loc="upper left")
    ax_comb.grid(True, ls=":", alpha=0.6)

    fig.suptitle(f"Check A: Distance Preservation in Optical Feature Space [{TARGET_EMBEDDING.upper()}]", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    
    fname = os.path.join(OUT_DIR, f"check_A_isometry_{TARGET_EMBEDDING}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    
    print(f"\n  ✅ Success! Saved to {fname}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
