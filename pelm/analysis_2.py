"""
Change the TARGET_DATASET toggle below.
For regression, the label kernel is replaced by a target-similarity kernel.
No separation for abalone (regression metric, not classification).

"""

import os, warnings
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial.distance   import pdist
from scipy.stats              import pearsonr, spearmanr
from sklearn.manifold         import TSNE
from sklearn.metrics.pairwise import euclidean_distances

os.environ["OMP_NUM_THREADS"]      = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

# ══════════════════════════════════════════════════════════════════════════════
TARGET_DATASET = "fsdd"   # "mnist", "fsdd", "abalone", "mushroom"
EMBEDDINGS = ["noise", "fourier"]
# ══════════════════════════════════════════════════════════════════════════════

def get_npz_paths(dataset, base_dir="/Users/anushkakumari/tcspc/pelm/npz_files"):
    return {emb: f"{base_dir}/{dataset}_train_fold1_{emb}.npz" for emb in EMBEDDINGS}

KNOWN_ACCURACY_ALL = {
    "noise": {
        "mnist":    {0: 97.10, 1: 99.21, 2: 95.63, 3: 96.14, 4: 96.64,
                     5: 96.75, 6: 96.56, 7: 93.78, 8: 94.97, 9: 94.84},
        "fsdd":     {0: 96.67, 1: 96.67, 2: 93.33, 3: 90.00, 4: 93.33,
                     5: 96.67, 6: 86.67, 7: 100.0, 8: 80.00, 9: 93.33},
        "abalone":  None,
        "mushroom": {0: 100.0, 1: 100.0},
    },
    "fourier": {
        "mnist":    {0: 98.88, 1: 99.38, 2: 96.32, 3: 97.43, 4: 97.25,
                     5: 97.20, 6: 98.12, 7: 95.91, 8: 97.02, 9: 95.04},
        "fsdd":     {0: 96.67, 1: 96.67, 2: 93.33, 3: 93.33, 4: 100.0,
                     5: 100.0, 6: 96.67, 7: 100.0, 8: 83.33, 9: 96.67},
        "abalone":  None,
        "mushroom": {0: 100.0, 1: 100.0},
    },
}

DATA_LOADERS = {
    "mnist":    "data_loader.MNIST_data_loader.get_mnist",
    "fsdd":     "data_loader.audio_data_loader.get_fsdd",
    "abalone":  "data_loader.abalone_data_loader.get_abalone",
    "mushroom": None,
}

MAX_PER_CLASS_CKA = 20
DIST_SAMPLES      = 50

TSNE_TARGET_N = {
    "mnist": 5000, "fsdd": 2000, "mushroom": 2000, "abalone": 2000,
}

# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def load_npz(key, path):
    data     = np.load(path, allow_pickle=False)
    keys     = list(data.keys())
    is_train = "H_train" in keys
    hk = "H_train" if is_train else "H_test"
    yk = "y_train" if is_train else "y_test"
    H  = data[hk].astype(np.float64)
    y  = data[yk].astype(np.int64 if np.issubdtype(data[yk].dtype, np.integer) else np.float64)
    if "last_idx" in keys:
        last = int(data["last_idx"])
        if last < len(H):
            print(f"  ⚠  {key}: truncating to {last} valid rows")
            H, y = H[:last], y[:last]
    print(f"  {key}: {len(H)} samples  {'(raw train)' if is_train else '(test)'}")
    return H, y, is_train


def preprocess_readout(H, is_raw):
    """
    Paper Eq.9 + Eq.11 — used for ridge regression, distance preservation,
    and t-SNE (no column centering).
    """
    H = H.copy()
    if is_raw:
        H -= H.mean(axis=1, keepdims=True)          # Eq.9
    H /= (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)  # Eq.11
    return H


def preprocess_kernel(H, is_raw):
    H = H.copy()
    if is_raw:
        H -= H.mean(axis=1, keepdims=True)          # Eq.9
    H -= H.mean(axis=0, keepdims=True)              # Eq.10
    H /= (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)  # Eq.11 (once)
    return H


def balanced_subset(H, y, n_per_class, dataset_name):
    rng = np.random.default_rng(42)
     # ─────────────────────────────────────────────────────────
    # REGRESSION FIX: Pure random uniform sampling for Abalone
    # ─────────────────────────────────────────────────────────
    if dataset_name == "abalone":
        target_n = min(len(H), 2000) 
        idx = rng.choice(len(H), size=target_n, replace=False)
        return H[idx], y[idx]
        
    unique_classes, counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[counts >= 5]
    if len(valid_classes) < 2:
        valid_classes = unique_classes[counts >= 2]
    if len(valid_classes) < 2:
        print("  [warn] Too few samples per class. Returning raw.")
        return H, y
    valid_counts     = np.array([np.sum(y == c) for c in valid_classes])
    actual_per_class = min(n_per_class, np.min(valid_counts))
    idx = []
    for c in valid_classes:
        ci = np.where(y == c)[0]
        idx.extend(rng.choice(ci, size=actual_per_class, replace=False))
    idx = np.array(idx)
    rng.shuffle(idx)
    return H[idx], y[idx]

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

# ─────────────────────────────────────────────────────────────────
#  KERNEL FUNCTIONS
# ─────────────────────────────────────────────────────────────────

def linear_kernel(H):
    return H @ H.T

def angular_rbf_kernel(H, gamma=None):
    C     = np.clip(H @ H.T, -1.0, 1.0)
    theta = np.arccos(C)
    if gamma is None:
        vals  = theta[np.triu_indices_from(theta, k=1)]
        gamma = 1.0 / (np.median(vals)**2 + 1e-12)
    return np.exp(-gamma * theta**2), gamma

def arc_cosine_k2_kernel(H):
    C     = np.clip(H @ H.T, -1.0, 1.0)
    theta = np.arccos(C)
    return (1.0 / np.pi) * (np.sin(theta) + (np.pi - theta) * C)

def ideal_kernel(y):
    y = y.astype(np.float64)
    if len(np.unique(y)) > 15:
        dy2   = (y[:, None] - y[None, :])**2
        gamma = np.median(dy2[dy2 > 0]) + 1e-12
        return np.exp(-dy2 / (2 * gamma))
    return (y[:, None] == y[None, :]).astype(np.float64)

def center_kernel(K):
    row_mean    = K.mean(axis=1, keepdims=True)
    col_mean    = K.mean(axis=0, keepdims=True)
    global_mean = K.mean()
    return K - row_mean - col_mean + global_mean

def cka_score(K, Ki):
    Kc  = center_kernel(K)
    Kci = center_kernel(Ki)
    num   = np.sum(Kc * Kci)
    denom = np.sqrt(np.sum(Kc**2) * np.sum(Kci**2))
    return float(num / (denom + 1e-12))

def separation(K, Ki, m=4096):
    Kc          = center_kernel(K)
    n           = Kc.shape[0]
    mask_diag   = np.eye(n, dtype=bool)
    mask_within  = (Ki == 1) & ~mask_diag
    mask_between = (Ki == 0)
    within_vals  = Kc[mask_within]
    between_vals = Kc[mask_between]
    w_mean = np.mean(within_vals)  if within_vals.size  > 0 else float("nan")
    b_mean = np.mean(between_vals) if between_vals.size > 0 else float("nan")
    w_var  = np.var(within_vals,  ddof=1) if within_vals.size  > 1 else 0
    b_var  = np.var(between_vals, ddof=1) if between_vals.size > 1 else 0
    pooled_std = np.sqrt((w_var + b_var) / 2.0) + 1e-12
    eta = w_mean - b_mean
    return {"within": w_mean, "between": b_mean,
            "sep": eta / pooled_std, "eta": eta, "snr": np.sqrt(m) * eta}

def per_class_sep(K, y, m=4096):
    Kc  = center_kernel(K)
    out = {}
    for c in np.unique(y):
        mask = (y == c)
        K_within  = Kc[np.ix_(mask, ~mask)]   # between
        K_between = Kc[np.ix_(mask, mask)]     # wait — swap names below
        # correct: within = same-class pairs, between = cross-class pairs
        K_w = Kc[np.ix_(mask, mask)]
        K_b = Kc[np.ix_(mask, ~mask)]
        idx_upper   = np.triu_indices_from(K_w, k=1)
        within_vals = K_w[idx_upper]
        between_vals = K_b.flatten()
        w_mean = np.mean(within_vals)  if within_vals.size  > 0 else float("nan")
        b_mean = np.mean(between_vals) if between_vals.size > 0 else float("nan")
        w_var  = np.var(within_vals,  ddof=1) if within_vals.size  > 1 else 0
        b_var  = np.var(between_vals, ddof=1) if between_vals.size > 1 else 0
        pooled_std = np.sqrt((w_var + b_var) / 2.0) + 1e-12
        eta = w_mean - b_mean
        out[c] = {"within": w_mean, "between": b_mean,
                  "sep": eta / pooled_std, "eta": eta, "snr": np.sqrt(m) * eta}
    return out

def heuristic_sep_label(s):
    if s > 1.5: return "Strong"
    if s > 0.5: return "Good"
    if s > 0.1: return "Weak"
    return "Failed"

# ─────────────────────────────────────────────────────────────────
#  1.  CKA + SEPARATION
# ─────────────────────────────────────────────────────────────────

def run_cka_analysis(key, H_kernel, y, acc_map, out_dir, dataset_name):
    H_kernel, y = balanced_subset(H_kernel, y, MAX_PER_CLASS_CKA, dataset_name=dataset_name)

    print(f"\n{'─'*75}")
    print(f"  CKA  [{key.upper()}]  N={len(y)}  (Eq.9→10→11 representation)")
    print(f"{'─'*75}")

    K_lin        = linear_kernel(H_kernel)
    K_rbf, gamma = angular_rbf_kernel(H_kernel)
    K_k2         = arc_cosine_k2_kernel(H_kernel)
    K_ideal      = ideal_kernel(y)

    s_lin = cka_score(K_lin, K_ideal)
    s_rbf = cka_score(K_rbf, K_ideal)
    s_k2  = cka_score(K_k2,  K_ideal)

    g_lin  = separation(K_lin, K_ideal)
    g_rbf  = separation(K_rbf, K_ideal)
    pc_lin = per_class_sep(K_lin, y)
    pc_rbf = per_class_sep(K_rbf, y)

    print(f"\n  Global CKA  →  Linear: {s_lin:.4f} | RBF: {s_rbf:.4f} | K2 (ReLU): {s_k2:.4f}")
    print(f"  Global Sep  →  Linear d: {g_lin['sep']:.4f}  | RBF d: {g_rbf['sep']:.4f} ({heuristic_sep_label(g_rbf['sep'])})")
    print(f"  SNR (√mη)   →  Linear: {g_lin['snr']:.2f} | RBF: {g_rbf['snr']:.2f}")
    print(f"  RBF gamma   = {gamma:.5f}")

    if acc_map:
        valid_classes = [c for c in sorted(np.unique(y))
                        if c in acc_map and not np.isnan(acc_map[c])]
        if len(valid_classes) > 1:
            acc_vals = [acc_map[c] for c in valid_classes]
            sep_vals = [pc_rbf[c]['sep'] for c in valid_classes]
            rho, p = spearmanr(sep_vals, acc_vals)
            print(f"  Spearman ρ (RBF Sep vs Accuracy) = {rho:.3f} (p={p:.3f})")

    print(f"\n  {'Class':<6} | {'RBF d':>8} | {'RBF √mη':>9} | {'Acc (%)':>7}")
    print(f"  {'─'*38}")
    for c in sorted(np.unique(y)):
        if c in pc_lin and c in pc_rbf:
            rs   = pc_rbf[c]['sep']
            rsnr = pc_rbf[c]['snr']
            ac   = acc_map.get(c, float('nan')) if acc_map else float('nan')
            ac_str = f"{ac:>7.2f}" if not np.isnan(ac) else "    ---"
            print(f"  {c:<6} | {rs:>8.4f} | {rsnr:>9.2f} | {ac_str}")

    _save_kernel_heatmaps(key, K_lin, K_rbf, K_ideal, y, s_lin, s_rbf,
                        g_lin, g_rbf, out_dir, dataset_name)
    return {
        "s_lin": s_lin, "s_rbf": s_rbf, "s_k2": s_k2,
        "g_lin": g_lin, "g_rbf": g_rbf,
        "pc_lin": pc_lin, "pc_rbf": pc_rbf,
        "y": y, "acc": acc_map,
    }


def _save_kernel_heatmaps(key, K_lin, K_rbf, K_ideal, y, s_lin, s_rbf,
                        g_lin, g_rbf, out_dir, dataset_name):
    for tag, K, score, glob in [("Linear", K_lin, s_lin, g_lin),
                                ("RBF",    K_rbf, s_rbf, g_rbf)]:
        idx  = np.argsort(y)
        Ks   = K[np.ix_(idx, idx)]
        Ki_s = K_ideal[np.ix_(idx, idx)]
        ov_s = (center_kernel(K.copy()) * center_kernel(K_ideal.copy()))[np.ix_(idx, idx)]

        fig = plt.figure(figsize=(15, 5))
        gs  = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
        ax1, ax2, ax3, cax = [fig.add_subplot(gs[i]) for i in range(4)]

        im = ax1.imshow(Ks, cmap="hot", aspect="auto")
        ax1.set_title("Empirical Kernel\n(sorted by class)", fontsize=14)
        ax2.imshow(Ki_s, cmap="hot", vmin=0, vmax=1, aspect="auto")
        ax2.set_title("Ideal Kernel", fontsize=14)
        ax3.imshow(ov_s, cmap="RdBu_r", aspect="auto")
        ax3.set_title(f"Centered Kernel Product\nCKA={score:.4f}", fontsize=14)
        plt.colorbar(im, cax=cax)
        
        if dataset_name.lower() == "abalone" or np.isnan(glob['sep']):
            title_text = f"[{key.upper()} — {tag}]  Global CKA = {score:.4f} (Continuous Target Kernel)"
        else:
            title_text = (f"[{key.upper()} — {tag}]  CKA={score:.4f}  "
                        f"Sep={glob['sep']:.3f}  ({heuristic_sep_label(glob['sep'])})")

        fig.suptitle(title_text, fontsize=16, fontweight="bold", y=1.05)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = os.path.join(out_dir, f"{key[0].lower()}_kernel_{dataset_name}_{tag}.png")
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"  {fname}")

# ─────────────────────────────────────────────────────────────────
#  2.  SEPARATION BAR CHARTS
# ─────────────────────────────────────────────────────────────────

def plot_individual_separation(key, r, out_dir, dataset_name):
    classes = sorted(r["pc_lin"].keys())
    x, w    = np.arange(len(classes)), 0.35
    lin_v   = np.array([r["pc_lin"][c]["sep"] for c in classes])
    rbf_v   = np.array([r["pc_rbf"][c]["sep"] for c in classes])

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.style.use("seaborn-v0_8-whitegrid")
    ax.bar(x - w/2, lin_v, w, label="Linear Sep (Cohen's d)", color="#4C72B0", edgecolor="black", alpha=0.85)
    ax.bar(x + w/2, rbf_v, w, label="RBF Sep (Cohen's d)",    color="#DD8452", edgecolor="black", alpha=0.85)
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(classes)
    ax.set_xlabel("Class", fontsize=15, fontweight="bold")
    ax.set_ylabel("Separation Metric", fontsize=15, fontweight="bold")

    acc_map = r.get("acc")
    if acc_map:
        acc_v = np.array([acc_map.get(c, np.nan) for c in classes])
        ax2   = ax.twinx()
        ax2.plot(x, acc_v, color="#55A868", marker="D", markersize=7, lw=2.5, label="Accuracy (%)")
        valid = acc_v[~np.isnan(acc_v)]
        if len(valid): ax2.set_ylim(max(0, valid.min() - 10), 103)
        ax2.set_ylabel("Accuracy (%)", fontsize=15, fontweight="bold")
        b, lb = ax.get_legend_handles_labels()
        a, la = ax2.get_legend_handles_labels()
        ax.legend(b + a, lb + la, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize=15)
    else:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=15)

    ax.set_title(f"{key.upper()} Embedding: Per-Class Separation & Accuracy\n"
                f"Global CKA — Lin:{r['s_lin']:.4f}  RBF:{r['s_rbf']:.4f}", fontsize=15)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{key[0].lower()}_separation_individual_{dataset_name}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  → {fname}")


def plot_separation_comparison(results, dataset_name, out_dir):
    embeddings = list(results.keys())
    if not embeddings: return
    fig = plt.figure(figsize=(9 * len(embeddings), 7))
    gs  = gridspec.GridSpec(1, len(embeddings), wspace=0.35)
    plt.style.use("seaborn-v0_8-whitegrid")
    for col, key in enumerate(embeddings):
        r       = results[key]
        classes = sorted(r["pc_lin"].keys())
        x, w    = np.arange(len(classes)), 0.35
        lin_v   = np.array([r["pc_lin"][c]["sep"] for c in classes])
        rbf_v   = np.array([r["pc_rbf"][c]["sep"] for c in classes])
        ax = fig.add_subplot(gs[col])
        ax.bar(x - w/2, lin_v, w, label="Linear Sep", color="#4C72B0", edgecolor="black", alpha=0.85)
        ax.bar(x + w/2, rbf_v, w, label="RBF Sep",    color="#DD8452", edgecolor="black", alpha=0.85)
        ax.axhline(0, color="black", lw=1, ls="--", alpha=0.7)
        ax.set_xticks(x); ax.set_xticklabels(classes)
        ax.set_xlabel("Class", fontsize=15, fontweight="bold")
        ax.set_ylabel("Separation Metric", fontsize=15, fontweight="bold")
        ax.set_title(f"{key.upper()} Embedding\nGlobal CKA — Lin:{r['s_lin']:.4f}  RBF:{r['s_rbf']:.4f}", fontsize=15)
        acc_map = r.get("acc")
        if acc_map:
            acc_v = np.array([acc_map.get(c, np.nan) for c in classes])
            ax2   = ax.twinx()
            ax2.plot(x, acc_v, color="#55A868", marker="D", markersize=7, lw=2.5, label="Accuracy (%)")
            valid = acc_v[~np.isnan(acc_v)]
            if len(valid): ax2.set_ylim(max(0, valid.min() - 10), 103)
            ax2.set_ylabel("Accuracy (%)", fontsize=15, fontweight="bold")
            b, lb = ax.get_legend_handles_labels()
            a, la = ax2.get_legend_handles_labels()
            ax.legend(b + a, lb + la, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=15)
        else:
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=15)
    fig.suptitle(f"PELM: Kernel Separation vs Accuracy ({dataset_name.upper()})",
                fontsize=15, fontweight="bold", y=1.03)
    plt.tight_layout()
    fname = os.path.join(out_dir, "separation_embedding_comparison.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"\n  → {fname}")

# ─────────────────────────────────────────────────────────────────
#  3.  DISTANCE PRESERVATION  (uses H_readout = Eq.9+11)
# ─────────────────────────────────────────────────────────────────

def run_distance_preservation(datasets_H, out_dir):
    targets = [d for d in datasets_H if d.get("X") is not None]
    if not targets: return

    n_plots = len(targets) + 1
    n_cols  = 2
    n_rows  = (n_plots + 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 6 * n_rows))
    axes = np.array(axes).flatten()
    ax_comb = axes[len(targets)]

    COLORS  = ["dodgerblue", "crimson", "mediumseagreen", "darkorange"]
    MARKERS = ["o", "s", "^", "D"]

    print(f"\n{'─'*60}")
    print(f"  PAIRWISE GEOMETRIC CORRELATION  ({DIST_SAMPLES} samples, Eq.9+11 repr.)")
    print(f"{'─'*60}")

    for i, d in enumerate(targets):
        ax   = axes[i]
        name = d["name"]
        H    = d["H_readout"][:DIST_SAMPLES]   # BUGFIX A2: use readout repr
        X    = d["X"]

        dx = pdist(X, "sqeuclidean")
        dh = pdist(H, "sqeuclidean")

        def z_score(v): return (v - v.mean()) / (v.std() + 1e-12)
        dx_n, dh_n = z_score(dx), z_score(dh)

        rp, _ = pearsonr(dx_n, dh_n)
        rs, _ = spearmanr(dx_n, dh_n)
        print(f"  {name:<24}  Pearson r = {rp:.3f}  |  Spearman ρ = {rs:.3f}")

        c, m = COLORS[i % 4], MARKERS[i % 4]
        ax.scatter(dx_n, dh_n, color=c, marker=m, alpha=0.5, s=30, edgecolors="none")
        ax.plot([-3, 3], [-3, 3], "k--", alpha=0.7, label="Ideal 1:1")
        ax.set_title(f"{name} Embedding\nPearson r={rp:.3f}  Spearman ρ={rs:.3f}",
                    fontsize=15, fontweight="bold")
        ax.set_xlabel("Original Space Dist ||x-y||² (Z-score)", fontsize=15)
        ax.set_ylabel("Optical Space Dist ||r(x)-r(y)||² (Z-score)", fontsize=15)
        ax.legend(fontsize=15); ax.grid(True, ls=":", alpha=0.6)
        ax_comb.scatter(dx_n, dh_n, color=c, marker=m, alpha=0.5, s=30,
                        edgecolors="none", label=f"{name} (r={rp:.2f}, ρ={rs:.2f})")

    ax_comb.plot([-3, 3], [-3, 3], "k--", alpha=0.7, label="Ideal 1:1")
    ax_comb.set_title("All Embeddings Combined", fontsize=15, fontweight="bold")
    ax_comb.set_xlabel("Original Space Dist ||x-y||² (Z-score)", fontsize=15)
    ax_comb.set_ylabel("Optical Space Dist ||r(x)-r(y)||² (Z-score)", fontsize=15)
    ax_comb.legend(fontsize=15, loc="upper left"); ax_comb.grid(True, ls=":", alpha=0.6)

    for j in range(len(targets) + 1, len(axes)):
        axes[j].axis("off")

    fig.suptitle("Relative Distance Structure Preservation: Original vs PELM Space "
                "(Eq.9+11 representation)", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()
    fname = os.path.join(out_dir, "distance_preservation_2x2.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"  → {fname}")

# ─────────────────────────────────────────────────────────────────
#  4.  t-SNE  
# ─────────────────────────────────────────────────────────────────

def run_tsne(key, H_readout, y, out_dir, dataset_name):
    """BUGFIX A3: dataset_name passed explicitly."""
    target_n  = TSNE_TARGET_N.get(dataset_name, 2000)
    n_cl      = len(np.unique(y))
    per_class = max(1, target_n // n_cl)
    # BUGFIX A3: pass dataset_name explicitly
    H_readout, y = balanced_subset(H_readout, y, per_class, dataset_name=dataset_name)
    n = len(H_readout)
    print(f"  t-SNE [{key}]  Target N={target_n}, Actual N={n}  classes={n_cl}  …")
    perp = max(5, min(30, (n // n_cl) // 2))
    tsne = TSNE(n_components=2, perplexity=perp, random_state=42, n_iter=1000,
                init="pca", learning_rate="auto", method="barnes_hut")
    H2   = tsne.fit_transform(H_readout)
    cmap = plt.get_cmap("tab10" if n_cl <= 10 else "tab20")
    fig, ax = plt.subplots(figsize=(9.5, 7))
    classes        = sorted(np.unique(y))
    class_to_idx   = {c: i for i, c in enumerate(classes)}
    for c in classes:
        m     = (y == c)
        c_idx = class_to_idx[c]
        ax.scatter(H2[m, 0], H2[m, 1], label=f"Class {c}",
                color=cmap(c_idx / max(n_cl - 1, 1)), s=12, alpha=0.75)
    ax.set_title(f"t-SNE — {key.upper()}  (N={n}, {n_cl} classes)",
                fontsize=15, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1", fontsize=15)
    ax.set_ylabel("t-SNE dim 2", fontsize=15)
    ax.legend(markerscale=2, fontsize=12, title="Class", ncol=1,
            bbox_to_anchor=(1.04, 1), loc="upper left", framealpha=1.0)
    plt.tight_layout()
    fname = os.path.join(out_dir, f"{key[0].lower()}_tsne_{dataset_name}.png")
    plt.savefig(fname, dpi=300, bbox_inches="tight"); plt.close()
    print(f"   {fname}")

# ─────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Unified Reservoir Analysis")
    parser.add_argument("--dataset", type=str,
                        choices=["mnist", "fsdd", "abalone", "mushroom"],
                        default=TARGET_DATASET)
    args = parser.parse_args()
    args.dataset = args.dataset.strip().lower()

    out_dir = os.path.join("analysis_results", args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(f"  PELM — Unified Reservoir Analysis (Dataset: {args.dataset.upper()})")
    print("=" * 60)

    npz_paths = get_npz_paths(args.dataset)

    #   H_readout → Eq.9+11       (for distance preservation, t-SNE)
    #   H_kernel  → Eq.9+10+11   (for CKA / kernel diagnostics)
    embeddings_data = {}
    for emb, path in npz_paths.items():
        if not path or not os.path.exists(path):
            print(f"  [skip] {emb.upper()}: file not found — {path}")
            continue
        H_raw, y, is_raw = load_npz(emb.upper(), path)
        embeddings_data[emb] = {
            "H_readout": preprocess_readout(H_raw, is_raw),   # Eq.9+11
            "H_kernel":  preprocess_kernel(H_raw, is_raw),    # Eq.9+10+11
            "y": y,
        }

    if not embeddings_data:
        print(f"\n[ERROR] No valid NPZ files found for '{args.dataset}'."); return

    print("\n" + "=" * 60 + "\n  CKA ANALYSIS (Eq.9→10→11 representation)\n" + "=" * 60)
    cka_results = {}
    for emb, d in embeddings_data.items():
        acc = KNOWN_ACCURACY_ALL.get(emb, {}).get(args.dataset)
        # BUGFIX A1+A3: pass H_kernel and dataset_name explicitly
        cka_results[emb] = run_cka_analysis(
            emb, d["H_kernel"].copy(), d["y"].copy(), acc, out_dir, args.dataset)

    print("\n" + "=" * 60 + "\n  t-SNE (Eq.9+11 representation)\n" + "=" * 60)
    for emb, d in embeddings_data.items():
        # BUGFIX A3: pass dataset_name
        run_tsne(emb, d["H_readout"].copy(), d["y"].copy(), out_dir, args.dataset)

    print("\n" + "=" * 60 + "\n  SEPARATION BAR CHARTS\n" + "=" * 60)
    for emb, res in cka_results.items():
        plot_individual_separation(emb, res, out_dir, args.dataset)
    plot_separation_comparison(cka_results, args.dataset, out_dir)

    print("\n" + "=" * 60 + "\n  PAIRWISE GEOMETRIC CORRELATION\n" + "=" * 60)
    loader_spec = DATA_LOADERS.get(args.dataset)
    loader_fn   = load_raw_loader(loader_spec)
    X_raw = None
    if loader_fn:
        try:
            (X_tr, _), _ = loader_fn()
            X_raw = X_tr[:DIST_SAMPLES].astype(np.float64).reshape(DIST_SAMPLES, -1)
        except Exception as e:
            print(f"  [warn] Could not load original data: {e}")

    if X_raw is not None:
        dist_input = []
        for emb, d in embeddings_data.items():
            dist_input.append({
                "name": emb.upper(),
                "H_readout": d["H_readout"],   
                "X": X_raw,
            })
        run_distance_preservation(dist_input, out_dir)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  {'Embedding':<12} | {'CKA Lin':>9} | {'CKA RBF':>9} | {'CKA K2':>9} | {'Sep RBF':>9}")
    print(f"  {'─'*65}")
    for emb, r in cka_results.items():
        print(f"  {emb.upper():<12} | {r['s_lin']:>9.4f} | {r['s_rbf']:>9.4f} | "
            f"{r['s_k2']:>9.4f} | {r['g_rbf']['sep']:>9.4f}  {heuristic_sep_label(r['g_rbf']['sep'])}")
    print(f"\n  All figures saved to: {os.path.abspath(out_dir)}/")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
