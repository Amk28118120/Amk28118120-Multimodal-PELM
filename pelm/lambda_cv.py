"""
lambda_cv.py — Parallel Cross-validated lambda selection for PELM
==========================================================================
Workflow:
  1. Set TARGET_DATASET at the top.
  2. Script automatically spawns parallel processes for drf, noise, fourier.
  3. K-fold CV over a lambda grid → pick best lambda per embedding.
  4. Retrain on ALL training data with best lambda.
  5. Evaluate ONCE on test.npz and generate Confusion Matrix (for classification).
  6. Saves cleanly to results/<dataset>/<embedding>/
"""

import os

# ──────────────────────────────────────────────────────────────────────────────
# CRITICAL FIX: Stop Numpy from CPU thrashing during multiprocessing
# ──────────────────────────────────────────────────────────────────────────────
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe for multiprocessing
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import concurrent.futures

# ──────────────────────────────────────────────────────────────────────────────
#  MASTER TOGGLE
# ──────────────────────────────────────────────────────────────────────────────
TARGET_DATASET = "abalone"  # Options: "mnist", "fsdd", "mushroom", "abalone"
NPZ_DIR        = "npz_files"
BASE_OUT_DIR   = "results"

FOLDS = 5
LAMBDA_GRID = np.logspace(-5, 1, 30)

EMBEDDINGS = ["drf", "noise", "fourier"]

TASK_MAP = {
    "mnist": "classification",
    "fsdd": "classification",
    "mushroom": "classification",
    "abalone": "regression"
}

# ══════════════════════════════════════════════════════════════════════════════
# IO & Ridge Helpers
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(path: str):
    data = np.load(path, allow_pickle=False)
    keys = list(data.keys())

    if "H_train" in keys:
        H = data["H_train"].astype(np.float64)
        y = data["y_train"]
        last = int(data["last_idx"]) if "last_idx" in keys else len(H)
        if last < len(H):
            H, y = H[:last], y[:last]
    elif "H_test" in keys:
        H = data["H_test"].astype(np.float64)
        y = data["y_test"]
    else:
        raise KeyError(f"Cannot find H_train/H_test in {path}")

    # CRITICAL FIX: Center and normalize BOTH Train and Test consistently.
    H -= H.mean(axis=1, keepdims=True)
    H /= (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    
    return H, y


def encode_labels(y_train, y_test=None):
    """Robust label encoding that handles non-contiguous classes."""
    classes = np.unique(y_train)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    y_tr_enc = np.array([class_to_idx[c] for c in y_train])
    
    if y_test is not None:
        y_te_enc = np.array([class_to_idx.get(c, -1) for c in y_test]) # -1 if unseen
        return y_tr_enc, y_te_enc, len(classes)
    
    return y_tr_enc, len(classes)


def one_hot(y, n_classes):
    Y = np.zeros((len(y), n_classes))
    Y[np.arange(len(y)), y.astype(int)] = 1.0
    return Y


def evaluate(H, y, beta, task):
    scores = H @ beta
    if task == "classification":
        preds = np.argmax(scores, axis=1)
        score = np.mean(preds == y.astype(int))
    else:
        preds = scores.flatten()
        score = -np.sqrt(np.mean((preds - y) ** 2)) # Returns negative RMSE internally
    return score, preds

# ══════════════════════════════════════════════════════════════════════════════
# Cross-validation
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(H, y, task, n_classes, folds, lambda_grid, emb_name, dataset_name):
    if task == "classification":
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    else:
        splitter = KFold(n_splits=folds, shuffle=True, random_state=42)

    fold_splits = list(splitter.split(H, y.astype(int)) if task == "classification" else splitter.split(H))
    scores = np.zeros((folds, len(lambda_grid)))

    # Compute target dynamic range for NRMSE mapping if regression task
    if task == "regression":
        y_range = 28.0 if dataset_name == "abalone" else (y.max() - y.min())

    for f, (tr_idx, val_idx) in enumerate(fold_splits):
        H_tr, H_val = H[tr_idx], H[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        Y_tr = one_hot(y_tr, n_classes) if task == "classification" else y_tr.reshape(-1, 1)

        # MASSIVE SPEEDUP: Precompute A and B once per fold
        A = H_tr.T @ H_tr
        B = H_tr.T @ Y_tr
        I = np.eye(H_tr.shape[1])

        for l_idx, lam in enumerate(lambda_grid):
            beta = np.linalg.solve(A + lam * I, B)
            score, _ = evaluate(H_val, y_val, beta, task)
            
            if task == "regression":
                # Convert internal negative RMSE to positive validation NRMSE
                rmse = -score
                scores[f, l_idx] = rmse / y_range
            else:
                scores[f, l_idx] = score
            
        print(f"  [{emb_name.upper():<12}] Fold {f+1}/{folds} completed...", flush=True)

    return scores.mean(axis=0), scores.std(axis=0)


def plot_cv_curve(lambda_grid, cv_mean, cv_std, best_lam, task, out_dir, tag):
    fig, ax = plt.subplots(figsize=(9, 5))
    
    if task == "classification":
        metric_label = "CV Accuracy"
        ylabel = "Accuracy (%)"
        y_vals = cv_mean * 100
        y_err  = cv_std * 100
        y_best = y_vals[np.argmax(cv_mean)]
        title_metric = f"Max Acc = {y_best:.2f}%"
        line_color = 'blue'
    else:
        metric_label = "CV NRMSE"
        ylabel = "Normalized RMSE (NRMSE)"
        y_vals = cv_mean
        y_err  = cv_std
        y_best = y_vals[np.argmin(cv_mean)] # Optimal lambda is where NRMSE is minimized
        title_metric = f"Min NRMSE = {y_best:.4f}"
        line_color = 'blue' 

    ax.semilogx(lambda_grid, y_vals, color=line_color, marker='o', markersize=5, linewidth=1.8, label=metric_label)
    ax.fill_between(lambda_grid, y_vals - y_err, y_vals + y_err, alpha=0.2, color=line_color, label='±1 std')
    ax.axvline(best_lam, color='red', linestyle='--', linewidth=1.8, label=f'Best λ = {best_lam:.2e}')
    ax.scatter([best_lam], [y_best], color='red', zorder=5, s=80)

    ax.set_xlabel('λ (regularization strength)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(f'{tag} — Lambda CV\nBest λ = {best_lam:.2e}  |  {title_metric}', fontsize=14, pad=12)
    
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, which='both', linestyle=':', alpha=0.6)
    plt.tight_layout()

    plt.savefig(os.path.join(out_dir, "lambda_cv_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# Worker Function
# ══════════════════════════════════════════════════════════════════════════════

def process_embedding(dataset, task, emb):
    train_path = os.path.join(NPZ_DIR, f"{dataset}_train_fold1_{emb}.npz")
    test_path  = os.path.join(NPZ_DIR, f"{dataset}_test_fold1_{emb}.npz")
    out_dir    = os.path.join(BASE_OUT_DIR, dataset, emb)

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        return emb, f"[SKIP] Missing files for {emb.upper()}"

    os.makedirs(out_dir, exist_ok=True)
    
    # 1. Load Data
    H_train_raw, y_train_raw = load_and_preprocess(train_path)
    H_test_raw, y_test_raw   = load_and_preprocess(test_path)
    
    # Robust Encoding
    if task == "classification":
        y_train, y_test, n_classes = encode_labels(y_train_raw, y_test_raw)
    else:
        y_train, y_test, n_classes = y_train_raw, y_test_raw, 1

    # 2. CV Tracked with custom metrics per task
    cv_mean, cv_std = cross_validate(H_train_raw, y_train, task, n_classes, FOLDS, LAMBDA_GRID, emb, dataset)
    
    if task == "classification":
        best_idx = int(np.argmax(cv_mean))
    else:
        best_idx = int(np.argmin(cv_mean)) # Min NRMSE is chosen for regression
        
    best_lam = LAMBDA_GRID[best_idx]
    
    plot_cv_curve(LAMBDA_GRID, cv_mean, cv_std, best_lam, task, out_dir, f"{dataset.upper()} {emb.upper()}")

    # 3. Retrain Full
    Y_train = one_hot(y_train, n_classes) if task == "classification" else y_train.reshape(-1, 1)
    
    A_full = H_train_raw.T @ H_train_raw
    B_full = H_train_raw.T @ Y_train
    beta = np.linalg.solve(A_full + best_lam * np.eye(H_train_raw.shape[1]), B_full)
    
    train_score, _ = evaluate(H_train_raw, y_train, beta, task)

    # 4. Evaluate Test
    test_score, test_preds = evaluate(H_test_raw, y_test, beta, task)

    # 5. Confusion Matrix (Classification Only)
    if task == "classification":
        cm = confusion_matrix(y_test, test_preds, labels=np.arange(n_classes))
        
        plt.rcParams.update({'font.size': 14})
        fig, ax = plt.subplots(figsize=(10, 10))
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap="Blues", values_format='d', ax=ax, colorbar=False, text_kw={'fontsize': 16})
        
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.2)
        cbar = fig.colorbar(disp.im_, cax=cax)
        cbar.ax.tick_params(labelsize=14)
        
        ax.set_title(f"{dataset.upper()} - {emb.upper()}\nTest Acc: {test_score*100:.2f}% | Best λ: {best_lam:.2e}", 
                        fontsize=15, pad=20)
        ax.set_xlabel("Predicted label", fontsize=15, labelpad=10)
        ax.set_ylabel("True label", fontsize=15, labelpad=10)
        ax.tick_params(axis='both', which='major', labelsize=14)
            
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=300, bbox_inches='tight')
        plt.close()
        plt.rcdefaults()

    # 6. Save Data
    np.save(os.path.join(out_dir, "beta.npy"), beta)
    
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(f"Dataset      : {dataset}\n")
        f.write(f"Embedding    : {emb}\n")
        f.write(f"Best lambda  : {best_lam:.6e}\n")
        if task == "classification":
            f.write(f"CV Acc       : {cv_mean[best_idx]*100:.2f}%\n")
            f.write(f"Train Acc    : {train_score*100:.2f}%\n")
            f.write(f"Test Acc     : {test_score*100:.2f}%\n")
        else:
            y_range = 28.0 if dataset == "abalone" else (y_test.max() - y_test.min())
            f.write(f"CV NRMSE     : {cv_mean[best_idx]:.4f}\n")
            f.write(f"Train RMSE   : {-train_score:.4f}\n")
            f.write(f"Test RMSE    : {-test_score:.4f}\n")
            f.write(f"Test NRMSE   : {-test_score / y_range:.4f}\n")

    res_str = f"  ✅ {emb.upper():<12} | Best λ: {best_lam:.4e} | "
    if task == "classification":
        res_str += f"Test Acc: {test_score*100:.2f}%"
    else:
        y_range = 28.0 if dataset == "abalone" else (y_test.max() - y_test.min())
        nrmse = -test_score / y_range
        res_str += f"Test RMSE: {-test_score:.4f} | NRMSE: {nrmse:.4f}"
        
    return emb, res_str

# ══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ══════════════════════════════════════════════════════════════════════════════

def main():
    task = TASK_MAP.get(TARGET_DATASET)
    if not task:
        print(f"Unknown dataset: {TARGET_DATASET}")
        return

    start_time = time.time()

    print("\n" + "="*70)
    print(f"  PELM Parallel Lambda CV & Test Evaluation")
    print(f"  Dataset : {TARGET_DATASET.upper()}")
    print(f"  Task    : {task.upper()}")
    print(f"  Folds   : {FOLDS}")
    print("="*70)
    print("  Dispatching parallel processes for: " + ", ".join(EMBEDDINGS) + "...\n")

    results = {}
    with concurrent.futures.ProcessPoolExecutor(max_workers=len(EMBEDDINGS)) as executor:
        futures = {executor.submit(process_embedding, TARGET_DATASET, task, emb): emb for emb in EMBEDDINGS}
        
        for future in concurrent.futures.as_completed(futures):
            emb = futures[future]
            try:
                emb_name, result_str = future.result()
                results[emb_name] = result_str
                print(f"\n{result_str}")
            except Exception as exc:
                print(f"\n  ❌ {emb.upper():<12} | [FAILED] {exc}")

    elapsed_time = time.time() - start_time
    mins, secs = divmod(elapsed_time, 60)

    print("\n" + "="*70)
    print(f"  ALL DONE! Outputs saved to: {BASE_OUT_DIR}/{TARGET_DATASET}/")
    print(f"  Total Execution Time: {int(mins)}m {int(secs)}s")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()