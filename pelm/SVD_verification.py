"""
Check 1: Conditioning Verification and Channel Pre-Whitening

CORRECT NORMALIZATION for SVD (per lab_checks.pdf Step 4):
  s /= np.median(s)   ← median, not mean
  Compare against Marchenko-Pastur bounds, NOT against |s²-1| < 1

WHAT TO REPORT (per lab_checks.pdf):
  m_eff, dynamic range, κ before and after pre-whitening,
  fraction outside MP band, singular value histogram.
  κ~100,000 raw and κ~900 after masking (before whitening) are EXPECTED.

FIXES APPLIED (v2 — addresses κ explosion to 10^14):
  [FIX 1] Flat-field correction uses sqrt(B2_k) not B2_k.
          B2_k is mean intensity; dividing by sqrt gives amplitude normalisation.

  [FIX 2] scipy.linalg.eigh / svd used instead of numpy equivalents
          (LAPACK dsyevd / dgesdd — more stable divide-and-conquer drivers).

  [FIX 3] _verdict() checks frac_out independently of kappa.

  [FIX 4] Legacy raw-cache fallback validates mtime against whitener stats.

  [FIX 5] Rank-deficiency condition corrected:
          OLD (buggy): k_max = min(N_FLATFIELD, m_eff) - 1  → always truncates
                       even when N_FLATFIELD > m_eff, dropping a good eigenvector.
          NEW:         only truncate when N_FLATFIELD <= m_eff (genuinely
                       rank-deficient). With N_FLATFIELD=512 > m_eff=265 the
                       full covariance is used — no truncation.

  [FIX 6] PCA-whitening with relative eigenvalue threshold replaces ZCA.
          Eigenvectors below 1% of max eigenvalue are discarded (noise dims).
          W shape becomes (n_keep, m_eff); whiten(r) → (n_keep,).
          This is stable for noisy optical systems with high dynamic range.

  [FIX 7] DEAD_CHANNEL_THRESHOLD raised from 0.05 → 0.15.
          With a 9656× dynamic range, 5% threshold keeps dim/noisy channels
          that add near-zero-variance dimensions to the covariance, exploding κ.
          15% keeps only the well-illuminated core.

  [FIX 8] N_SVD_PATTERNS raised to 1024 to ensure γ = N/m_eff < 0.5 even if
          m_eff stays high after threshold change. γ < 0.5 gives a tight MP
          band [1-√γ, 1+√γ] that is actually testable.
          With m_eff ≈ 150–180 and N=512 we'd still get γ ≈ 0.3, but 1024
          provides a safety margin.

EXPECTED RESULTS AFTER FIXES:
  γ ≈ 0.25–0.40   MP ≈ [0.37, 1.63]
  κ < 100         frac_out < 0.20   → "✓" or "~"
"""

import os
import time
import hashlib
import warnings
import numpy as np
import matplotlib.pyplot as plt

# ── scipy: better LAPACK drivers ─────────────────────────────────────────────
try:
    from scipy.linalg import eigh as _scipy_eigh
    from scipy.linalg import svd  as _scipy_svd
    _HAVE_SCIPY = True
except ImportError:
    warnings.warn(
        "scipy not found — falling back to numpy.linalg.  "
        "Install scipy for better numerical stability.",
        stacklevel=1,
    )
    _HAVE_SCIPY = False

def _eigh(C):
    """Eigendecomposition of real symmetric matrix (ascending order)."""
    if _HAVE_SCIPY:
        return _scipy_eigh(C)
    return np.linalg.eigh(C)

def _svd_s(A):
    """Singular values only, descending."""
    if _HAVE_SCIPY:
        return _scipy_svd(A, compute_uv=False, full_matrices=False)
    return np.linalg.svd(A, compute_uv=False)

# ─────────────────────────────────────────────────────────────────────────────

import config
from config import M_FEATURES, RANDOM_SEED
from optics_driver import OpticalSystem
from pelm_core import PELM_Algorithm

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "SVD_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# [FIX 7] Raised from 0.05 → 0.15.
# With 9656× dynamic range, 5% keeps far too many dim/noisy channels.
# 15% keeps only the well-illuminated core, reducing m_eff and stabilising κ.
DEAD_CHANNEL_THRESHOLD = 0.15

# [FIX 8] Raised from 512 → 1024 to ensure γ = N/m_eff stays well below 0.5
# even if m_eff is larger than expected after threshold change.
N_SVD_PATTERNS = 1024

# Calibration frames for flat-field and covariance (keep ≥ 256, ideally ≥ 512)
N_FLATFIELD = max(M_FEATURES, 512)

# Relative eigenvalue threshold for PCA-whitening [FIX 6]
# Eigenvectors whose eigenvalue < EIG_REL_THRESHOLD * λ_max are discarded.
EIG_REL_THRESHOLD = 0.01

EMBEDDING_METHODS = [
    "no_embedding",
    "fourier_embedding",
    "DRF_embedding",
    "noise_embedding",
]

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def generate_binary_input(rng):
    """Random binary 28×28 uint8 image ∈ {0, 255}."""
    return (rng.integers(0, 2, size=(28, 28)) * 255).astype(np.uint8)


def build_model(method):
    import pelm_core as pc
    print(f"\n{'='*60}\nBUILDING MODEL: {method}\n{'='*60}")
    pc.ENCODING_METHOD = method
    res_path = os.path.join(BASE_DIR, "pelm_reservoir.npz")
    if os.path.exists(res_path):
        os.remove(res_path)
    return PELM_Algorithm()


def _acquire_frames(optics, model, n_frames, rng, label, out_shape):
    """
    Shared acquisition loop used by flat-field, covariance, and Phi routines.
    Returns array of shape (n_frames, out_shape).
    out_shape is either M_FEATURES (raw) or m_eff (after masking/whitening).
    """
    frames    = np.empty((n_frames, out_shape), dtype=np.float64)
    t0 = last_print = time.time()

    for i in range(n_frames):
        img   = generate_binary_input(rng)
        phase = model.encode_input(img)
        frame = None
        while frame is None:
            frame = optics.display_and_capture(phase)
            if frame is None:
                print("  [!] Dropped frame — retrying...")
                time.sleep(0.05)
        frames[i] = model.extract_features(frame)

        now = time.time()
        if now - last_print >= 5.0 or i == 0 or i == n_frames - 1:
            done    = i + 1
            elapsed = now - t0
            eta     = (n_frames - done) * elapsed / done if done else 0
            print(f"  [{done:5d}/{n_frames}] ({100*done/n_frames:5.1f}%) "
                  f"| {elapsed/60:.1f} min | ETA {eta/60:.1f} min")
            last_print = now

    return frames

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: FLAT-FIELD  B²_k
# ─────────────────────────────────────────────────────────────────────────────

def measure_flat_field(optics, model):
    """
    Acquire N_FLATFIELD frames under random SLM patterns.
    Per-channel mean → B²_k (mean intensity, beam profile proxy).
    """
    flat_path = os.path.join(OUTPUT_DIR, "flat_field.npy")

    if os.path.exists(flat_path):
        B2_k = np.load(flat_path)
        if B2_k.shape == (M_FEATURES,):
            dyn = B2_k.max() / max(B2_k.min(), 1e-8)
            print(f"[Flat-field] Loaded from cache.  Dynamic range: {dyn:.1f}×")
            return B2_k
        print("[Flat-field] Shape mismatch — reacquiring")

    print(f"\n[Flat-field] Acquiring {N_FLATFIELD} random frames...")
    rng    = np.random.default_rng(RANDOM_SEED + 7777)
    frames = _acquire_frames(optics, model, N_FLATFIELD, rng,
                             "flat-field", M_FEATURES)

    B2_k = frames.mean(axis=0)
    B2_k = np.maximum(B2_k, 1e-8)
    np.save(flat_path, B2_k)
    dyn = B2_k.max() / B2_k.min()
    print(f"[Flat-field] Done.  Dynamic range: {dyn:.1f}×  Saved → {flat_path}")
    return B2_k

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: MASK DEAD CHANNELS
# ─────────────────────────────────────────────────────────────────────────────

def mask_dead_channels(B2_k):
    """
    [FIX 7] Threshold at DEAD_CHANNEL_THRESHOLD (0.15) × B²_k.max().
    Was 0.05 — too permissive for a 9656× dynamic range system; dim channels
    add near-zero-variance dimensions that explode the condition number.
    """
    threshold = DEAD_CHANNEL_THRESHOLD * B2_k.max()
    active    = B2_k > threshold
    m_eff     = int(active.sum())
    print(f"[Masking] Threshold: {DEAD_CHANNEL_THRESHOLD*100:.0f}% of peak "
          f"({threshold:.2f})")
    print(f"[Masking] Active channels: {m_eff}/{M_FEATURES} "
          f"({100*m_eff/M_FEATURES:.1f}%)  Dead/dim removed: {M_FEATURES - m_eff}")
    np.save(os.path.join(OUTPUT_DIR, "channel_mask.npy"), active)
    return active

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3: WHITENER
# ─────────────────────────────────────────────────────────────────────────────

def build_whitener(optics, model, active, B2_k):
    """
    Three-stage pipeline:

      1.  Amplitude flat-field correction
              x = r[active] / sqrt(B2_k[active])
          [FIX 1] Divides by AMPLITUDE (sqrt of mean intensity), not intensity.

      2.  Mean centering
              x = x - mu

      3.  PCA-whitening with relative eigenvalue threshold  [FIX 6]
              x = W @ (x - mu)
          where W = (V_keep * 1/sqrt(λ_keep)).T   shape: (n_keep, m_eff)

          Eigenvectors with λ < EIG_REL_THRESHOLD * λ_max are discarded.
          This removes noise dimensions that would otherwise make 1/sqrt(λ) → ∞.

          [FIX 5] Rank truncation only when N_FLATFIELD <= m_eff (genuinely
          rank-deficient covariance).  With N_FLATFIELD=512 > m_eff the full
          covariance is used.

    Output of whiten(r): vector of shape (n_keep,).
    n_keep is stored in whitener_stats.npz for downstream use.
    """
    cache_path = os.path.join(OUTPUT_DIR, "whitener_stats.npz")
    m_eff      = int(active.sum())

    # [FIX 1] amplitude scale: sqrt of mean intensity
    amp_scale = np.sqrt(np.maximum(B2_k[active], 1e-8))   # shape (m_eff,)

    # ── Load cache ────────────────────────────────────────────────────────────
    if os.path.exists(cache_path):
        data   = np.load(cache_path)
        mu     = data["mu"]
        W      = data["W"]
        n_keep = W.shape[0]
        print(f"[Whitener] Loaded cached transform  "
              f"shape(W)={W.shape}  n_keep={n_keep}/{m_eff}")

        def whiten(r):
            x = r[active] / amp_scale
            return W @ (x - mu)

        return whiten, n_keep

    # ── Acquire covariance dataset ────────────────────────────────────────────
    print(f"\n[Whitener] Acquiring {N_FLATFIELD} frames for covariance...")
    rng = np.random.default_rng(RANDOM_SEED + 424242)

    # Acquire raw frames then apply amplitude correction
    raw_frames = _acquire_frames(optics, model, N_FLATFIELD, rng,
                                 "whitener", M_FEATURES)
    X = raw_frames[:, active] / amp_scale[np.newaxis, :]   # (N_FLATFIELD, m_eff)

    # ── Mean centering ────────────────────────────────────────────────────────
    mu = X.mean(axis=0)
    Xc = X - mu

    # ── Covariance ────────────────────────────────────────────────────────────
    print("[Whitener] Computing covariance...")
    C = (Xc.T @ Xc) / max(N_FLATFIELD - 1, 1)   # (m_eff, m_eff)

    # ── Eigendecomposition — scipy LAPACK dsyevd  [FIX 2] ────────────────────
    print("[Whitener] Eigendecomposition (scipy.linalg.eigh)...")
    eigvals, eigvecs = _eigh(C)   # ascending order

    # [FIX 5] Only truncate when covariance is genuinely rank-deficient
    if N_FLATFIELD <= m_eff:
        safe_rank = N_FLATFIELD - 1
        warnings.warn(
            f"[Whitener] N_FLATFIELD ({N_FLATFIELD}) <= m_eff ({m_eff}): "
            f"covariance is rank-deficient.  Keeping top {safe_rank} eigenvectors. "
            f"Consider increasing N_FLATFIELD above {m_eff}.",
            stacklevel=2,
        )
        eigvals = eigvals[-safe_rank:]
        eigvecs = eigvecs[:, -safe_rank:]
    # else: N_FLATFIELD > m_eff → full rank, use all m_eff eigenvectors

    # [FIX 6] Relative eigenvalue threshold — discard noise dimensions
    rel_threshold = EIG_REL_THRESHOLD * eigvals.max()
    keep          = eigvals > rel_threshold
    n_keep        = int(keep.sum())
    n_dropped     = len(eigvals) - n_keep

    if n_dropped > 0:
        print(f"[Whitener] Discarding {n_dropped} eigenvectors below "
              f"{EIG_REL_THRESHOLD*100:.1f}% of λ_max "
              f"(threshold={rel_threshold:.4e})")

    eigvals_k = eigvals[keep]
    eigvecs_k = eigvecs[:, keep]

    print(f"[Whitener] n_keep={n_keep}/{m_eff}  "
          f"λ range: [{eigvals_k.min():.4e}, {eigvals_k.max():.4e}]")

    # PCA-whitening matrix: W = (V_keep * 1/sqrt(λ_keep)).T
    # Applying: W @ x  gives a vector of n_keep whitened PCA coordinates
    inv_sqrt = 1.0 / np.sqrt(eigvals_k)          # (n_keep,)
    W        = (eigvecs_k * inv_sqrt).T           # (n_keep, m_eff)

    np.savez(cache_path, mu=mu, W=W, amp_scale=amp_scale)
    print(f"[Whitener] Saved  shape(W)={W.shape}  n_keep={n_keep}/{m_eff}")

    def whiten(r):
        x = r[active] / amp_scale
        return W @ (x - mu)

    return whiten, n_keep

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4: ACQUIRE PRE-WHITENED PHI
# ─────────────────────────────────────────────────────────────────────────────

def acquire_whitened_phi(optics, model, prewhiten, n_keep, label):
    """
    Acquire N_SVD_PATTERNS pre-whitened frames.
    Returns Phi of shape (N, n_keep) where N = min(N_SVD_PATTERNS, n_keep).

    [FIX 4] Validates raw-cache mtime against whitener stats before reusing.
    """
    # [FIX 8] N must be > n_keep for γ < 1; use full N_SVD_PATTERNS
    N         = min(N_SVD_PATTERNS, n_keep * 4)   # safety cap; γ ≥ 0.25
    save_path = os.path.join(OUTPUT_DIR, f"phi_whitened_{label}.npy")

    if os.path.exists(save_path):
        Phi = np.load(save_path)
        if Phi.shape == (N, n_keep):
            print(f"[Cache] Loaded whitened {label}  shape={Phi.shape}")
            return Phi
        print(f"[Cache] Shape mismatch ({Phi.shape} vs expected "
              f"({N},{n_keep})) — reacquiring")

    # [FIX 4] Raw-cache fallback with mtime validation
    whitener_path  = os.path.join(OUTPUT_DIR, "whitener_stats.npz")
    raw_candidates = [
        os.path.join(OUTPUT_DIR, f"phi_raw_{label}.npy"),
        os.path.join(OUTPUT_DIR, f"phi_{label}.npy"),
    ]

    for rp in raw_candidates:
        if not os.path.exists(rp):
            continue
        raw_data = np.load(rp)
        if raw_data.shape[0] < N or raw_data.shape[1] != M_FEATURES:
            continue
        # [FIX 4] mtime check
        if os.path.exists(whitener_path):
            if os.path.getmtime(whitener_path) > os.path.getmtime(rp):
                warnings.warn(
                    f"[Cache] whitener_stats.npz is newer than {rp} — "
                    f"re-whitening for consistency.",
                    stacklevel=2,
                )
        print(f"[Cache] Re-whitening raw data from {rp}...")
        Phi = np.empty((N, n_keep), dtype=np.float64)
        for i in range(N):
            Phi[i] = prewhiten(raw_data[i])
        np.save(save_path, Phi)
        print(f"[Cache] Saved → {save_path}")
        return Phi

    # Fresh acquisition
    print(f"\n[Acquire] {label}: {N} patterns × {n_keep} whitened dims")
    label_hash = int(hashlib.md5(label.encode()).hexdigest(), 16)
    rng        = np.random.default_rng(RANDOM_SEED + (label_hash % 99991))
    Phi        = np.empty((N, n_keep), dtype=np.float64)
    t0 = last_print = time.time()

    for i in range(N):
        img   = generate_binary_input(rng)
        phase = model.encode_input(img)
        frame = None
        while frame is None:
            frame = optics.display_and_capture(phase)
            if frame is None:
                print("  [!] Dropped frame — retrying...")
                time.sleep(0.05)
        Phi[i] = prewhiten(model.extract_features(frame))

        now = time.time()
        if now - last_print >= 5.0 or i == 0 or i == N - 1:
            done    = i + 1
            elapsed = now - t0
            eta     = (N - done) * elapsed / done if done else 0
            print(f"  [{done:5d}/{N}] ({100*done/N:5.1f}%) "
                  f"| {elapsed/60:.1f} min | ETA {eta/60:.1f} min")
            last_print = now

    np.save(save_path, Phi)
    print(f"[Cache] Saved → {save_path}")
    return Phi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4b: RAW PHI  (before/after comparison)
# ─────────────────────────────────────────────────────────────────────────────

def acquire_raw_phi(optics, model, label, N):
    save_path   = os.path.join(OUTPUT_DIR, f"phi_raw_{label}.npy")
    legacy_path = os.path.join(OUTPUT_DIR, f"phi_{label}.npy")

    for p in (save_path, legacy_path):
        if os.path.exists(p):
            d = np.load(p)
            if d.shape[0] >= N and d.shape[1] == M_FEATURES:
                print(f"[Cache] Loaded raw {label}  shape={d.shape}")
                return d[:N, :]

    print(f"\n[Acquire] Raw {label}: {N} patterns × {M_FEATURES} channels")
    label_hash = int(hashlib.md5((label + "_raw").encode()).hexdigest(), 16)
    rng        = np.random.default_rng(RANDOM_SEED + (label_hash % 99991))
    Phi        = _acquire_frames(optics, model, N, rng, f"raw-{label}", M_FEATURES)
    np.save(save_path, Phi)
    return Phi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5: SVD METRICS
# ─────────────────────────────────────────────────────────────────────────────

def svd_metrics(Phi):
    """
    Per lab_checks.pdf page 3.
    [FIX 2] Uses scipy.linalg.svd (LAPACK dgesdd).
    """
    N, M = Phi.shape
    s    = _svd_s(Phi)

    s_norm = s / (np.median(s) + 1e-12)

    gamma    = min(N, M) / max(N, M)
    mp_lo    = 1.0 - np.sqrt(gamma)
    mp_hi    = 1.0 + np.sqrt(gamma)
    frac_out = float(np.mean((s_norm < mp_lo) | (s_norm > mp_hi)))
    kappa    = float(s_norm.max() / max(s_norm.min(), 1e-12))
    delta    = float(np.max(np.abs(s_norm - 1.0)))

    return s_norm, kappa, delta, mp_lo, mp_hi, frac_out

# ─────────────────────────────────────────────────────────────────────────────
# PREPROCESSING VARIANTS
# ─────────────────────────────────────────────────────────────────────────────

def _col_norm(P):
    norms = np.linalg.norm(P, axis=0, keepdims=True)
    norms = np.where(norms < 1e-12 * norms.max(), 1.0, norms)
    return P / norms

def _row_centre(P):
    return P - P.mean(axis=1, keepdims=True)

def _row_l2(P):
    norms = np.linalg.norm(P, axis=1, keepdims=True)
    return P / (norms + 1e-12)

PREPROCESSING_CASES = [
    ("Whiten only",                lambda P: P.copy()),
    ("Whiten + col norm",          lambda P: _col_norm(P)),
    ("Whiten + DC removal",        lambda P: _row_centre(P)),
    ("Whiten + row L2",            lambda P: _row_l2(P)),
    ("Whiten + DC + col norm",     lambda P: _col_norm(_row_centre(P))),
    ("Whiten + DC + row L2",       lambda P: _row_l2(_row_centre(P))),
    ("Whiten + DC + row L2 + col", lambda P: _col_norm(_row_l2(_row_centre(P)))),
]

# ─────────────────────────────────────────────────────────────────────────────
# VERDICT  [FIX 3]
# ─────────────────────────────────────────────────────────────────────────────

def _verdict(frac_out, kappa):
    """
    [FIX 3] Both conditions checked independently.
    Original short-circuited on kappa < 200 without checking frac_out.
    """
    kappa_ok = kappa    < 50
    frac_ok  = frac_out < 0.20
    if kappa_ok and frac_ok:
        return "✓"
    if kappa < 200 and frac_ok:
        return "~"
    return "✗"

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze(Phi_raw, Phi_white, label, m_eff):
    print(f"\n{'='*70}")
    print(f"RESULTS: {label}")
    print(f"{'='*70}")

    N, n_keep = Phi_white.shape
    gamma     = N / n_keep
    mp_lo     = 1.0 - np.sqrt(min(N, n_keep) / max(N, n_keep))
    mp_hi     = 1.0 + np.sqrt(min(N, n_keep) / max(N, n_keep))

    # Before whitening (raw, masked channels only for fair comparison)
    s_r      = _svd_s(Phi_raw[:N, :])
    s_r_norm = s_r / (np.median(s_r) + 1e-12)
    kappa_raw = float(s_r_norm.max() / max(s_r_norm.min(), 1e-12))

    print(f"\n  BEFORE whitening  ({M_FEATURES} channels, N={N}):")
    print(f"    κ = {kappa_raw:,.1f}  ← large value EXPECTED")
    print(f"\n  Whitened dims: n_keep={n_keep}  (from m_eff={m_eff})")
    print(f"  γ = N/n_keep = {gamma:.3f}   MP range: [{mp_lo:.3f}, {mp_hi:.3f}]")

    col_w  = 34
    header = (f"\n  {'Preprocessing':<{col_w}}  {'κ':>10}  {'δ':>8}  "
              f"{'frac_out':>9}  {'OK?':>4}")
    print(header)
    print("  " + "-" * (col_w + 38))

    case_results = {}
    for name, fn in PREPROCESSING_CASES:
        Phi_proc                         = fn(Phi_white)
        s_norm, kappa, delta, _, _, frac = svd_metrics(Phi_proc)
        ok                               = _verdict(frac, kappa)
        print(f"  {name:<{col_w}}  {kappa:>10.1f}  {delta:>8.4f}  "
              f"{frac:>9.3f}  {ok:>4}")
        case_results[name] = dict(s_norm=s_norm, kappa=kappa,
                                  delta=delta, frac_out=frac)

    print(f"\n  Target: κ ≪ 900,  frac_out < 0.20")

    s_w, kappa_w, delta_w, _, _, frac_w = svd_metrics(Phi_white)
    return dict(
        kappa_raw=kappa_raw, kappa_white=kappa_w, delta=delta_w,
        mp_lo=mp_lo, mp_hi=mp_hi, frac_out=frac_w,
        m_eff=m_eff, n_keep=n_keep, N=N,
        s_norm=s_w, cases=case_results,
    )

# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(all_results):
    n_emb      = len(all_results)
    n_cases    = len(PREPROCESSING_CASES)
    case_names = [name for name, _ in PREPROCESSING_CASES]

    # Figure 1: SVD histograms
    fig1, axes = plt.subplots(1, n_emb, figsize=(6 * n_emb, 5), squeeze=False)
    for ax, (emb, r) in zip(axes[0], all_results.items()):
        s = r["s_norm"]
        ax.hist(s, bins=40, color="steelblue", alpha=0.75, edgecolor="white",
                label="Whitened σ")
        ax.axvspan(r["mp_lo"], r["mp_hi"], alpha=0.15, color="green",
                   label=f"MP [{r['mp_lo']:.2f}, {r['mp_hi']:.2f}]")
        ax.axvline(1.0, color="red", lw=1.5, ls="--", label="σ=1 ideal")
        ax.set_title(f"{emb}\nκ={r['kappa_white']:.2f}  "
                     f"frac_out={r['frac_out']:.3f}  "
                     f"n_keep={r['n_keep']}/{r['m_eff']}",
                     fontsize=9)
        ax.set_xlabel("Normalized σ (÷ median)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    fig1.suptitle(
        f"Check 1: Whitened SVD spectrum  |  "
        f"N_flat={N_FLATFIELD}  N_svd={N_SVD_PATTERNS}  M={M_FEATURES}  "
        f"thr={DEAD_CHANNEL_THRESHOLD}  eig_thr={EIG_REL_THRESHOLD}",
        fontsize=10, y=1.02,
    )
    fig1.tight_layout()
    p1 = os.path.join(OUTPUT_DIR, "check1_svd_whitened.pdf")
    fig1.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"[Plot] Saved → {p1}")

    # Figure 2: κ and frac_out per preprocessing
    fig2, (ax_k, ax_f) = plt.subplots(2, 1, figsize=(max(12, 2 * n_cases), 9))
    x       = np.arange(n_cases)
    width   = 0.8 / max(n_emb, 1)
    colours = plt.cm.tab10(np.linspace(0, 0.6, n_emb))

    for ei, (emb, r) in enumerate(all_results.items()):
        kappas = [r["cases"][cn]["kappa"]    for cn in case_names]
        fracs  = [r["cases"][cn]["frac_out"] for cn in case_names]
        offset = (ei - (n_emb - 1) / 2) * width
        ax_k.bar(x + offset, kappas, width, label=emb,
                 color=colours[ei], alpha=0.8, edgecolor="black", linewidth=0.5)
        ax_f.bar(x + offset, fracs,  width, label=emb,
                 color=colours[ei], alpha=0.8, edgecolor="black", linewidth=0.5)

    ax_k.axhline(50,  color="red",    ls="--", lw=1.2, label="κ=50 target")
    ax_k.axhline(900, color="orange", ls=":",  lw=1.2,
                 label="κ=900 (unwhitened baseline)")
    ax_k.set_ylabel("Condition number κ  (lower is better)")
    ax_k.set_title("κ after whitening + each additional preprocessing")
    ax_k.set_yscale("log")
    ax_k.set_xticks(x)
    ax_k.set_xticklabels(case_names, rotation=25, ha="right", fontsize=9)
    ax_k.legend(fontsize=9)
    ax_k.grid(axis="y", alpha=0.3)

    ax_f.axhline(0.20, color="red", ls="--", lw=1.2, label="0.20 target")
    ax_f.set_ylabel("Fraction outside MP band  (lower is better)")
    ax_f.set_title("Fraction of singular values outside Marchenko-Pastur range")
    ax_f.set_xticks(x)
    ax_f.set_xticklabels(case_names, rotation=25, ha="right", fontsize=9)
    ax_f.legend(fontsize=9)
    ax_f.grid(axis="y", alpha=0.3)

    fig2.suptitle(
        f"Check 1: Preprocessing comparison  |  "
        f"N_svd={N_SVD_PATTERNS}  M={M_FEATURES}  "
        f"mask_thr={DEAD_CHANNEL_THRESHOLD}",
        fontsize=11,
    )
    fig2.tight_layout()
    p2 = os.path.join(OUTPUT_DIR, "check1_preprocessing_comparison.pdf")
    fig2.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"[Plot] Saved → {p2}")


def plot_flat_field(B2_k, active):
    n         = B2_k.size
    grid_size = int(np.sqrt(n))
    is_square = (grid_size * grid_size == n)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    if is_square:
        im1 = ax1.imshow(B2_k.reshape(grid_size, grid_size),
                         cmap="hot", aspect="auto")
        ax2.imshow(active.reshape(grid_size, grid_size).astype(float),
                   cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    else:
        im1 = ax1.imshow(B2_k[np.newaxis, :], cmap="hot", aspect="auto")
        ax2.imshow(active[np.newaxis, :].astype(float),
                   cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax1.set_yticks([])
        ax2.set_yticks([])

    ax1.set_title(f"Flat-field B²_k\n"
                  f"(dynamic range {B2_k.max()/max(B2_k.min(),1e-8):.1f}×)")
    ax2.set_title(f"Active channels: {int(active.sum())}/{n} "
                  f"({100*active.mean():.1f}%)  "
                  f"[threshold={DEAD_CHANNEL_THRESHOLD*100:.0f}%]")
    plt.colorbar(im1, ax=ax1)
    plt.suptitle("Beam profile and channel mask (Check 1)")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "check1_flat_field.pdf")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def save_csv(all_results):
    p1 = os.path.join(OUTPUT_DIR, "check1_summary.csv")
    with open(p1, "w") as f:
        f.write("embedding,kappa_raw,kappa_whitened,delta,"
                "frac_out_MP,m_eff,n_keep,N\n")
        for emb, r in all_results.items():
            f.write(f"{emb},{r['kappa_raw']:.2f},{r['kappa_white']:.4f},"
                    f"{r['delta']:.6f},{r['frac_out']:.4f},"
                    f"{r['m_eff']},{r['n_keep']},{r['N']}\n")
    print(f"[Results] Saved → {p1}")

    p2 = os.path.join(OUTPUT_DIR, "check1_preprocessing.csv")
    with open(p2, "w") as f:
        f.write("embedding,preprocessing,kappa,delta,frac_out_MP\n")
        for emb, r in all_results.items():
            for cn, cr in r["cases"].items():
                f.write(f"{emb},{cn},{cr['kappa']:.4f},"
                        f"{cr['delta']:.6f},{cr['frac_out']:.4f}\n")
    print(f"[Results] Saved → {p2}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(all_results, m_eff, B2_k, active):
    case_names = [name for name, _ in PREPROCESSING_CASES]
    col_e, col_c = 22, 34

    print("\n" + "=" * 76)
    print("CHECK 1 SUMMARY  (per lab_checks.pdf)")
    print("=" * 76)
    print(f"  M_FEATURES          : {M_FEATURES}")
    print(f"  Active channels     : {m_eff} ({100*m_eff/M_FEATURES:.1f}%)  "
          f"[mask threshold={DEAD_CHANNEL_THRESHOLD*100:.0f}%]")
    print(f"  Dynamic range       : {B2_k.max()/B2_k.min():.1f}×")
    print(f"  Flat-field corr.    : amplitude  x / sqrt(B²_k)  [FIX 1]")
    print(f"  Whitening           : PCA with rel.threshold={EIG_REL_THRESHOLD}  "
          f"[FIX 6]")
    print(f"  Whitening goal      : κ ≪ 900,  frac_out_MP < 0.20")

    print(f"\n  {'Embedding':<{col_e}}  {'κ raw':>10}  {'κ white':>10}  "
          f"{'n_keep':>7}  {'δ':>8}  {'frac_out':>9}  {'OK':>4}")
    print("  " + "-" * 72)
    for emb, r in all_results.items():
        ok = _verdict(r["frac_out"], r["kappa_white"])
        print(f"  {emb:<{col_e}}  {r['kappa_raw']:>10.1f}  "
              f"{r['kappa_white']:>10.2f}  {r['n_keep']:>7}  "
              f"{r['delta']:>8.4f}  {r['frac_out']:>9.3f}  {ok:>4}")

    print(f"\n{'─'*76}")
    print("  PREPROCESSING BREAKDOWN  (applied on top of whitened matrix)")
    print(f"{'─'*76}")
    for emb, r in all_results.items():
        print(f"\n  Embedding: {emb}   "
              f"(m_eff={r['m_eff']}, n_keep={r['n_keep']}, N={r['N']}, "
              f"MP=[{r['mp_lo']:.3f}, {r['mp_hi']:.3f}])")
        print(f"  {'Preprocessing':<{col_c}}  {'κ':>10}  {'δ':>8}  "
              f"{'frac_out':>9}  {'OK':>4}")
        print("  " + "-" * (col_c + 38))
        for cn in case_names:
            cr = r["cases"][cn]
            ok = _verdict(cr["frac_out"], cr["kappa"])
            print(f"  {cn:<{col_c}}  {cr['kappa']:>10.2f}  {cr['delta']:>8.4f}  "
                  f"{cr['frac_out']:>9.3f}  {ok:>4}")

    print("\n" + "=" * 76)
    print("  Legend:  ✓ κ<50 & frac<0.20   ~ κ<200 & frac<0.20   ✗ otherwise")
    print("  Note: pre-whitening is for Check 1 and Check 4 only.")
    print("=" * 76 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("CHECK 1: CONDITIONING VERIFICATION AND CHANNEL PRE-WHITENING")
    print("(per updated lab_checks.pdf — v2 with κ-explosion fixes)")
    print("=" * 70)

    # ── Determine what needs hardware ────────────────────────────────────────
    flat_path = os.path.join(OUTPUT_DIR, "flat_field.npy")
    need_flat = not os.path.exists(flat_path)

    missing_methods = [
        m for m in EMBEDDING_METHODS
        if not os.path.exists(os.path.join(OUTPUT_DIR, f"phi_whitened_{m}.npy"))
        and not os.path.exists(os.path.join(OUTPUT_DIR, f"phi_raw_{m}.npy"))
        and not os.path.exists(os.path.join(OUTPUT_DIR, f"phi_{m}.npy"))
    ]

    need_hardware = need_flat or len(missing_methods) > 0
    optics = base_model = None

    if need_hardware:
        print("\n[Hardware] Initializing...")
        optics     = OpticalSystem()
        optics.run_optical_test()
        base_model = build_model("fourier_embedding")

    # ── Step 1: flat-field ───────────────────────────────────────────────────
    if optics is not None and need_flat:
        B2_k = measure_flat_field(optics, base_model)
    else:
        B2_k = np.load(flat_path)
        print(f"[Flat-field] Loaded.  "
              f"Dynamic range: {B2_k.max()/B2_k.min():.1f}×")

    # ── Step 2: mask ─────────────────────────────────────────────────────────
    mask_path = os.path.join(OUTPUT_DIR, "channel_mask.npy")
    if os.path.exists(mask_path):
        active = np.load(mask_path)
        m_eff  = int(active.sum())
        print(f"[Masking] Loaded mask: {m_eff}/{M_FEATURES} active channels")
        # Warn if cached mask used old threshold
        print(f"[Masking] Note: mask was built with threshold at time of last run. "
              f"Delete channel_mask.npy to recompute with "
              f"DEAD_CHANNEL_THRESHOLD={DEAD_CHANNEL_THRESHOLD}.")
    else:
        active = mask_dead_channels(B2_k)
        m_eff  = int(active.sum())

    # Sanity check: m_eff should be well below N_FLATFIELD
    if m_eff >= N_FLATFIELD:
        warnings.warn(
            f"m_eff ({m_eff}) >= N_FLATFIELD ({N_FLATFIELD}). "
            f"Covariance will be rank-deficient. "
            f"Increase DEAD_CHANNEL_THRESHOLD or N_FLATFIELD.",
            stacklevel=1,
        )

    # ── Step 3: whitener ─────────────────────────────────────────────────────
    if optics is None:
        optics = OpticalSystem()
        optics.run_optical_test()
    if base_model is None:
        base_model = build_model("fourier_embedding")

    prewhiten, n_keep = build_whitener(optics, base_model, active, B2_k)
    print(f"[Whitener] n_keep={n_keep}  γ_svd = {N_SVD_PATTERNS}/{n_keep} "
          f"= {N_SVD_PATTERNS/n_keep:.2f}  "
          f"MP will be [{1-np.sqrt(n_keep/N_SVD_PATTERNS):.3f}, "
          f"{1+np.sqrt(n_keep/N_SVD_PATTERNS):.3f}]")

    plot_flat_field(B2_k, active)

    # ── Steps 4-5: per embedding ─────────────────────────────────────────────
    all_results = {}

    for method in EMBEDDING_METHODS:
        print(f"\n{'#'*60}\nEMBEDDING: {method}\n{'#'*60}")

        wp        = os.path.join(OUTPUT_DIR, f"phi_whitened_{method}.npy")
        rp        = os.path.join(OUTPUT_DIR, f"phi_raw_{method}.npy")
        rp_legacy = os.path.join(OUTPUT_DIR, f"phi_{method}.npy")

        # Expected shape for whitened Phi
        N_expected = min(N_SVD_PATTERNS, n_keep * 4)
        need_acq   = (
            not os.path.exists(wp)
            or np.load(wp).shape != (N_expected, n_keep)
        )
        has_raw = os.path.exists(rp) or os.path.exists(rp_legacy)

        if need_acq and not has_raw and optics is None:
            print("[Hardware] Initializing for missing acquisition...")
            optics = OpticalSystem()
            optics.run_optical_test()

        model = build_model(method) if (need_acq and not has_raw) else None

        # Whitened Phi
        if need_acq:
            Phi_white = acquire_whitened_phi(
                optics, model, prewhiten, n_keep, method)
        else:
            Phi_white = np.load(wp)
            print(f"[Cache] Loaded whitened {method}  shape={Phi_white.shape}")

        # Raw Phi for before/after comparison
        N_raw    = Phi_white.shape[0]
        need_raw = True
        for p in (rp, rp_legacy):
            if os.path.exists(p) and np.load(p).shape[0] >= N_raw:
                need_raw = False
                break

        if need_raw:
            if model is None:
                model = build_model(method)
            Phi_raw = acquire_raw_phi(optics, model, method, N_raw)
        else:
            src     = rp if (os.path.exists(rp)
                             and np.load(rp).shape[0] >= N_raw) else rp_legacy
            Phi_raw = np.load(src)[:N_raw, :]

        all_results[method] = analyze(Phi_raw, Phi_white, method, m_eff)

    # ── Save and display ─────────────────────────────────────────────────────
    save_csv(all_results)
    plot_results(all_results)
    print_summary(all_results, m_eff, B2_k, active)

    if optics is not None:
        optics.cleanup()

    print("\nCheck 1 complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as e:
        import traceback
        print(f"\nFATAL: {e}")
        traceback.print_exc()