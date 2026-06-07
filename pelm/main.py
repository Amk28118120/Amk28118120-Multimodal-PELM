"""
MNIST classification using a hardware-based photonic extreme learning machine.
Implementation follows Pierangeli et al. (2021).

Additions for 60,000-sample runs:
  1. Checkpoint every 1000 samples — resume on crash, no data lost
  2. std threshold corrected: 1.0 → 0.01
  3. CAM_EXPOSURE_US logged in results
"""

import numpy as np
import time
import os
import matplotlib.pyplot as plt
import cv2

from config import *
from data_loader.MNIST_data_loader import get_mnist
from data_loader.audio_data_loader import get_fsdd
from data_loader.abalone_data_loader import get_abalone
from optics_driver import OpticalSystem
from config import DATASET_TYPE
from pelm_core import PELM_Algorithm
from data_loader.Mushroom_data_loader import get_mushroom 
from data_loader.deepfake_loader import get_deepfake

CHECKPOINT_FILE = f"{DATASET_TYPE}_train_fold{Fold_ID}.npz"
TEST_CHECKPOINT_FILE = f"{DATASET_TYPE}_test_fold{Fold_ID}.npz"
CHECKPOINT_INTERVAL = CKPT_INT


# ============================================================
# CHECKPOINT HELPERS
# ============================================================

def save_checkpoint(H_train, y_train, last_idx):
    try:
        np.savez(CHECKPOINT_FILE, H_train=H_train,
                 y_train=y_train, last_idx=last_idx)
        print(f"[Checkpoint] Saved at sample {last_idx}")
    except Exception as e:
        print(f"[Checkpoint] WARNING: Save failed: {e}")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_FILE):
        return None
    try:
        data = np.load(CHECKPOINT_FILE, allow_pickle=False)
        H_train = data['H_train']
        y_train = data['y_train']
        last_idx = int(data['last_idx'])
        print(f"\n[Checkpoint] Found checkpoint at sample {last_idx}")
        if H_train.shape != (N_TRAIN, M_FEATURES):
            print(f"[Checkpoint] Shape mismatch — ignoring, starting fresh")
            return None
        return H_train, y_train, last_idx
    except Exception as e:
        print(f"[Checkpoint] Load failed ({e}) — starting fresh")
        return None


def save_test_checkpoint(H_test,y_test, last_idx):
    try:
        np.savez(TEST_CHECKPOINT_FILE, H_test=H_test, y_test=y_test, last_idx=last_idx)
        print(f"[Test Checkpoint] Saved at sample {last_idx}")
    except Exception as e:
        print(f"[Test Checkpoint] WARNING: Save failed: {e}")


def load_test_checkpoint():
    if not os.path.exists(TEST_CHECKPOINT_FILE):
        return None
    try:
        data = np.load(TEST_CHECKPOINT_FILE, allow_pickle=False)
        H_test = data["H_test"]
        y_test_saved = data["y_test"]
        last_idx = int(data["last_idx"])
        print(f"\n[Test Checkpoint] Found checkpoint at sample {last_idx}")
        if H_test.shape != (N_TEST, M_FEATURES):
            print("[Test Checkpoint] Shape mismatch — starting fresh")
            return None
        return H_test, y_test_saved, last_idx
    except Exception as e:
        print(f"[Test Checkpoint] Load failed ({e}) — starting fresh")
        return None


# ============================================================
# HARDWARE BREAK
# ============================================================

def hardware_break(optics, break_duration= 1 * 60):
    """Shutdown, cool, reinitialize. Camera re-arms via daemon on reinit."""
    print("\n" + "=" * 70)
    print("HARDWARE BREAK  (Cooling time)")
    print("=" * 70)

    try:
        optics.cleanup()
    except Exception as e:
        print(f"[WARN] cleanup failed: {e}")

    print(f"\nSleeping {break_duration/60:.0f} min...\n")
    time.sleep(break_duration)

    print("Reinitializing...")
    optics = OpticalSystem()
    optics.run_optical_test()
    print("Hardware break complete. Resuming...\n")
    return optics


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n" + "=" * 60)
    print("PHOTONIC EXTREME LEARNING MACHINE")
    print("Pierangeli et al., Photonics Research 2021")
    print("=" * 60)
    print(f"\n[Dataset] Using: {DATASET_TYPE.upper()}")
    
    # ========================================================
    # LOAD DATA (dataset-aware)
    # ========================================================
    
    if DATASET_TYPE == "mnist":
        (X_train, y_train), (X_test, y_test) = get_mnist()
        n_classes = 10

    elif DATASET_TYPE == "fsdd":
        (X_train, y_train), (X_test, y_test) = get_fsdd()
        n_classes = 10
        
    elif DATASET_TYPE == "deepfake":
        (X_train, y_train), (X_test, y_test) = get_deepfake()
        n_classes = 2  # Binary classification: Real vs Fake
    
    elif DATASET_TYPE == "mushroom":
        (X_train, y_train), (X_test, y_test) = get_mushroom()
        n_classes = 2  # e (edible) or p (poisonous)
        
    elif DATASET_TYPE == "abalone":
        (X_train, y_train), (X_test, y_test) = get_abalone()

        if TASK_TYPE == "classification":
            n_classes = int(y_train.max()) + 1
        else:
            n_classes = 1   # regression output       
    else:
        raise ValueError(f"Unknown DATASET_TYPE: {DATASET_TYPE}")
    
    # Validate dataset size
    total_available = len(X_train) + len(X_test)
    requested_total = N_TRAIN + N_TEST
    max_available = DATASET_SIZES.get(DATASET_TYPE, total_available)
    
    print(f"[Dataset] Available: {max_available} samples (train+test)")
    print(f"[Dataset] Requested: {N_TRAIN} train + {N_TEST} test = {requested_total}")
    print(f"[Dataset] Got: {len(X_train)} train + {len(X_test)} test = {total_available}")
    
    if requested_total > total_available:
        print(f"\n[WARNING] Requested {requested_total} but only {total_available} available!")
        print(f"[WARNING] Adjust N_TRAIN/N_TEST in config.py")
        ans = input("Continue anyway? (y/n): ").strip().lower()
        if ans != 'y':
            print("Aborting.")
            return
    
    if N_TRAIN / M_FEATURES < 0.5:
        print(f"\n[WARNING] Low N/M ratio: {N_TRAIN/M_FEATURES:.1f} (recommended ≥ 20)")
        print(f"[WARNING] Model may underfit. Consider increasing N_TRAIN or decreasing M_FEATURES")
    
    print()

    checkpoint = load_checkpoint()

    if checkpoint is not None:
        H_train, _, start_idx = checkpoint
        ans = input(f"[Resume] Continue from sample {start_idx}? (y/n): ").strip().lower()
        if ans != 'y':
            print("[Resume] Starting fresh")
            os.remove(CHECKPOINT_FILE)
            H_train = np.empty((N_TRAIN, M_FEATURES), dtype=np.float64)
            start_idx = 0
    else:
        H_train = np.empty((N_TRAIN, M_FEATURES), dtype=np.float64)
        start_idx = 0

    optics = OpticalSystem()
    model = PELM_Algorithm()

    model.visualize_embedding()
    optics.run_optical_test()
    model.get_statistics()

    encode = model.encode_input
    extract = model.extract_features
    capture = optics.display_and_capture

    # ========================================================
    # TRAINING PHASE
    # ========================================================

    print(f"\n[Training] Acquiring features for samples {start_idx}–{N_TRAIN}")

    experiment_start = time.time()
    work_duration = 60 * 60
    break_duration = 1 * 60

    total_break_time = 0.0
    next_break_time = work_duration

    last_print = time.time()
    skipped = 0
    _last_raw_mean  = 0.0
    _last_feat_mean = 0.0
    _last_feat_std  = 0.0
    _last_sat       = 0.0

    try:

        for i in range(start_idx, N_TRAIN):

            elapsed_work = time.time() - experiment_start - total_break_time

            if elapsed_work >= next_break_time:
                optics = hardware_break(optics, break_duration)
                capture = optics.display_and_capture
                total_break_time += break_duration
                next_break_time += work_duration

            now = time.time()

            if (now - last_print >= 10.0) or i == start_idx or i == N_TRAIN - 1:

                elapsed_work = now - experiment_start - total_break_time
                done = i - start_idx
                eta = ((N_TRAIN - i) * elapsed_work / done) if done > 0 else 0
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] "
                    f"  [{i:5d}/{N_TRAIN}] ({100*i/N_TRAIN:5.1f}%) | "
                    f"Work: {elapsed_work/60:.1f}min | "
                    f"ETA: {eta/60:.1f}min | Skipped: {skipped} | "
                    f"Raw(Mn:{_last_raw_mean:.1f} Sat:{_last_sat:.2f}%) | "
                    f"Bin(Mn:{_last_feat_mean:.2f} Std:{_last_feat_std:.2f})"
                )

                last_print = now

            phase_mask = encode(X_train[i])

            try:
                camera_frame = capture(phase_mask)
            except RuntimeError as e:

                print(f"\n[FATAL] Hardware error at sample {i}: {e}")
                print("[FATAL] Saving checkpoint...")
                save_checkpoint(H_train, y_train, i)
                raise

            if camera_frame is None:
                skipped += 1
                continue

            features = extract(camera_frame)
            H_train[i] = features
            _last_raw_mean  = float(np.mean(camera_frame))
            _last_feat_mean = float(np.mean(features))
            _last_feat_std  = float(np.std(features))
            _last_sat       = 100.0 * float(np.sum(camera_frame >= 254)) / camera_frame.size

            # if SAVE_EXAMPLE_IMAGES and i < NUM_EXAMPLES_TO_SAVE:

            #     model.save_phase_mask_example(X_train[i], idx=i)
            #     model.visualize_feature_extraction(camera_frame, H_train[i], idx=i)
            #     img = camera_frame
            #     if img.dtype != 'uint8':
            #         img = img.astype(float)
            #         img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            #         img = (img * 255).astype('uint8')

            #     cv2.imwrite(f"camera_frames/camera_frame_{i:05d}.png", img)
            if SAVE_EXAMPLE_IMAGES and i < NUM_EXAMPLES_TO_SAVE:
                model.save_phase_mask_example(X_train[i], idx=i)
                model.visualize_feature_extraction(camera_frame, H_train[i], idx=i)

                img_8bit = np.clip(
                    camera_frame.astype(np.float32) / 1023.0 * 255.0, 0, 255
                ).astype(np.uint8)

                cv2.imwrite(f"camera_frames/camera_frame_{i:05d}.png", img_8bit)


            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(H_train, y_train, i + 1)

    except KeyboardInterrupt:

        print(f"\n[Training] Interrupted — saving checkpoint...")
        save_checkpoint(H_train, y_train, i)
        raise

    training_time = time.time() - experiment_start - total_break_time

    print(f"\n[Training] Complete ({training_time/60:.1f} min work)")
    print(f"[Training]   Valid: {N_TRAIN - skipped}/{N_TRAIN}  Skipped: {skipped}")
    print(f"[Training]   Mean: {H_train.mean():.4f}  Std: {H_train.std():.4f}")

    # ========================================================
    # FEATURE QUALITY CHECK
    # ========================================================

    if H_train.max() == 0:
        print("\n[ERROR] All features zero — camera receiving no light")
        return

    if H_train.std() < 0.01:
        print(f"\n[ERROR] No feature variation (std={H_train.std():.6f})")
        return

    if H_train.mean() < 0.5:
        print(f"\n[WARNING] Low mean ({H_train.mean():.4f}) — "
                f"increase CAM_EXPOSURE_US (currently {CAM_EXPOSURE_US} us)")

    # H matrix visualization
    plt.figure(figsize=(12, 8))
    samples_to_show = min(500, N_TRAIN)
    vmax = np.percentile(H_train[H_train > 0], 95) if np.any(H_train > 0) else 1
    plt.imshow(H_train[:samples_to_show], aspect='auto',
                cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(label='Feature Value')
    plt.xlabel('Feature Index (M)')
    plt.ylabel('Sample Index (N)')
    plt.title(f'Feature Matrix H | Mean: {H_train.mean():.4f}  Std: {H_train.std():.4f}')
    plt.tight_layout()
    plt.savefig('H_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("[Training] Saved H_matrix.png")

    # ========================================================
    # RIDGE REGRESSION
    # ========================================================

    print(f"\n[Training] Training ridge regression readout ({n_classes} classes)...")

    t0 = time.time()

    n_train_used = N_TRAIN
    H = H_train[:n_train_used]
    H = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-8)

    if TASK_TYPE == "classification":
        Y = np.eye(n_classes)[y_train[:n_train_used]]
    else:
        Y = y_train[:n_train_used].astype(np.float64).reshape(-1, 1)

    model.beta = np.linalg.solve(
        H.T @ H + LAMBDA_REG * np.eye(M_FEATURES),
        H.T @ Y
    )

    regression_time = time.time() - t0

    print(f"[Training] Ridge regression complete ({regression_time:.1f}s)")

    try:
        np.savez("training_matrices.npz",
                    beta=model.beta)
        print("[Training] Saved beta")
    except Exception as e:
        print(f"[Training] Save beta failed: {e}")

    optics = hardware_break(optics, break_duration)
    capture = optics.display_and_capture

    # ========================================================
    # TESTING PHASE
    # ========================================================

    print(f"\n[Testing] Running on {N_TEST} samples")

    test_checkpoint = load_test_checkpoint()

    if test_checkpoint is not None:
        H_test, _, test_start_idx = test_checkpoint
        ans = input(f"[Resume Test] Continue from sample {test_start_idx}? (y/n): ").strip().lower()
        if ans != 'y':
            print("[Test Resume] Starting fresh")
            os.remove(TEST_CHECKPOINT_FILE)
            H_test = np.empty((N_TEST, M_FEATURES))
            test_start_idx = 0
    else:
        H_test = np.empty((N_TEST, M_FEATURES))
        test_start_idx = 0

    test_start = time.time()
    test_skipped = 0
    correct_so_far = 0
    error_sum = 0.0
    _last_raw_mean  = 0.0
    _last_feat_mean = 0.0
    _last_feat_std  = 0.0
    _last_sat       = 0.0

    total_break_time = 0.0
    next_break_time = work_duration
    last_print = time.time()

    try:

        for i in range(test_start_idx, N_TEST):

            elapsed_work = time.time() - test_start - total_break_time

            if elapsed_work >= next_break_time:

                optics = hardware_break(optics, break_duration)
                capture = optics.display_and_capture

                total_break_time += break_duration
                next_break_time += work_duration

            now = time.time()
            if (now - last_print >= 10.0) or i == test_start_idx or i == N_TEST - 1:
                elapsed_work = now - test_start - total_break_time
                done = i - test_start_idx
                eta = ((N_TEST - i) * elapsed_work / done) if done > 0 else 0
               # running_acc = (100.0 * correct_so_far / done) if done > 0 else 0.0
                if TASK_TYPE == "classification":
                    running_metric = (100.0 * correct_so_far / done) if done > 0 else 0.0
                else:
                    running_metric = np.sqrt(error_sum / done) if done > 0 else 0.0
                timestamp = time.strftime("%H:%M:%S")
                print(
                    f"[{timestamp}] "
                    f"  [{i:5d}/{N_TEST}] ({100*i/N_TEST:5.1f}%) | "
                    f"Work: {elapsed_work/60:.1f}min | "
                    f"ETA: {eta/60:.1f}min | "
                    f"{'Acc' if TASK_TYPE=='classification' else 'RMSE'}: {running_metric:.3f} |  Skipped: {test_skipped} | "
                    f"Raw(Mn:{_last_raw_mean:.1f} Sat:{_last_sat:.2f}%) | "
                    f"Bin(Mn:{_last_feat_mean:.2f} Std:{_last_feat_std:.2f})"
                )
                last_print = now

            phase_mask = encode(X_test[i])

            try:
                camera_frame = capture(phase_mask)
            except RuntimeError as e:
                print(f"\n[FATAL] Camera failed at test {i}: {e}")
                save_test_checkpoint(H_test, y_test, i)
                raise

            if camera_frame is None:
                test_skipped += 1
                continue

            features = extract(camera_frame)
            
            H_test[i] = features
            features = features / (np.linalg.norm(features) + 1e-8)
            _last_raw_mean  = float(np.mean(camera_frame))
            _last_feat_mean = float(np.mean(features))
            _last_feat_std  = float(np.std(features))
            _last_sat       = 100.0 * float(np.sum(camera_frame >= 254)) / camera_frame.size

            output = features @ model.beta

            if TASK_TYPE == "classification":
                pred = int(np.argmax(output))
                if pred == y_test[i]:
                    correct_so_far += 1
                    
            else:
                pred = output.item()
                error = (pred - y_test[i]) ** 2
                error_sum += error

            if (i + 1) % CHECKPOINT_INTERVAL == 0:
                save_test_checkpoint(H_test, y_test, i+1)

    except KeyboardInterrupt:
        print("\n[Testing] Interrupted — saving checkpoint...")
        save_test_checkpoint(H_test,y_test, i)
        raise

    test_time = time.time() - test_start
    H_test_norm = H_test / (np.linalg.norm(H_test, axis=1, keepdims=True) + 1e-8)

    scores = H_test_norm @ model.beta
    #scores = H_test @ model.beta

    if TASK_TYPE == "classification":
        predictions = np.argmax(scores, axis=1)
    else:
        predictions = scores.flatten()

    if TASK_TYPE == "classification":
        accuracy = 100.0 * np.sum(predictions == y_test) / N_TEST
    else:
        mse = np.mean((predictions - y_test[:N_TEST]) ** 2)
        rmse = np.sqrt(mse)

    total_time = training_time + regression_time + test_time

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    if TASK_TYPE == "classification":
        print(f"  Test accuracy:   {accuracy:.2f}%")
    else:
        rmse = np.sqrt(np.mean((predictions - y_test[:N_TEST]) ** 2))

        y_range = y_test.max() - y_test.min()
        nrmse = rmse / 28.0

        print(f"  RMSE:            {rmse:.4f}")
        print(f"  NRMSE:           {nrmse:.4f}")
    if DATASET_TYPE == "mnist":
        print(f"  Paper target:    92.18%  (N=60,000, M=4096)")
    elif DATASET_TYPE == "abalone":
        print(f"paper target rmse ≈ 2.1–2.3,nrmse ≈ 0.07–0.08")
    print(f"  Training time:   {training_time/60:.1f} min  (work only)")
    print(f"  Regression:      {regression_time:.1f} s")
    print(f"  Testing time:    {test_time/60:.1f} min")
    print(f"  Total:           {total_time/60:.1f} min")
    print(f"  Train skipped:   {skipped}/{N_TRAIN}")
    print(f"  Test skipped:    {test_skipped}/{N_TEST}")
    print(f"  Exposure:        {CAM_EXPOSURE_US} us")

    print("=" * 60 + "\n")

    # ========================================================
    # # Confusion matrix
    # ========================================================
    if TASK_TYPE == "classification":
        try:
            from sklearn.metrics import confusion_matrix, classification_report
            cm = confusion_matrix(y_test, predictions)
            fig, ax = plt.subplots(figsize=(12, 10))
            im = ax.imshow(cm, cmap='Blues', aspect='auto')
            ax.set_xticks(np.arange(n_classes))
            ax.set_yticks(np.arange(n_classes))
            ax.set_xticklabels(np.arange(n_classes))
            ax.set_yticklabels(np.arange(n_classes))
            ax.set_xlabel('Predicted Label', fontsize=12)
            ax.set_ylabel('True Label', fontsize=12)
            ax.set_title(f'Confusion Matrix — Accuracy: {accuracy:.2f}%', fontsize=14)
            for r in range(n_classes):
                for c in range(n_classes):
                    color = "white" if cm[r, c] > cm.max() / 2 else "black"
                    ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                            color=color, fontsize=8)
            plt.colorbar(im, ax=ax, label='Count')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("[Results] Saved confusion_matrix.png")

            print("\nPer-class accuracy:")
            for cls in range(n_classes):
                mask = (y_test == cls)
                total_d = np.sum(mask)
                if total_d == 0:
                    continue
                correct = np.sum(predictions[mask] == cls)
                print(f"  Class {cls}: {100.*correct/total_d:.1f}%  ({correct}/{total_d})")

            print("\nClassification Report:")
            print(classification_report(y_test, predictions,
                                        target_names=[f'Class {i}' for i in range(n_classes)]))
        except ImportError:
            print("[Results] sklearn not available — skipping confusion matrix")
        except Exception as e:
            print(f"[Results] Confusion matrix error: {e}")

        # if os.path.exists(CHECKPOINT_FILE):
        #     os.remove(CHECKPOINT_FILE)

        # if os.path.exists(TEST_CHECKPOINT_FILE):
        #     os.remove(TEST_CHECKPOINT_FILE)

    try:
        optics.cleanup()
    except Exception as e:
        print(f"[WARN] cleanup failed: {e}")

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 60 + "\n")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print("\n\n[Main] Interrupted — checkpoint saved, run again to resume")

    except Exception as e:

        print(f"\n\n[Main] FATAL: {type(e).__name__}: {e}")

        import traceback
        traceback.print_exc()