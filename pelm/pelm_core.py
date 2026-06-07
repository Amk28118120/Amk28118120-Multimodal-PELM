"""
Core implementation of the photonic extreme learning machine (PELM).
Implements the encoding, fixed photonic reservoir, and linear readout
described in Pierangeli et al. (2021).
"""

import os
import numpy as np
import scipy.ndimage
from config import *
import cv2

RESERVOIR_FILE = "pelm_reservoir.npz"


class PELM_Algorithm:
    """
    Photonic Extreme Learning Machine.

    The reservoir dynamics are implemented physically via free-space
    optical propagation; this class handles encoding and readout only.
    """

    def __init__(self):
        print("[PELM] Initializing model")

        # Seed set HERE so it always fires before W and channel_coords
        # are generated, regardless of what other code ran before.
        np.random.seed(RANDOM_SEED)

        # Load reservoir from disk if saved; generate+save if not.
        # Guarantees W and channel_coords are bit-for-bit identical
        # across training and testing sessions.
        if os.path.exists(RESERVOIR_FILE):
            data = np.load(RESERVOIR_FILE)
            self.W = data['W']
            self.channel_coords = [tuple(c) for c in data['channel_coords']]
            # Restore W_1d for fourier_embedding (empty array = None)
            if 'W_1d' in data and len(data['W_1d']) > 0:
                self.W_1d = data['W_1d']
            else:
                self.W_1d = None
            assert self.W.shape == (SLM_HEIGHT, SLM_WIDTH), \
                (f"Reservoir shape mismatch {self.W.shape} vs "
                    f"({SLM_HEIGHT},{SLM_WIDTH})! "
                    f"Delete {RESERVOIR_FILE} and rerun.")
            assert len(self.channel_coords) == M_FEATURES, \
                (f"Channel count mismatch {len(self.channel_coords)} vs "
                    f"{M_FEATURES}! Delete {RESERVOIR_FILE} and rerun.")
            print(f"[PELM] Reservoir loaded <- {RESERVOIR_FILE}")
        else:
            self.W_1d = None   # will be set by _generate_embedding_matrix if needed
            self.W = self._generate_embedding_matrix()
            self.channel_coords = self._generate_channel_coordinates()
            np.savez(RESERVOIR_FILE,
                        W=self.W,
                        W_1d=self.W_1d if self.W_1d is not None else np.array([]),
                        channel_coords=np.array(self.channel_coords, dtype=np.int32))
            print(f"[PELM] Reservoir saved -> {RESERVOIR_FILE}")

        # Pre-compute channel slicing indices
        self._precompute_channel_slices()

        # Pre-compute MNIST zoom parameters (used in uint8 path of encode_input)
        self._precompute_zoom_params()

        self.beta = None

    
       

    # ======================================================================
    # EMBEDDING MATRIX
    # ======================================================================

    def _generate_embedding_matrix(self):
        """
        Generate the fixed embedding phase mask W (Appendix A.2).
        
        Returns
        -------
        W : ndarray
            Phase mask in the range [0, π].
        """
        h, w = SLM_HEIGHT, SLM_WIDTH
        
        if ENCODING_METHOD == "noise_embedding":
            W_raw = np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
            W = scipy.ndimage.gaussian_filter(W_raw, sigma=NOISE_CORR_LENGTH)
            
            # Normalize correlated noise to [0, π]
            W_min, W_max = W.min(), W.max()
            if W_max > W_min:
                W = (W - W_min) / (W_max - W_min) * PHASE_RANGE_EMBEDDING
            else:
                W = np.zeros_like(W)
        
        elif ENCODING_METHOD == "fourier_embedding":
            W_real = np.zeros((h, w), dtype=np.float32)
            W_imag = np.zeros((h, w), dtype=np.float32)
            
            x = np.linspace(0, 1, w, dtype=np.float32)
            y = np.linspace(0, 1, h, dtype=np.float32)
            X, Y = np.meshgrid(x, y)
            
            for n in range(1, FOURIER_FREQUENCIES + 1):
                phi = np.random.uniform(0, 2 * np.pi)
                omega = 2.0 * np.pi * n
                
                # Fixed diagonal propagation direction
                phase = omega * (X + Y)
                
                W_real += np.cos(phase + phi)
                W_imag += np.sin(phase + phi)
            
            # Wrapped phase in [-π, π]
            W = np.arctan2(W_imag, W_real).astype(np.float32)
            
            # Map wrapped phase directly to [0, π]
            W = (W + np.pi) / (2 * np.pi) * PHASE_RANGE_EMBEDDING
            
        elif ENCODING_METHOD == "no_embedding":

            self.W_1d = None

            W = np.zeros((h, w), dtype=np.float32)

        
        else:
            raise ValueError(f"Unsupported encoding method: {ENCODING_METHOD}")
        
        return W.astype(np.float32)
    # ======================================================================
    # CHANNEL COORDINATES
    # ======================================================================

    def _generate_channel_coordinates(self):
        """
        Select spatial output channels corresponding to camera bins.

        Returns
        -------
        list of (y, x) tuples
        """
        M_grid = int(np.ceil(np.sqrt(M_FEATURES)))
        margin = CAM_BIN_SIZE

        usable_h = CAM_HEIGHT - 2 * margin
        usable_w = CAM_WIDTH  - 2 * margin

        channels = []
        for i in range(M_FEATURES):
            gy = i // M_grid
            gx = i % M_grid

            y = margin + (gy / (M_grid + 1)) * usable_h
            x = margin + (gx / (M_grid + 1)) * usable_w

            y += np.random.uniform(-CAM_BIN_SIZE // 2, CAM_BIN_SIZE // 2)
            x += np.random.uniform(-CAM_BIN_SIZE // 2, CAM_BIN_SIZE // 2)

            y = int(np.clip(y, margin, CAM_HEIGHT - margin))
            x = int(np.clip(x, margin, CAM_WIDTH  - margin))

            channels.append((y, x))

        return channels

    def _precompute_channel_slices(self):
        """Pre-compute all channel slice indices."""
        self.channel_slices = []
        for y, x in self.channel_coords:
            y0 = max(0, y - CAM_BIN_SIZE // 2)
            y1 = min(CAM_HEIGHT, y0 + CAM_BIN_SIZE)
            x0 = max(0, x - CAM_BIN_SIZE // 2)
            x1 = min(CAM_WIDTH,  x0 + CAM_BIN_SIZE)
            self.channel_slices.append((y0, y1, x0, x1))

    def _precompute_zoom_params(self):
        """Pre-compute MNIST isotropic zoom target size and centering offsets."""
        zoom_factor      = 18
        self.zoomed_size = MNIST_INPUT_SIZE * zoom_factor  # 504 px
        self.zoom_h      = min(self.zoomed_size, SLM_HEIGHT)
        self.zoom_w      = min(self.zoomed_size, SLM_WIDTH)
        self.zoom_y0     = (SLM_HEIGHT - self.zoom_h) // 2
        self.zoom_x0     = (SLM_WIDTH  - self.zoom_w) // 2

    # ======================================================================
    # ENCODING
    # ======================================================================

    def encode_input(self, img_2d: np.ndarray) -> np.ndarray:
        """
        Encode input (MNIST OR audio) into phase mask.

        noise_embedding (both MNIST and audio)
        ----------------------------------------
        Input is zoomed to fill part of the SLM, then added to the
        SLM-sized 2D noise mask W.

        fourier_embedding (MNIST only)
        --------------------------------
        Follows Fig 2d of the paper exactly:
        1. Flatten 28×28 → 1D vector of length L=784
        2. Normalise to [0, π]
        3. Add element-wise to the 1D carrier W_1d (also [0, π])
        4. Wrap to [0, 2π]
        5. Reshape to 28×28, then zoom to SLM with nearest-neighbour
            so each input node k occupies one uniform-phase block on the SLM.

        Audio always uses the noise_embedding path regardless of config,
        since the Fourier carrier is defined for the fixed MNIST input size.
        """
        # --------------------------------------------------
        # Normalize input safely
        # --------------------------------------------------
        if img_2d.dtype == np.uint8:
            x_norm = img_2d.astype(np.float32) / 255.0
        else:
            x_norm = img_2d.astype(np.float32)
            x_norm = np.clip(x_norm, 0.0, 1.0)

        h, w = x_norm.shape
        is_mnist = (img_2d.dtype == np.uint8
                    and h == MNIST_INPUT_SIZE
                    and w == MNIST_INPUT_SIZE)

        # --------------------------------------------------
        # FOURIER ENCODING PATH  (MNIST only, Fig 2d)
        # --------------------------------------------------
        if (ENCODING_METHOD == "fourier_embedding"
                and self.W_1d is not None):

            # 1. Flatten to L=784 and scale to [0, π]
            x_flat = x_norm.flatten() * PHASE_RANGE_INPUT          # [0, π]

            # 2. Add 1D carrier element-wise  (both [0, π] → sum in [0, 2π])
            phase_1d = x_flat + self.W_1d                           # [0, 2π]

            # 3. Wrap to [0, 2π]
            phase_1d = np.mod(phase_1d, 2.0 * np.pi).astype(np.float32)

            # 4. Reshape to 28×28, then zoom to SLM with nearest-neighbour
            #    so each of the 784 nodes becomes a uniform-phase block
            phase_2d = phase_1d.reshape(MNIST_INPUT_SIZE, MNIST_INPUT_SIZE)
            phase_slm = scipy.ndimage.zoom(
                phase_2d,
                (SLM_HEIGHT / MNIST_INPUT_SIZE, SLM_WIDTH / MNIST_INPUT_SIZE),
                order=0
            ).astype(np.float32)

            return phase_slm

        # --------------------------------------------------
        # NOISE ENCODING PATH  (noise_embedding, or audio)
        # --------------------------------------------------
        # Determine target size
        if is_mnist:
            target_h, target_w = self.zoom_h, self.zoom_w
            
        else:
            target_max_h = int(SLM_HEIGHT * AUDIO_FILL_H)
            target_max_w = int(SLM_WIDTH * AUDIO_FILL_W)
            scale = min(target_max_h / h, target_max_w / w)
            target_h = max(1, int(h * scale))
            target_w = max(1, int(w * scale))

        img = scipy.ndimage.zoom(
            x_norm,
            (target_h / h, target_w / w),
            order=0
        ).astype(np.float32)

        actual_h, actual_w = img.shape
        y0 = (SLM_HEIGHT - actual_h) // 2
        x0 = (SLM_WIDTH  - actual_w) // 2

        phase = self.W.copy()
        phase[y0:y0 + actual_h, x0:x0 + actual_w] += img * PHASE_RANGE_INPUT
        np.mod(phase, 2 * np.pi, out=phase)

        return phase
    # ======================================================================
    # FEATURE EXTRACTION
    # ======================================================================

    def extract_features(self, camera_frame: np.ndarray) -> np.ndarray:
        """
        Extract reservoir features by spatial binning (Appendix B).

        Parameters
        ----------
        camera_frame : uint8 ndarray  shape (CAM_HEIGHT, CAM_WIDTH)

        Returns
        -------
        ndarray  shape (M_FEATURES,)
        """
        features = np.empty(M_FEATURES, dtype=np.float64)
        for i, (y0, y1, x0, x1) in enumerate(self.channel_slices):
            features[i] = np.mean(camera_frame[y0:y1, x0:x1])
        return features

    # ======================================================================
    # RIDGE REGRESSION  (n_classes parameter added for audio)
    # ======================================================================

    def train_ridge_regression(self,
                                H_train: np.ndarray,
                                y_labels: np.ndarray,
                                n_classes: int = 10) -> None:
        """
        Train the linear readout using ridge regression (Eq. 1).

            β = (H^T H + λI)^{-1} H^T T

        Parameters
        ----------
        H_train   : (N, M)  feature matrix
        y_labels  : (N,)    integer class labels
        n_classes : int     number of output classes
                            10  for MNIST / FSDD / UrbanSound8K
                            50  for ESC-50
                            Default = 10 so all existing MNIST calls are unchanged.
        """
        n_samples = H_train.shape[0]
        print(f"[Ridge] Training  samples={n_samples}  "
                f"features={M_FEATURES}  classes={n_classes}")

        # One-hot target matrix

        if y_labels.ndim == 1 and y_labels.dtype in [np.int32, np.int64]:
            # classification
            T = np.zeros((n_samples, n_classes))
            T[np.arange(n_samples), y_labels] = 1.0
        else:
            # regression
            T = y_labels.reshape(-1, 1)
        HT  = H_train.T
        HTH = HT @ H_train
        self.beta = np.linalg.solve(
            HTH + LAMBDA_REG * np.eye(M_FEATURES, dtype=np.float64),
            HT @ T
        )

        train_preds = np.argmax(H_train @ self.beta, axis=1)
        train_acc   = np.mean(train_preds == y_labels)
        print(f"[Ridge] Train accuracy: {train_acc * 100:.2f}%")

        if train_acc < 0.5:
            print("WARNING: Very low training accuracy — check hardware / encoding!")

    def predict(self, features: np.ndarray) -> int:
        """Predict class label from a single feature vector."""
        if self.beta is None:
            raise RuntimeError("Model has not been trained")
        return int(np.argmax(features @ self.beta))

    # ======================================================================
    # VISUALISATION
    # ======================================================================

    def visualize_embedding(self):
        """Save embedding matrix W as image."""
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 8))
            plt.imshow(self.W, cmap='twilight', vmin=0, vmax=np.pi)
            plt.colorbar(label='Phase (radians)')
            plt.title(f'Embedding Matrix W ({ENCODING_METHOD})\nShape: {self.W.shape}')
            plt.tight_layout()
            plt.savefig('embedding_matrix.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("[PELM] Saved embedding_matrix.png")
        except Exception as e:
            print(f"[PELM] Could not save embedding visualization: {e}")

    def save_phase_mask_example(self, img_2d: np.ndarray, idx: int = 0):
        """Save example of input image + phase mask."""
        try:
            import matplotlib.pyplot as plt
            phase_mask = self.encode_input(img_2d)

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            # Show input — handle both MNIST (uint8) and mel (float)
            if img_2d.dtype == np.uint8:
                axes[0].imshow(img_2d, cmap='gray')
            # axes[0].set_title(f'MNIST Input #{idx}')
                axes[0].set_title(f'{DATASET_TYPE.upper()} Input #{idx}')
            else:
                axes[0].imshow(img_2d, origin='lower', aspect='auto', cmap='magma')
                axes[0].set_title(f'Log-Mel Spec #{idx}')
            axes[0].axis('off')

            im1 = axes[1].imshow(self.W, cmap='twilight', vmin=0, vmax=np.pi)
            axes[1].set_title('Embedding W')
            axes[1].axis('off')
            plt.colorbar(im1, ax=axes[1], label='Phase (rad)', fraction=0.046)

            im2 = axes[2].imshow(phase_mask, cmap='twilight', vmin=0, vmax=2*np.pi)
            axes[2].set_title('Phase Mask\n(W + Input)')
            axes[2].axis('off')
            plt.colorbar(im2, ax=axes[2], label='Phase (rad)', fraction=0.046)

            plt.tight_layout()
            plt.savefig(f'phase_encoding_example_{idx}.png', dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[PELM] Saved phase_encoding_example_{idx}.png")
        except Exception as e:
            print(f"[PELM] Could not save phase mask example: {e}")

    def visualize_feature_extraction(self,
                                        camera_frame: np.ndarray,
                                        features: np.ndarray,
                                        idx: int = 0):
        """Visualize how features are extracted from camera frame."""
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))

            axes[0].imshow(camera_frame, cmap='hot', vmin=0, vmax=255)
            axes[0].set_title('Camera Frame')
            axes[0].axis('off')

            axes[1].bar(range(M_FEATURES), features, width=1, color='steelblue')
            axes[1].set_xlabel('Feature Index')
            axes[1].set_ylabel('Mean Intensity')
            axes[1].set_title(f'Extracted Features ({M_FEATURES} values)\n'
                                f'Mean: {np.mean(features):.1f}, '
                                f'Std: {np.std(features):.1f}')
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(f'feature_extraction_example_{idx}.png',
                        dpi=150, bbox_inches='tight')
            plt.close()
            print(f"[PELM] Saved feature_extraction_example_{idx}.png")
        except Exception as e:
            print(f"[PELM] Could not save feature extraction visualization: {e}")

    def get_statistics(self):
        """Print a brief configuration summary."""
        from config import N_TRAIN, N_TEST
        print("\n" + "=" * 60)
        print("PELM CONFIGURATION")
        print("=" * 60)
        print(f"Encoding:       {ENCODING_METHOD}")
        print(f"Features (M):   {M_FEATURES}")
        print(f"Training (N):   {N_TRAIN}")
        print(f"Testing:        {N_TEST}")
        print(f"N/M ratio:      {N_TRAIN / M_FEATURES:.2f}")
        print(f"Bin size:       {CAM_BIN_SIZE}×{CAM_BIN_SIZE}")
        print(f"Regularization: λ = {LAMBDA_REG}")
        print("=" * 60 + "\n")
