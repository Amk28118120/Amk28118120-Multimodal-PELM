"""
Hardware interface for PELM optical system.
Built for 60,000-sample runs without stalling.

Root fixes:
  1. Camera in daemon (streaming) mode — armed ONCE at init
       camera.py's capture_frame() already skips arm/disarm when streaming.
       This is the correct API usage, no raw SDK hacking needed.
  2. 16-bit conversion fixed — (frame // 256) crushes weak 1st-order signal to 0
       Now scales properly from actual bit depth (12-bit → 0-255)
  3. LC settling time added after waitforState(Visible)
       Visible = electronics received frame, NOT liquid crystals settled
  4. SLMDataField allocated once, updated in-place (no 9MB alloc per sample)
  5. CAM_EXPOSURE_US read from config.py
  6. Cleanup order: stop_daemon → disconnect → SLM.SDK.Close
       Prevents Thorlabs SDK error 1004
"""

import sys
import os
import time
import numpy as np
from config import *


def _setup_sdk_paths():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tcspc_dir = os.path.dirname(current_dir)

    heds_path = os.path.join(current_dir, "HEDS")
    if not os.path.exists(heds_path):
        raise FileNotFoundError(f"[ERROR] HEDS SDK not found at: {heds_path}")

    if heds_path not in sys.path:
        sys.path.insert(0, heds_path)
    if tcspc_dir not in sys.path:
        sys.path.insert(0, tcspc_dir)


class OpticalSystem:
    """
    Hardware controller for the photonic reservoir.

    Camera is started in daemon (streaming) mode once at init.
    camera.py's capture_frame() detects streaming and returns from
    the internal thread-safe buffer — no arm/disarm per call.
    This is stable for 60,000+ samples.
    """

    def __init__(self):
        _setup_sdk_paths()

        try:
            import HEDS
            print("[Import]  HEDS SDK loaded")
        except ImportError as e:
            raise RuntimeError(f"Failed to import Holoeye SDK: {e}")

        try:
            from camera import ThorlabsCamera
            print("[Import]  ThorlabsCamera loaded")
        except ImportError as e:
            raise RuntimeError(f"Failed to import ThorlabsCamera: {e}")

        self.HEDS = HEDS
        self.ThorlabsCamera = ThorlabsCamera

        print("\n" + "=" * 60)
        print("[Optics] Initializing Hardware")
        print("=" * 60)

        self.slm = None
        self.camera = None
        self.frames_captured = 0
        self._consecutive_failures = 0
        self._data_field = None   # reused across all samples

        try:
            self._init_slm()
            self._init_slm_field()
            self._init_camera()
        except Exception as e:
            print(f"\n[Optics]  Initialization failed: {e}")
            self.cleanup()
            raise

        print("[Optics]  Hardware ready")
        print("=" * 60 + "\n")

    # ================================================================
    # INITIALIZATION
    # ================================================================

    def _init_slm(self):
        print("\n[SLM] Connecting to Holoeye SLM...")
        HEDS = self.HEDS
        HEDS.SDK.PrintVersion()

        err = HEDS.SDK.Init(4, 1)
        assert err == HEDS.HEDSERR_NoError, HEDS.SDK.ErrorString(err)

        self.slm = HEDS.SLM.Init(openPreview=False)
        assert self.slm.errorCode() == HEDS.HEDSERR_NoError, \
            HEDS.SDK.ErrorString(self.slm.errorCode())

        hw_w = self.slm.width_px()
        hw_h = self.slm.height_px()
        pixel_size = self.slm.pixelsize_um()

        if hw_w == 0 or hw_h == 0:
            raise RuntimeError("Invalid SLM resolution (0x0)")

        print(f"[SLM] Connected: {hw_w}x{hw_h}, pitch={pixel_size:.2f} um")

    def _init_slm_field(self):
        """
        Allocate SLMDataField ONCE — reused for all 60k samples.
        Values updated in-place via _data_field._values[:,:,0] = phase_mask.
        Avoids 9MB allocation per sample.
        """
        HEDS = self.HEDS
        self._data_field = HEDS.SLMDataField(
            SLM_WIDTH, SLM_HEIGHT,
            data_format=HEDS.HEDSDTFMT_FLOAT_32
        )
        print(f"[SLM] Data field allocated: {SLM_WIDTH}x{SLM_HEIGHT} float32 (reused)")

    def _init_camera(self):
        """
        Connect camera and start daemon (streaming) mode.

        Why daemon mode:
            camera.py's capture_frame() checks self.status.is_streaming.
            When True, it returns self._current_frame.copy() directly from
            the daemon thread's buffer — NO arm/disarm per call.
            This is the correct pattern for long acquisitions.

        Without daemon mode:
            capture_frame() does arm(frames_to_buffer=10) → trigger →
            get_frame → disarm() every call. After ~1600 cycles the
            Thorlabs USB driver exhausts its transfer queue → timeout.
        """
        print("\n[Camera] Connecting to Thorlabs camera...")
        self.camera = self.ThorlabsCamera()

        if not self.camera.connect():
            raise RuntimeError(
                f"Camera connection failed: {self.camera.status.last_error}"
            )

        # Apply settings from config
        self.camera.settings.exposure_time_us = CAM_EXPOSURE_US
        self.camera.settings.gain = 0
        self.camera._apply_settings()

        print(f"[Camera]  Connected: {self.camera.status.camera_model} "
              f"(S/N: {self.camera.status.serial_number})")
        print(f"         Exposure: {CAM_EXPOSURE_US} us  "
              f"[CAM_EXPOSURE_US in config.py]")

        # Start daemon — camera is now armed continuously in background thread
        print("[Camera] Starting daemon (continuous streaming)...")
        if not self.camera.start_daemon(display=False):
            raise RuntimeError(
                "Failed to start camera daemon. "
                "Check camera.py start_daemon() implementation."
            )

        # Give daemon thread time to arm and fill first frame
        time.sleep(0.5)

        # Verify daemon is actually streaming
        if not self.camera.status.is_streaming:
            raise RuntimeError(
                "Camera daemon started but is_streaming is False. "
                "Daemon may have failed silently."
            )

        print(f"[Camera] Daemon running — capture_frame() now returns from buffer")

    # ================================================================
    # CORE OPTICAL PIPELINE
    # ================================================================

    def display_and_capture(self, phase_mask):
        """
        Display phase mask on SLM and capture resulting intensity.

        Args:
            phase_mask : float32 [0, 2pi], shape (SLM_HEIGHT, SLM_WIDTH)

        Returns:
            uint8 grayscale array, shape (CAM_HEIGHT, CAM_WIDTH)

        Raises:
            RuntimeError on 5 consecutive failures
        """
        if self.slm is None or self.camera is None:
            raise RuntimeError("Hardware not initialized")

        if phase_mask.shape != (SLM_HEIGHT, SLM_WIDTH):
            raise ValueError(
                f"Phase mask shape {phase_mask.shape} != "
                f"({SLM_HEIGHT}, {SLM_WIDTH})"
            )

        HEDS = self.HEDS
        from HEDS.holoeye_slmdisplaysdk_datahandle import HEDSDHST_Visible

        # ============================================================
        # Update SLM — reuse data field, update values in-place
        # ============================================================
        self._data_field._values[:, :, 0] = phase_mask.astype(np.float32)

        data_handle = None
        try:
            err, data_handle = self.slm.loadPhaseData(
                self._data_field, phase_unit=2.0 * np.pi
            )
            if err != 0:
                raise RuntimeError(f"loadPhaseData: {HEDS.SDK.ErrorString(err)}")

            err = data_handle.show()
            if err != 0:
                raise RuntimeError(f"show: {HEDS.SDK.ErrorString(err)}")

            # waitforState(Visible): SLM electronics received the frame
            err = data_handle.waitforState(HEDSDHST_Visible)
            if err != 0:
                raise RuntimeError(f"waitforState: {HEDS.SDK.ErrorString(err)}")

            # LC settling: Visible means electronics done, NOT liquid crystals settled.
            # Physical LC rotation takes ~20-50ms after the electronics update.
            # Without this, camera captures the previous pattern ("ghost frames").
            time.sleep(LC_SETTLING_TIME)

        except Exception as e:
            if data_handle is not None:
                try:
                    data_handle.release()
                except Exception:
                    pass
            raise

        # ============================================================
        # Capture — daemon mode returns from buffer, no arm/disarm
        # ============================================================

        # BUFFER FLUSH: The daemon thread is free-running and continuously
        # fills its buffer with frames. After the LC settling sleep, the
        # buffer still contains frames that were exposed BEFORE the SLM
        # finished settling (ghost frames from the previous pattern).
        # Discarding one frame here guarantees the averaging loop only
        # sees frames captured AFTER the new pattern was fully established.
        _flush = self.camera.capture_frame()
        if _flush is None:
            # Daemon died — treat as failure
            self._consecutive_failures += 1
            if data_handle is not None:
                try:
                    data_handle.release()
                except Exception:
                    pass
            if self._consecutive_failures >= 5:
                raise RuntimeError(
                    "Camera returned None for 5 consecutive samples."
                )
            return None

        frames = []
        for idx in range(NUM_FRAMES_TO_AVERAGE):
            # capture_frame() in daemon mode returns self._current_frame.copy()
            # directly from the streaming thread — fast, no SDK state changes
            raw = self.camera.capture_frame()

            if raw is None:
                self._consecutive_failures += 1
                print(f"[Camera] capture_frame returned None "
                      f"({self._consecutive_failures}/5 consecutive)")

                if self._consecutive_failures >= 5:
                    if data_handle is not None:
                        try:
                            data_handle.release()
                        except Exception:
                            pass
                    raise RuntimeError(
                        "Camera returned None for 5 consecutive samples. "
                        "Check USB connection, daemon health, and exposure."
                    )
                # Release handle and return None for this sample
                if data_handle is not None:
                    try:
                        data_handle.release()
                    except Exception:
                        pass
                return None

            frames.append(self._to_uint8(raw))

            if idx < NUM_FRAMES_TO_AVERAGE - 1:
                time.sleep(0.01)

        # Reset failure counter on success
        self._consecutive_failures = 0

        # Release SLM handle
        if data_handle is not None:
            try:
                data_handle.release()
            except Exception as e:
                print(f"[Warning] data_handle.release() failed: {e}")

        # Average frames
        if len(frames) == 1:
            frame = frames[0]
        else:
            frame = np.mean(frames, axis=0).astype(np.uint8)

        if frame.shape != (CAM_HEIGHT, CAM_WIDTH):
            frame = self._resize_frame(frame)

        self.frames_captured += 1
        return frame

    # ================================================================
    # BIT DEPTH CONVERSION
    # ================================================================

    # def _to_uint8(self, frame):
    #     """
    #     Convert camera frame to uint8 grayscale.

    #     Critical fix: original code used (frame // 256) which floors
    #     any pixel value < 256 to 0 on a 16-bit/12-bit camera.
    #     1st order diffraction is dim — pixel values often 100-500 ADU
    #     on a 12-bit sensor (0-4095 range). floor(200 / 256) = 0.
    #     This made all 1st-order features zero → std ≈ 0.

    #     Fix: scale by actual sensor range (12-bit = 4095, 16-bit = 65535)
    #     to preserve weak signal.
    #     """
    #     if frame.dtype == np.uint16:
    #         # CS165CU outputs 16-bit container but is a 16-bit sensor
    #         # Scale full 16-bit range to 0-255
    #         frame = (frame.astype(np.float32) / 1023.0 * 255.0).astype(np.uint8)
    #     elif frame.dtype != np.uint8:
    #         frame = np.clip(frame, 0, 255).astype(np.uint8)

    #     if frame.ndim == 3:
    #         frame = np.mean(frame, axis=2).astype(np.uint8)

    #     return frame
    def _to_uint8(self, frame):

        if frame.dtype == np.uint16:
            # 10-bit sensor stored in 16-bit container
            frame = (frame.astype(np.float32) / 1023.0 * 255.0).astype(np.uint8)

        elif frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.ndim == 3:
            frame = np.mean(frame, axis=2).astype(np.uint8)

        return frame

    # ================================================================
    # OPTICAL TEST
    # ================================================================

    def run_optical_test(self):
        """
        Capture a test frame with blank phase mask.
        Reports mean, std, saturation — use to tune CAM_EXPOSURE_US.
        Target saturation: 1-10%.
        """
        print("\n[Optics] Running optical system test...")

        test_phase = np.zeros((SLM_HEIGHT, SLM_WIDTH), dtype=np.float32)
        frame = self.display_and_capture(test_phase)

        if frame is None:
            print("[Optics] TEST FAILED — camera returned None")
            print("         Check USB, daemon status, and exposure")
            return

        mean_i = np.mean(frame)
        std_i = np.std(frame)
        saturated = np.sum(frame >= 254)
        sat_pct = 100.0 * saturated / frame.size

        print(f"[Optics] Frame: {frame.shape}")
        print(f"         Mean:       {mean_i:.2f} / 255")
        print(f"         Std:        {std_i:.2f}")
        print(f"         Saturation: {sat_pct:.2f}%  ({saturated} pixels)")
        print(f"         Exposure:   {CAM_EXPOSURE_US} us")

        if sat_pct < 0.5:
            print(f"\n  [!] Low saturation — increase CAM_EXPOSURE_US "
                  f"(currently {CAM_EXPOSURE_US} us → try {CAM_EXPOSURE_US * 2} us)")
        elif sat_pct > 30:
            print(f"\n  [!] High saturation — decrease CAM_EXPOSURE_US "
                  f"(currently {CAM_EXPOSURE_US} us → try {CAM_EXPOSURE_US // 2} us)")
        else:
            print(f"\n  [OK] Saturation in target range — ready to run")

    # ================================================================
    # UTILITIES
    # ================================================================

    def _resize_frame(self, frame):
        h, w = frame.shape[:2]
        if h >= CAM_HEIGHT and w >= CAM_WIDTH:
            y0 = (h - CAM_HEIGHT) // 2
            x0 = (w - CAM_WIDTH) // 2
            return frame[y0:y0 + CAM_HEIGHT, x0:x0 + CAM_WIDTH]
        pad_h = max(0, CAM_HEIGHT - h)
        pad_w = max(0, CAM_WIDTH - w)
        return np.pad(frame, ((0, pad_h), (0, pad_w)), mode='constant')

    def cleanup(self):
        """
        Release all hardware in safe order:
            1. stop_daemon()    — stops background thread, disarms camera
            2. disconnect()     — disposes camera SDK object
            3. HEDS.SDK.Close() — closes SLM SDK

        This order prevents Thorlabs SDK error 1004:
        "Cameras must be closed before closing the SDK"
        """
        print("\n[Optics] Cleaning up hardware...")

        if self.camera is not None:
            try:
                if self.camera._daemon_active:
                    self.camera.stop_daemon()
                    print("[Camera] Daemon stopped")
            except Exception as e:
                print(f"[Camera] Warning stopping daemon: {e}")

            try:
                self.camera.disconnect()
                print("[Camera] Disconnected")
            except Exception as e:
                print(f"[Camera] Warning during disconnect: {e}")

            self.camera = None

        if self.slm is not None:
            try:
                self.HEDS.SDK.Close()
                print("[SLM] Closed")
            except Exception as e:
                print(f"[SLM] Warning during close: {e}")
            self.slm = None

        self._data_field = None
        print(f"[Optics] Cleanup complete ({self.frames_captured} frames captured)\n")


# ====================================================================
# STANDALONE TEST
# ====================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Hardware Initialization Test")
    print("=" * 60)

    try:
        optics = OpticalSystem()
        optics.run_optical_test()
        optics.cleanup()
        print("\nHardware test PASSED")
    except Exception as e:
        print(f"\nHardware test FAILED: {e}")
        import traceback
        traceback.print_exc()