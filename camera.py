import numpy as np
import time
import threading
import logging
import argparse
import signal
import sys
import os
import json
import yaml
from typing import Optional, Callable, Tuple, Dict, Any
from dataclasses import dataclass, asdict
from collections import deque
import cv2
import ctypes

import os

current_dir = os.path.dirname(os.path.abspath(__file__))
dlls_dir = os.path.join(current_dir, "dlls")

if not os.path.exists(dlls_dir):
    raise FileNotFoundError(f"DLL directory not found: {dlls_dir}")

# Python 3.8+ (Windows)
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(dlls_dir)

# Load Thorlabs DLLs
dlls_dir = f'{os.getcwd()}/dlls'
current_dir = os.path.dirname(os.path.abspath(__file__))
dlls_dir = os.path.join(current_dir, "dlls")
os.add_dll_directory(dlls_dir)

dlls = [
    f'{dlls_dir}/thorlabs_tsi_LUT.dll',
    f'{dlls_dir}/thorlabs_tsi_camera_sdk.dll',
    f'{dlls_dir}/thorlabs_tsi_color_processing.dll',
    f'{dlls_dir}/thorlabs_tsi_color_processing_vector_avx2.dll',
    f'{dlls_dir}/thorlabs_tsi_demosaic.dll',
    f'{dlls_dir}/thorlabs_tsi_demosaic_vector_avx2.dll',
    f'{dlls_dir}/thorlabs_tsi_loggerx.dll',
    f'{dlls_dir}/thorlabs_tsi_mono_to_color_processing.dll',
    f'{dlls_dir}/thorlabs_tsi_polarization_processor.dll',
    f'{dlls_dir}/thorlabs_tsi_polarization_processor_vector_avx2.dll',
    f'{dlls_dir}/thorlabs_tsi_polarization_processor_vector_avx512.dll',
    f'{dlls_dir}/thorlabs_tsi_usb_hotplug_monitor.dll',
    f'{dlls_dir}/thorlabs_tsi_zelux_camera_device.dll'
]

for dll in dlls:    
    try:
        ctypes.windll.LoadLibrary(dll)
    except Exception as e:
        print(f"Failed to load {dll}: {e}")

try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame, TLCameraError
    from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE, TRIGGER_POLARITY
except ImportError as e:
    logging.error(f"Failed to import Thorlabs SDK: {e}")
    raise
# Load Thorlabs DLLs (tcspc/dlls)
current_dir = os.path.dirname(os.path.abspath(__file__))
dlls_dir = os.path.join(current_dir, "dlls")

if not os.path.exists(dlls_dir):
    raise FileNotFoundError(f"DLL directory not found: {dlls_dir}")

# Required for Python 3.8+ on Windows
if hasattr(os, "add_dll_directory"):
    os.add_dll_directory(dlls_dir)

dlls = [
    f'{dlls_dir}/thorlabs_tsi_LUT.dll',
    f'{dlls_dir}/thorlabs_tsi_camera_sdk.dll',
    f'{dlls_dir}/thorlabs_tsi_color_processing.dll',
    f'{dlls_dir}/thorlabs_tsi_color_processing_vector_avx2.dll',
    f'{dlls_dir}/thorlabs_tsi_demosaic.dll',
    f'{dlls_dir}/thorlabs_tsi_demosaic_vector_avx2.dll',
    f'{dlls_dir}/thorlabs_tsi_loggerx.dll',
    f'{dlls_dir}/thorlabs_tsi_mono_to_color_processing.dll',
    f'{dlls_dir}/thorlabs_tsi_polarization_processor.dll',
    f'{dlls_dir}/thorlabs_tsi_polarization_processor_vector_avx2.dll',
    f'{dlls_dir}/thorlabs_tsi_polarization_processor_vector_avx512.dll',
    f'{dlls_dir}/thorlabs_tsi_usb_hotplug_monitor.dll',
    f'{dlls_dir}/thorlabs_tsi_zelux_camera_device.dll'
]

for dll in dlls:
    dll_path = os.path.join(dlls_dir, dll)
    try:
        ctypes.WinDLL(dll_path)
    except OSError as e:
        raise RuntimeError(f"Failed to load DLL: {dll_path}") from e
THORLABS_SDK_AVAILABLE = True
try:
    from thorlabs_tsi_sdk.tl_camera import TLCameraSDK, TLCamera, Frame, TLCameraError
    from thorlabs_tsi_sdk.tl_camera_enums import OPERATION_MODE, TRIGGER_POLARITY
except Exception as e:
    logging.error(f"Thorlabs SDK not available: {e}")
    THORLABS_SDK_AVAILABLE = False
    # Provide safe fallbacks to avoid NameError in environments without the SDK
    TLCameraSDK = None
    TLCamera = None
    Frame = None
    TLCameraError = Exception
    # Minimal OPERATION_MODE placeholder used only for symbolic reference
    OPERATION_MODE = type("OPERATION_MODE", (), {"SOFTWARE_TRIGGERED": 0})
    TRIGGER_POLARITY = type("TRIGGER_POLARITY", (), {})
#Load Thorlabs DLLs (tcspc/dlls)


@dataclass
class CameraSettings:
    """Container for camera settings"""
    exposure_time_us: int = 10000
    gain: int = 0
    roi: Optional[Tuple[int, int, int, int]] = None
    binx: int = 1
    biny: int = 1
    frames_per_trigger: int = 0


@dataclass
class DisplaySettings:
    """Container for display settings"""
    window_name: str = "Thorlabs Camera"
    window_width: int = 800
    window_height: int = 600
    show_fps: bool = True
    show_timestamp: bool = True
    show_frame_count: bool = True
    auto_contrast: bool = True
    zoom_factor: float = 1.0
    colormap: int = cv2.COLORMAP_HOT  # Default colormap for mono cameras
    bayer_pattern: str = "BG"  # Try "BG","GB","RG","GR" if colors look wrong


@dataclass
class CameraStatus:
    """Container for camera status information"""
    is_connected: bool = False
    is_streaming: bool = False
    is_displaying: bool = False
    current_fps: float = 0.0
    frames_captured: int = 0
    last_error: Optional[str] = None
    camera_model: Optional[str] = None
    serial_number: Optional[str] = None


class ThorlabsCamera:
    """
    Thorlabs camera control system for streaming and capture
    
    Provides programmatic frame capture and daemon-style streaming
    Compatible with laboratory control system architecture
    """
    
    def __init__(self, config="config.yaml", camera_serial=None, buffer_size=200):    #buffer_size increased from 30 to 200 change date-07/03/26    
        self.config_file = config
        self.camera_serial = camera_serial
        self.buffer_size = buffer_size
        
        # SDK and camera objects
        self._sdk = None
        self._camera = None
        
        # Threading for daemon mode
        self._daemon_thread = None
        self._daemon_active = False
        self._stop_daemon = threading.Event()
        self._frame_lock = threading.Lock()
        
        # Display threading
        self._display_thread = None
        self._display_active = False
        self._stop_display = threading.Event()
        
        # Frame management
        self._frame_buffer = deque(maxlen=buffer_size)
        self._current_frame = None
        self._frame_timestamp = None
        self._display_frame = None
        
        # Status and settings
        self.status = CameraStatus()
        self.settings = CameraSettings()
        self.display_settings = DisplaySettings()
        
        # FPS calculation
        self._fps_timestamps = deque(maxlen=30)
        
        # Callbacks and output
        self._frame_callback = None
        self._save_frames = False
        self._output_dir = "frames"
        
        # Logging setup
        self.logger = logging.getLogger(__name__)
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
        # Load configuration
        self.update_config()
        
        # Signal handling for daemon mode
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def update_config(self, config=None):
        """Load configuration from YAML file"""
        if config:
            self.config_file = config
        
        if not os.path.exists(self.config_file):
            self.logger.warning(f"Config file {self.config_file} not found, using defaults")
            return
        
        try:
            with open(self.config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            # Update settings from config
            camera_config = config_data.get('camera', {})
            for key, value in camera_config.items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            
            # Update display settings from config
            display_config = config_data.get('display', {})
            for key, value in display_config.items():
                if hasattr(self.display_settings, key):
                    setattr(self.display_settings, key, value)
            
            # Update system settings
            system_config = config_data.get('system', {})
            self.buffer_size = system_config.get('buffer_size', self.buffer_size)
            self._output_dir = system_config.get('output_dir', self._output_dir)
            self._save_frames = system_config.get('save_frames', False)
            
            self.logger.info("Configuration updated")
            
        except Exception as e:
            self.logger.error(f"Failed to load config: {e}")

    def connect(self):
        """Connect to Thorlabs camera"""
        try:
            self.logger.info("Connecting to Thorlabs camera...")

            if not THORLABS_SDK_AVAILABLE:
                self._set_error("Thorlabs SDK not available in Python path or DLLs missing")
                return False

            if self._sdk is None:
                # Initialize SDK (will be None if SDK not installed)
                self._sdk = TLCameraSDK()
            
            # Discover cameras
            available_cameras = self._sdk.discover_available_cameras()
            if not available_cameras:
                self._set_error("No cameras found")
                return False
            
            self.logger.info(f"Found cameras: {available_cameras}")
            
            # Select camera
            if self.camera_serial and self.camera_serial in available_cameras:
                selected_serial = self.camera_serial
            else:
                selected_serial = available_cameras[0]
            
            # Open camera
            self._camera = self._sdk.open_camera(selected_serial)
            
            # Update status
            self.status.serial_number = selected_serial
            self.status.camera_model = self._camera.model
            self.status.is_connected = True
            self.status.last_error = None
            
            # Apply settings
            self._apply_settings()
            
            self.logger.info(f"Connected to {self.status.camera_model} (S/N: {selected_serial})")
            return True
            
        except Exception as e:
            self._set_error(f"Connection failed: {str(e)}")
            return False

    def disconnect(self):
        """Disconnect camera and cleanup"""
        try:
            self.stop_display()
            self.stop_daemon()
            
            if self._camera:
                if self._camera.is_armed:
                    self._camera.disarm()
                self._camera.dispose()
                self._camera = None
                
            if self._sdk:
                self._sdk.dispose()
                self._sdk = None
                
            self.status.is_connected = False
            self.logger.info("Camera disconnected")
            
        except Exception as e:
            self.logger.error(f"Disconnect error: {e}")

    def capture_frame(self):
        """Capture single frame programmatically"""
        if not self.status.is_connected:
            self._set_error("Camera not connected")
            return None
        
        try:
            # Use current frame if streaming
            if self.status.is_streaming:
                with self._frame_lock:
                    if self._current_frame is not None:
                        return self._current_frame.copy()
                    else:                               #else block added change date-07/03/26
                        time.sleep(0.002)
                        return None
                    
            # Single shot capture
            was_streaming = self.status.is_streaming
            if was_streaming:
                self._stop_streaming()
            
            self._camera.frames_per_trigger_zero_for_unlimited = 1
            self._camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
            
            self._camera.arm(frames_to_buffer=10)
            self._camera.issue_software_trigger()
            
            # Wait for frame with timeout
            timeout_start = time.time()
            while time.time() - timeout_start < 5.0:
                frame = self._camera.get_pending_frame_or_null()
                if frame is not None:
                    captured_frame = frame.image_buffer.copy()
                    self._camera.disarm()
                    
                    if was_streaming:
                        self._start_streaming()
                    
                    return captured_frame
                time.sleep(0.01)
            
            self._camera.disarm()
            self._set_error("Frame capture timeout")
            return None
            
        except Exception as e:
            self._set_error(f"Capture failed: {str(e)}")
            return None

    def start_daemon(self, display=False):
        """Start streaming daemon in background thread"""
        if self._daemon_active:
            self.logger.warning("Daemon already running")
            return True
        
        if not self.status.is_connected:
            self._set_error("Camera not connected")
            return False
        
        try:
            self._stop_daemon.clear()
            
            # Start camera streaming BEFORE daemon thread    change date-07/03/26
            if not self._start_streaming():
                return False
 
            self._daemon_thread = threading.Thread(target=self._daemon_loop, daemon=False)
            self._daemon_thread.start()
            
            self._daemon_active = True
            self.logger.info("Daemon started - streaming in background")
            
            # Start display if requested
            if display:
                self.start_display()
            
            return True
            
        except Exception as e:
            self._set_error(f"Daemon start failed: {str(e)}")
            return False

    def stop_daemon(self):
        """Stop streaming daemon"""
        if not self._daemon_active:
            return
        
        self.logger.info("Stopping daemon...")
        self._stop_daemon.set()
        
        if self._daemon_thread and self._daemon_thread.is_alive():
            self._daemon_thread.join(timeout=5.0)
        
        self._daemon_active = False
        self.status.is_streaming = False
        self.logger.info("Daemon stopped")

    def start_display(self):
        """Start real-time display in separate thread"""
        if self._display_active:
            self.logger.warning("Display already active")
            return True
            
        try:
            self._stop_display.clear()
            self._display_thread = threading.Thread(target=self._display_loop, daemon=False)
            self._display_thread.start()
            
            self._display_active = True
            self.status.is_displaying = True
            self.logger.info("Display started")
            return True
            
        except Exception as e:
            self.logger.error(f"Display start failed: {e}")
            return False

    def stop_display(self):
        """Stop real-time display"""
        if not self._display_active:
            return
            
        self.logger.info("Stopping display...")
        self._stop_display.set()
        
        if self._display_thread and self._display_thread.is_alive():
            self._display_thread.join(timeout=3.0)
        
        self._display_active = False
        self.status.is_displaying = False
        
        # Close OpenCV windows
        try:
            cv2.destroyWindow(self.display_settings.window_name)
        except:
            pass
        
        self.logger.info("Display stopped")

    def get_status(self):
        """Get current system status"""
        return {
            'connected': self.status.is_connected,
            'streaming': self.status.is_streaming,
            'displaying': self.status.is_displaying,
            'daemon_active': self._daemon_active,
            'fps': self.status.current_fps,
            'frames_captured': self.status.frames_captured,
            'last_error': self.status.last_error,
            'camera_model': self.status.camera_model,
            'serial_number': self.status.serial_number
        }

    def get_current_frame(self):
        """Get most recent frame (thread-safe)"""
        with self._frame_lock:
            if self._current_frame is not None:
                return self._current_frame.copy()
        return None

    def save_frame(self, frame, filename=None):
        """Save frame to disk"""
        if frame is None:
            return False
        
        if not os.path.exists(self._output_dir):
            os.makedirs(self._output_dir)
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{timestamp}.png"
        
        filepath = os.path.join(self._output_dir, filename)
        
        try:
            # Normalize to 8-bit if needed
            if frame.dtype != np.uint8:
                frame_normalized = ((frame - frame.min()) / 
                                  (frame.max() - frame.min()) * 255).astype(np.uint8)
            else:
                frame_normalized = frame.copy()
            
            # If single channel and a Bayer pattern was set, demosaic to BGR before saving
            if frame_normalized.ndim == 2:
                pattern = (self.display_settings.bayer_pattern or "").upper()
                const_name = f"COLOR_BAYER_{pattern}2BGR"
                conv_code = getattr(cv2, const_name, None)
                if conv_code is not None:
                    frame_to_write = cv2.cvtColor(frame_normalized, conv_code)
                elif self.display_settings.colormap is not None and self.display_settings.colormap >= 0:
                    frame_to_write = cv2.applyColorMap(frame_normalized, self.display_settings.colormap)
                else:
                    frame_to_write = cv2.cvtColor(frame_normalized, cv2.COLOR_GRAY2BGR)
            else:
                frame_to_write = frame_normalized
            
            cv2.imwrite(filepath, frame_to_write)
            self.logger.info(f"Frame saved: {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Save failed: {e}")
            return False

    def set_frame_callback(self, callback):
        """Set callback for each new frame"""
        self._frame_callback = callback

    def set_display_settings(self, **kwargs):
        """Update display settings"""
        for key, value in kwargs.items():
            if hasattr(self.display_settings, key):
                setattr(self.display_settings, key, value)
                self.logger.info(f"Display setting updated: {key} = {value}")

    def wait_for_stabilization(self):
        """Wait for camera to stabilize after settings change"""
        if not self.status.is_connected:
            return False
        
        self.logger.info("Waiting for camera stabilization...")
        time.sleep(1.0)  # Basic stabilization wait
        return True

    def shutdown(self):
        """Complete system shutdown"""
        self.logger.info("Shutting down camera system...")
        self.stop_display()
        self.stop_daemon()
        self.disconnect()
        self.logger.info("Shutdown complete")

    # Private methods
    
    def _daemon_loop(self):
        """Main daemon streaming loop"""
        self.logger.info("Daemon loop started")
        
        try:
            # if not self._start_streaming():           change date -07/03/26   
                # return
            
            while not self._stop_daemon.is_set():
                try:
                    frame = self._camera.get_pending_frame_or_null()
                    if frame is not None:
                        self._process_frame(frame)
                    else:
                        time.sleep(0.002)   #time sleep increased from 0.001 to 0.002 change date-07/03/26
                        
                except Exception as e:
                    self.logger.error(f"Daemon error: {e}")
                    break
                    
        finally:
            self._stop_streaming()
            self.logger.info("Daemon loop ended")

    def _display_loop(self):
        """Display loop for real-time visualization"""
        self.logger.info("Display loop started")
        
        # Create window
        cv2.namedWindow(self.display_settings.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.display_settings.window_name, 
                        self.display_settings.window_width, 
                        self.display_settings.window_height)
        
        # Setup mouse callback for zoom/pan
        cv2.setMouseCallback(self.display_settings.window_name, self._mouse_callback)
        
        try:
            while not self._stop_display.is_set():
                # Get current frame
                display_frame = None
                with self._frame_lock:
                    if self._current_frame is not None:
                        display_frame = self._current_frame.copy()
                
                if display_frame is not None:
                    # Process frame for display
                    processed_frame = self._process_display_frame(display_frame)
                    
                    # Add overlay information
                    if (self.display_settings.show_fps or 
                        self.display_settings.show_timestamp or 
                        self.display_settings.show_frame_count):
                        processed_frame = self._add_overlay(processed_frame)
                    
                    # Display frame
                    cv2.imshow(self.display_settings.window_name, processed_frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 'q' or ESC
                        self.logger.info("Display stopped by user")
                        break
                    elif key == ord('s'):  # Save current frame
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"display_capture_{timestamp}.png"
                        self.save_frame(display_frame, filename)
                    elif key == ord('c'):  # Toggle auto contrast
                        self.display_settings.auto_contrast = not self.display_settings.auto_contrast
                        self.logger.info(f"Auto contrast: {self.display_settings.auto_contrast}")
                    elif key == ord('r'):  # Reset zoom
                        self.display_settings.zoom_factor = 1.0
                        self.logger.info("Zoom reset")
                else:
                    cv2.waitKey(30)  # Wait longer if no frame
                    
        except Exception as e:
            self.logger.error(f"Display error: {e}")
        finally:
            cv2.destroyWindow(self.display_settings.window_name)
            self.logger.info("Display loop ended")

    def _process_display_frame(self, frame):
        """Process frame for display"""
        try:
            # Convert to proper display format
            if frame.dtype != np.uint8:
                if self.display_settings.auto_contrast:
                    # Auto contrast adjustment
                    frame_min, frame_max = np.percentile(frame, [1, 99])
                    frame_normalized = np.clip((frame - frame_min) / (frame_max - frame_min), 0, 1)
                    display_frame = (frame_normalized * 255).astype(np.uint8)
                else:
                    # Simple normalization
                    frame_normalized = ((frame - frame.min()) / 
                                      (frame.max() - frame.min()) * 255).astype(np.uint8)
                    display_frame = frame_normalized
            else:
                display_frame = frame.copy()
            
            # Apply zoom if needed
            if abs(self.display_settings.zoom_factor - 1.0) > 0.01:
                height, width = display_frame.shape[:2]
                new_width = int(width * self.display_settings.zoom_factor)
                new_height = int(height * self.display_settings.zoom_factor)
                display_frame = cv2.resize(display_frame, (new_width, new_height))
            
            # If grayscale / Bayer, convert to BGR (demosaic) if bayer_pattern set.
            if len(display_frame.shape) == 2:
                pattern = (self.display_settings.bayer_pattern or "").upper()
                const_name = f"COLOR_BAYER_{pattern}2BGR"
                conv_code = getattr(cv2, const_name, None)
                if conv_code is not None:
                    # use OpenCV demosaic for Bayer -> BGR
                    display_frame = cv2.cvtColor(display_frame, conv_code)
                elif self.display_settings.colormap >= 0:
                    # fallback: apply colormap to visualize mono data
                    display_frame = cv2.applyColorMap(display_frame, self.display_settings.colormap)
                else:
                    # convert gray to BGR for overlays
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)
            
            return display_frame
            
        except Exception as e:
            self.logger.error(f"Display processing error: {e}")
            return frame

    def _add_overlay(self, frame):
        """Add information overlay to display frame"""
        try:
            overlay_y = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            color = (0, 255, 0)  # Green
            thickness = 1
            
            if self.display_settings.show_fps:
                fps_text = f"FPS: {self.status.current_fps:.1f}"
                cv2.putText(frame, fps_text, (10, overlay_y), font, font_scale, color, thickness)
                overlay_y += 25
            
            if self.display_settings.show_frame_count:
                count_text = f"Frames: {self.status.frames_captured}"
                cv2.putText(frame, count_text, (10, overlay_y), font, font_scale, color, thickness)
                overlay_y += 25
            
            if self.display_settings.show_timestamp:
                timestamp = time.strftime("%H:%M:%S")
                cv2.putText(frame, timestamp, (10, overlay_y), font, font_scale, color, thickness)
                overlay_y += 25
            
            # Add camera info at top right
            if self.status.camera_model:
                model_text = f"{self.status.camera_model}"
                text_size = cv2.getTextSize(model_text, font, font_scale, thickness)[0]
                cv2.putText(frame, model_text, 
                          (frame.shape[1] - text_size[0] - 10, 30), 
                          font, font_scale, color, thickness)
            
            # Add keyboard shortcuts at bottom
            shortcuts = "Keys: Q=quit, S=save, C=contrast, R=reset zoom"
            text_size = cv2.getTextSize(shortcuts, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            cv2.putText(frame, shortcuts, 
                      (10, frame.shape[0] - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 128, 128), 1)
            
            return frame
            
        except Exception as e:
            self.logger.error(f"Overlay error: {e}")
            return frame

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for zoom/pan"""
        if event == cv2.EVENT_MOUSEWHEEL:
            # Zoom with mouse wheel
            if flags > 0:  # Scroll up
                self.display_settings.zoom_factor *= 1.1
            else:  # Scroll down
                self.display_settings.zoom_factor /= 1.1
            
            # Limit zoom range
            self.display_settings.zoom_factor = max(0.1, min(5.0, self.display_settings.zoom_factor))

    def _start_streaming(self):
        """Internal streaming start"""
        try:
            self._camera.frames_per_trigger_zero_for_unlimited = 0
            self._camera.operation_mode = OPERATION_MODE.SOFTWARE_TRIGGERED
            
            self._camera.arm(frames_to_buffer=100)          #buffer increased from 10 to 100 change date-07/03/26
            self._camera.issue_software_trigger()
            time.sleep(0.05)  # Short delay to ensure streaming starts      change date-07/03/26    


            self.status.is_streaming = True
            self.status.frames_captured = 0
            self._fps_timestamps.clear()
            
            return True
            
        except Exception as e:
            self._set_error(f"Streaming start failed: {str(e)}")
            return False

    def _stop_streaming(self):
        """Internal streaming stop"""
        try:
            if self._camera and self._camera.is_armed:
                self._camera.disarm()
            self.status.is_streaming = False
            
        except Exception as e:
            self.logger.error(f"Streaming stop error: {e}")

    def _process_frame(self, frame):
        """Process received frame"""
        try:
            # Update current frame
            with self._frame_lock:
                self._current_frame = frame.image_buffer.copy()
                self._frame_timestamp = time.time()
                self._frame_buffer.append(self._current_frame)
            
            # Update statistics
            self.status.frames_captured += 1
            self._update_fps()
            
            # Save frame if enabled
            if self._save_frames:
                timestamp = time.strftime("%Y%m%d_%H%M%S_%f")[:-3]
                self.save_frame(self._current_frame, f"frame_{timestamp}.png")
            
            # Call frame callback
            if self._frame_callback:
                try:
                    self._frame_callback(self._current_frame)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
                    
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")

    def _update_fps(self):
        """Update FPS calculation"""
        current_time = time.time()
        self._fps_timestamps.append(current_time)
        
        if len(self._fps_timestamps) >= 2:
            time_span = self._fps_timestamps[-1] - self._fps_timestamps[0]
            if time_span > 0:
                self.status.current_fps = (len(self._fps_timestamps) - 1) / time_span

    def _apply_settings(self):
        """Apply current settings to camera"""
        if not self._camera:
            return False
        
        try:
            # Exposure time
            if hasattr(self._camera, 'exposure_time_us'):
                self._camera.exposure_time_us = self.settings.exposure_time_us
            
            # Gain
            if hasattr(self._camera, 'gain') and self.settings.gain >= 0:
                gain_range = self._camera.gain_range
                if gain_range.min <= self.settings.gain <= gain_range.max:
                    self._camera.gain = self.settings.gain
            
            # Binning
            if hasattr(self._camera, 'binx'):
                self._camera.binx = self.settings.binx
            if hasattr(self._camera, 'biny'):
                self._camera.biny = self.settings.biny
            
            # ROI
            if self.settings.roi:
                from thorlabs_tsi_sdk.tl_camera import ROI
                roi = ROI(*self.settings.roi)
                self._camera.roi = roi
            
            self.logger.info("Camera settings applied")
            return True
            
        except Exception as e:
            self.logger.error(f"Settings application failed: {e}")
            return False

    def _set_error(self, error_msg):
        """Set error status"""
        self.status.last_error = error_msg
        self.logger.error(error_msg)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown()
        sys.exit(0)


def main():
    """Main entry point for command line usage"""
    parser = argparse.ArgumentParser(description="Thorlabs Camera Control System")
    parser.add_argument("--config", "-c", default="config.yaml", 
                       help="Configuration file path")
    parser.add_argument("--serial", "-s", help="Camera serial number")
    parser.add_argument("--daemon", "-d", action="store_true", 
                       help="Run as streaming daemon")
    parser.add_argument("--display", action="store_true", 
                       help="Show live display window")
    parser.add_argument("--capture", action="store_true", 
                       help="Capture single frame")
    parser.add_argument("--output", "-o", help="Output filename for capture")
    parser.add_argument("--status", action="store_true", 
                       help="Show camera status")
    parser.add_argument("--kill", "-k", action="store_true", 
                       help="Kill running daemon")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose logging")
    
    args = parser.parse_args()
    
    # Setup logging
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Handle kill command
    if args.kill:
        # In a real implementation, this would communicate with running daemon
        print("Kill command not implemented - use Ctrl+C to stop daemon")
        return
    
    # Create camera instance
    camera = ThorlabsCamera(config=args.config, camera_serial=args.serial)
    
    try:
        # Connect to camera
        if not camera.connect():
            print(f"Failed to connect: {camera.status.last_error}")
            return 1
        
        # Handle different modes
        if args.status:
            status = camera.get_status()
            print("Camera Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
                
        elif args.capture:
            print("Capturing frame...")
            frame = camera.capture_frame()
            if frame is not None:
                filename = args.output or f"capture_{time.strftime('%Y%m%d_%H%M%S')}.png"
                if camera.save_frame(frame, filename):
                    print(f"Frame captured: {filename}")
                else:
                    print("Failed to save frame")
            else:
                print("Failed to capture frame")
                
        elif args.daemon:
            print("Starting daemon mode - Press Ctrl+C to stop")
            display_enabled = args.display
            if camera.start_daemon(display=display_enabled):
                try:
                    while camera._daemon_active:
                        time.sleep(1)
                        status = camera.get_status()
                        print(f"FPS: {status['fps']:.1f}, Frames: {status['frames_captured']}, Display: {status['displaying']}")
                except KeyboardInterrupt:
                    print("\nStopping daemon...")
                    camera.stop_daemon()
            else:
                print("Failed to start daemon")
                
        elif args.display:
            print("Starting display mode - Press 'q' in window or Ctrl+C to stop")
            # Start daemon for streaming
            if camera.start_daemon():
                # Start display
                if camera.start_display():
                    try:
                        while camera._display_active:
                            time.sleep(0.1)
                    except KeyboardInterrupt:
                        print("\nStopping display...")
                        camera.stop_display()
                        camera.stop_daemon()
                else:
                    print("Failed to start display")
                    camera.stop_daemon()
            else:
                print("Failed to start streaming for display")
                
        else:
            print("No action specified. Use --help for options")
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera.shutdown()
        
    return 0


if __name__ == "__main__":
    sys.exit(main())