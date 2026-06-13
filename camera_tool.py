"""
Camera capture tool for the LangGraph ReAct agent.

Captures a temporary snapshot from the first available camera device.
Maintains a rolling buffer of at most MAX_CAPTURES images on disk.
"""

import os
import glob
import base64
from datetime import datetime

import cv2

# ── Configuration ─────────────────────────────────────────────────────────────

CAPTURE_DIR = "/tmp/agent_captures"   # Temporary; lives in RAM-backed tmpfs
MAX_CAPTURES = 20                      # Maximum images kept on disk at once
CAMERA_INDEX = None                    # None = auto-detect on first use

# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_camera() -> int | None:
    """Scan /dev/video* and return the index of the first working camera."""
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"   📷 Camera auto-detected at /dev/video{i}")
                return i
    return None


def _evict_old_captures():
    """Delete oldest captures when the buffer exceeds MAX_CAPTURES."""
    pattern = os.path.join(CAPTURE_DIR, "capture_*.jpg")
    files = sorted(glob.glob(pattern))          # ascending by timestamp name
    excess = len(files) - MAX_CAPTURES + 1      # +1 makes room for the new one
    for path in files[:excess]:
        try:
            os.remove(path)
        except OSError:
            pass


def capture_image_to_disk() -> str:
    """
    Capture one frame from the camera and save it as a JPEG.

    Returns:
        Absolute path to the saved image file.

    Raises:
        RuntimeError: If no camera is found or the frame cannot be grabbed.
    """
    global CAMERA_INDEX

    os.makedirs(CAPTURE_DIR, exist_ok=True)

    # Auto-detect camera index once
    if CAMERA_INDEX is None:
        CAMERA_INDEX = _detect_camera()
    if CAMERA_INDEX is None:
        raise RuntimeError("No camera device found on this system.")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        # Try re-detection in case the device changed
        CAMERA_INDEX = _detect_camera()
        if CAMERA_INDEX is None:
            raise RuntimeError("Cannot open camera device.")
        cap = cv2.VideoCapture(CAMERA_INDEX)

    try:
        # Warm-up: discard a few frames so auto-exposure settles
        for _ in range(3):
            cap.read()

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to grab a frame from the camera.")

        _evict_old_captures()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
        cv2.imwrite(path, frame)
        print(f"   📷 Image saved: {path}")
        return path
    finally:
        cap.release()


def image_to_base64(path: str) -> str:
    """Return a base64-encoded JPEG string suitable for an LLM vision API."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")
