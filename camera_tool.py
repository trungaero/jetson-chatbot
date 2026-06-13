"""
Camera capture tool for the LangGraph ReAct agent.

Captures a temporary snapshot from the first available camera device
using v4l2py 3.0.0 (which delegates to linuxpy under the hood).
Maintains a rolling buffer of at most MAX_CAPTURES images on disk.
"""

import io
import os
import glob
import base64
import warnings
from datetime import datetime
from PIL import Image

# Suppress the "no longer maintained" warning — we know, it still works fine
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from v4l2py.device import Device, BufferType, PixelFormat

# ── Configuration ─────────────────────────────────────────────────────────────

CAPTURE_DIR  = "/tmp/agent_captures"
MAX_CAPTURES = 20
CAPTURE_WIDTH  = 640
CAPTURE_HEIGHT = 480
WARMUP_FRAMES  = 4          # discard frames so auto-exposure settles

_CAMERA_PATH: str | None = None   # cached after first successful detection

# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_camera() -> str | None:
    """Return the path of the first /dev/video* that can stream frames."""
    for path in sorted(glob.glob("/dev/video*")):
        try:
            with Device(path) as cam:
                cam.set_format(
                    BufferType.VIDEO_CAPTURE,
                    CAPTURE_WIDTH, CAPTURE_HEIGHT,
                    "MJPG",
                )
                # try to pull one frame to confirm it really works
                for frame in cam:
                    _ = frame.data
                    break
            print(f"   📷 Camera detected: {path}")
            return path
        except Exception:
            continue
    return None


def _evict_old_captures() -> None:
    """Delete oldest captures when the rolling buffer is full."""
    files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "capture_*.jpg")))
    excess = len(files) - MAX_CAPTURES + 1   # +1 makes room for the new one
    for path in files[:max(0, excess)]:
        try:
            os.remove(path)
        except OSError:
            pass


def _frame_to_pil(frame) -> Image.Image:
    """
    Convert a v4l2py Frame to a PIL Image.

    MJPEG frames are already valid JPEG bytes — just wrap them.
    YUYV frames need manual conversion via numpy.
    """
    pf = frame.pixel_format
    raw = frame.data

    if pf in (PixelFormat.MJPEG, PixelFormat.JPEG):
        return Image.open(io.BytesIO(raw))

    if pf == PixelFormat.YUYV:
        import numpy as np
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(
            frame.height, frame.width, 2
        )
        # YUYV → RGB via YCbCr
        y  = arr[:, :, 0].astype(np.float32)
        cb = arr[:, :, 1].astype(np.float32) - 128
        cr = arr[:, :, 1].astype(np.float32) - 128  # alternating; approximate
        r = np.clip(y + 1.402  * cr, 0, 255).astype(np.uint8)
        g = np.clip(y - 0.344  * cb - 0.714 * cr, 0, 255).astype(np.uint8)
        b = np.clip(y + 1.772  * cb, 0, 255).astype(np.uint8)
        rgb = np.stack([r, g, b], axis=2)
        return Image.fromarray(rgb, "RGB")

    raise RuntimeError(
        f"Unsupported pixel format: {pf.name}. "
        "Camera must support MJPEG or YUYV. "
        "Run `v4l2-ctl --list-formats-ext` to check."
    )


# ── Public API ────────────────────────────────────────────────────────────────

def capture_image_to_disk() -> str:
    """
    Capture one frame from the camera and save it as a JPEG.

    Returns:
        Absolute path to the saved JPEG file.

    Raises:
        RuntimeError: If no camera is found or the frame cannot be grabbed.
    """
    global _CAMERA_PATH
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    if _CAMERA_PATH is None:
        _CAMERA_PATH = _detect_camera()
    if _CAMERA_PATH is None:
        raise RuntimeError("No camera device found on this system.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with Device(_CAMERA_PATH) as cam:
                cam.set_format(
                    BufferType.VIDEO_CAPTURE,
                    CAPTURE_WIDTH, CAPTURE_HEIGHT,
                    "MJPG",
                )
                frame = None
                for i, f in enumerate(cam):
                    if i >= WARMUP_FRAMES:
                        frame = f
                        break

        if frame is None:
            raise RuntimeError("Camera stream ended before a frame was captured.")

        image = _frame_to_pil(frame)
        _evict_old_captures()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
        image.save(path, "JPEG", quality=85)
        print(f"   📷 Image saved: {path}  ({frame.width}x{frame.height} {frame.pixel_format.name})")
        return path

    except Exception as e:
        _CAMERA_PATH = None   # force re-detection next call
        raise RuntimeError(f"Camera capture failed: {e}") from e


def image_to_base64(path: str) -> str:
    """Return a base64-encoded JPEG string suitable for an LLM vision API."""
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")