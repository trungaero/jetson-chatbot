import io
import os
import glob
import base64
from datetime import datetime
from v4l2py import Device
from PIL import Image

CAPTURE_DIR = "/tmp/agent_captures"
MAX_CAPTURES = 20
_CAMERA_PATH = None


def _detect_camera() -> str | None:
    for path in sorted(glob.glob("/dev/video*")):
        try:
            with Device.from_path(path) as cam:
                cam.video_capture.set_format(640, 480, "MJPG")
                # if it opens without error, it's usable
                print(f"   📷 Camera detected: {path}")
                return path
        except Exception:
            continue
    return None


def _evict_old_captures():
    files = sorted(glob.glob(os.path.join(CAPTURE_DIR, "capture_*.jpg")))
    for path in files[:max(0, len(files) - MAX_CAPTURES + 1)]:
        try:
            os.remove(path)
        except OSError:
            pass


def capture_image_to_disk() -> str:
    global _CAMERA_PATH
    os.makedirs(CAPTURE_DIR, exist_ok=True)

    if _CAMERA_PATH is None:
        _CAMERA_PATH = _detect_camera()
    if _CAMERA_PATH is None:
        raise RuntimeError("No camera device found on this system.")

    try:
        with Device.from_path(_CAMERA_PATH) as cam:
            cam.video_capture.set_format(640, 480, "MJPG")
            # discard a few frames so auto-exposure settles
            for i, frame in enumerate(cam):
                if i >= 4:
                    jpeg_bytes = bytes(frame)
                    break

        # v4l2py in MJPG mode gives raw JPEG bytes — verify and re-encode
        image = Image.open(io.BytesIO(jpeg_bytes))
        _evict_old_captures()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(CAPTURE_DIR, f"capture_{timestamp}.jpg")
        image.save(path, "JPEG", quality=85)
        print(f"   📷 Image saved: {path}")
        return path

    except Exception as e:
        _CAMERA_PATH = None  # force re-detection next time
        raise RuntimeError(f"Camera capture failed: {e}") from e


def image_to_base64(path: str) -> str:
    with open(path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")