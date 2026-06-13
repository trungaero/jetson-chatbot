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
import requests as _http
from PIL import Image
import config

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


def capture_image_b64() -> dict:
    """Capture an image and return its path, base64 string, and timestamp.
    Returns:
        A dict with:
          - "image_path": path to the saved JPEG on disk
          - "base64_jpeg": base64-encoded JPEG string ready for a vision model
          - "timestamp": ISO-format capture timestamp
    """
    try:
        path = capture_image_to_disk()
        b64 = image_to_base64(path)
        from datetime import datetime as _dt
        return {
            "image_path": path,
            "base64_jpeg": b64,
            "timestamp": _dt.now().isoformat(timespec="seconds"),
        }
    except Exception as e:
        return {
            "image_path": None,
            "base64_jpeg": None,
            "timestamp": None,
            "error": str(e),
        }


def describe_image_with_local_llm(base64_jpeg: str, user_question: str = "") -> str:
    """
    Send a base64 JPEG to the local llama-server (gemma4-e2b) for visual
    description using the OpenAI multimodal image_url format.

    llama-server accepts images as:
        { "type": "image_url",
          "image_url": { "url": "data:image/jpeg;base64,<data>" } }

    Args:
        base64_jpeg: Base64-encoded JPEG string (no data-URI prefix).
        user_question: Optional context from the user's original query.

    Returns:
        A plain-text description of the image, or an error message.
    """
    data_uri = f"data:image/jpeg;base64,{base64_jpeg}"

    prompt_text = (
        f"The user asked: \"{user_question}\"\nDescribe what you see in this image."
        if user_question
        else "Describe what you see in this image."
    )

    payload = {
        "model": config.LLAMA_MODEL,
        "max_tokens": config.VISION_MAX_TOKENS,
        "temperature": 0.5,
        "stream": False,
        "messages": [
            {"role": "system", "content": config.VISION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                    {"type": "text", "text": prompt_text},
                ],
            },
        ],
    }

    try:
        resp = _http.post(
            f"{config.LLAMA_SERVER_URL}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text or "I could not generate a description."
    except _http.exceptions.Timeout:
        return "Vision request timed out. Please try again."
    except _http.exceptions.ConnectionError:
        return (
            f"Could not reach the llama-server at {config.LLAMA_SERVER_URL}. "
            "Make sure it is running."
        )
    except _http.exceptions.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:300]
        except Exception:
            pass
        return f"llama-server vision error ({e}): {body}"
    except (KeyError, IndexError) as e:
        return f"Unexpected response format from llama-server: {e}"
    except Exception as e:
        return f"Unexpected error during vision analysis: {e}"
