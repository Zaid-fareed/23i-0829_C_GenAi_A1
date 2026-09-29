"""Upload validation, preprocessing and encoding helpers."""
import base64
import io

import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

SIZE = 128
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 40_000_000
ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


def decode_upload(data: bytes, content_type: str | None) -> Image.Image:
    """Validate an uploaded image (type, size, decodability) and return an RGB PIL image."""
    if content_type and content_type.lower() not in ALLOWED:
        raise HTTPException(415, f"Unsupported file type '{content_type}'. Use JPEG, PNG, WebP or BMP.")
    if not data:
        raise HTTPException(400, "Empty file.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large ({len(data) / 1e6:.1f} MB); limit is {MAX_BYTES // 1_000_000} MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        im = Image.open(io.BytesIO(data))
        if im.width * im.height > MAX_PIXELS:
            raise HTTPException(413, "Image resolution too large.")
        return im.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise HTTPException(400, "File is not a valid image.")


def to_tensor(im: Image.Image) -> np.ndarray:
    """PIL -> float32 (3,128,128) in [0,1], direct bicubic resize (identical to the training pipeline)."""
    a = np.asarray(im.resize((SIZE, SIZE), Image.Resampling.BICUBIC), dtype=np.float32) / 255.0
    return np.ascontiguousarray(a.transpose(2, 0, 1))


def to_data_url(arr: np.ndarray) -> str:
    """(3,H,W) or (H,W) float in [0,1] -> PNG data URL."""
    a = np.clip(arr, 0, 1)
    a = (a.transpose(1, 2, 0) if a.ndim == 3 else a) * 255.0 + 0.5
    buf = io.BytesIO()
    Image.fromarray(a.astype(np.uint8)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a - b) ** 2))
    return float(10 * np.log10(1.0 / max(mse, 1e-10)))


def error_map(restored: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Mean-over-channels absolute error, scaled so 0.5 error = white."""
    return np.clip(np.abs(restored - target).mean(0) / 0.5, 0, 1)
