import os
import numpy as np
import cv2
import tifffile
import imageio.v3 as iio
from src.core.image_data import ImageData

SUPPORTED_EXTENSIONS = {
    ".tiff", ".tif", ".png", ".bmp", ".jpg", ".jpeg",
    ".pgm", ".ppm", ".pfm", ".exr", ".hdr",
}


def load_image(path: str) -> ImageData:
    """Load any supported image format into ImageData. Preserves 16-bit."""
    data = ImageData()
    data.path = path
    data.filename = os.path.basename(path)
    ext = os.path.splitext(path)[1].lower()

    raw = _read_raw(path, ext)
    if raw is None:
        raise ValueError(f"Could not load image: {path}")

    data.raw = raw
    data.height, data.width = raw.shape[:2]
    data.channels = 1 if raw.ndim == 2 else raw.shape[2]
    data.bit_depth = 16 if raw.dtype == np.uint16 else 8
    data.display = raw.copy()
    data.metadata = _read_metadata(path, ext)
    return data


def _read_raw(path: str, ext: str) -> np.ndarray | None:
    # TIFF — use tifffile for full 16-bit support
    if ext in (".tiff", ".tif"):
        try:
            arr = tifffile.imread(path)
            return _normalize_shape(arr)
        except Exception:
            pass

    # EXR / HDR — use imageio
    if ext in (".exr", ".hdr", ".pfm"):
        try:
            arr = iio.imread(path)
            # Convert float to uint16
            arr = (np.clip(arr, 0, 1) * 65535).astype(np.uint16)
            return _normalize_shape(arr)
        except Exception:
            pass

    # Everything else — OpenCV (handles 16-bit PNG, BMP, PGM)
    arr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        return None
    if arr.ndim == 3 and arr.shape[2] == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGRA2RGBA)
    return arr


def _normalize_shape(arr: np.ndarray) -> np.ndarray:
    """Ensure array is HxW or HxWxC with max 4 channels."""
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        c = arr.shape[2]
        if c in (1, 2, 3, 4):
            return arr
        # Multi-channel TIFF (e.g. 6-channel) — keep all, caller handles
        return arr
    if arr.ndim == 4:
        # Some TIFFs: frames x H x W x C — take first frame
        return arr[0]
    return arr


def _read_metadata(path: str, ext: str) -> dict:
    meta = {}
    try:
        stat = os.stat(path)
        meta["file_size_kb"] = round(stat.st_size / 1024, 1)
        meta["modified"] = stat.st_mtime
    except Exception:
        pass
    if ext in (".tiff", ".tif"):
        try:
            with tifffile.TiffFile(path) as tf:
                if tf.pages:
                    page = tf.pages[0]
                    meta["tiff_compression"] = str(page.compression)
                    meta["tiff_photometric"] = str(page.photometric)
        except Exception:
            pass
    return meta


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def list_images_in_folder(folder: str) -> list[str]:
    """Return sorted list of supported image paths in folder."""
    paths = []
    try:
        for name in sorted(os.listdir(folder)):
            full = os.path.join(folder, name)
            if os.path.isfile(full) and is_supported(full):
                paths.append(full)
    except Exception:
        pass
    return paths
