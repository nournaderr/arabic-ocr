from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    """Load JPEG / PNG / BMP as BGR numpy array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def save_image(img: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def resize_if_large(img: np.ndarray, max_dim: int = 2000) -> np.ndarray:
    """Downscale proportionally so the larger dimension ≤ max_dim.

    Prevents memory issues on high-res phone photos while keeping aspect ratio.
    Returns the original array unchanged if already within limits.
    """
    h, w = img.shape[:2]
    larger = max(h, w)
    if larger <= max_dim:
        return img
    scale = max_dim / larger
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
