from pathlib import Path

import cv2
import numpy as np


def crop_region(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    padding: int = 2,
) -> np.ndarray:
    """Crop a rectangle from img with optional padding and bounds checking."""
    h, w = img.shape[:2]
    y1c = max(0, y1 - padding)
    y2c = min(h, y2 + padding)
    x1c = max(0, x1 - padding)
    x2c = min(w, x2 + padding)
    return img[y1c:y2c, x1c:x2c]


def save_crops(
    crops: list[np.ndarray],
    directory: str | Path,
    prefix: str = "crop",
) -> None:
    """Save a list of image crops as numbered PNG files for debugging."""
    out_dir = Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, crop in enumerate(crops):
        cv2.imwrite(str(out_dir / f"{prefix}_{idx:04d}.png"), crop)
