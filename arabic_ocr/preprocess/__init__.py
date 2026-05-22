import cv2
import numpy as np

import arabic_ocr.config as _cfg
from .enhance import enhance
from .binarize import binarize
from .deskew import deskew
from .filter import filter_noise


_MIN_HEIGHT = 400  # minimum image height for reliable OCR segmentation


def _upscale(img: np.ndarray) -> np.ndarray:
    """Upscale small images to at least _MIN_HEIGHT pixels tall.

    Low-resolution inputs (< 150 DPI equivalent) produce line crops that are
    only 10–15 px tall, which is too thin for column-projection valley finding.
    INTER_CUBIC preserves text stroke edges better than nearest-neighbour.
    """
    h = img.shape[0]
    if h >= _MIN_HEIGHT:
        return img
    scale = _MIN_HEIGHT / h
    new_w = int(img.shape[1] * scale)
    return cv2.resize(img, (new_w, _MIN_HEIGHT), interpolation=cv2.INTER_CUBIC)


def preprocess(img: np.ndarray, frame_number: int = 0) -> np.ndarray:
    """Full preprocessing pipeline: upscale → enhance → binarize → deskew → filter → pad.

    Returns a padded binary (uint8, white background) ready for segmentation.
    """
    img    = _upscale(img)
    gray   = enhance(img)
    binary = binarize(gray)
    clean  = filter_noise(binary)
    deskewed, angle = deskew(clean)

    # 5% padding on each side (20% was adding too much blank space)
    h, w = deskewed.shape
    pad_h = int(h * 0.05)
    pad_w = int(w * 0.05)
    padded = np.pad(deskewed, ((pad_h, pad_h), (pad_w, pad_w)), constant_values=255)

    if _cfg.DEBUG:
        _cfg.PREPROCESS_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_gray.png"),   gray)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_binary.png"), binary)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_clean.png"),  clean)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_deskew.png"), deskewed)
        cv2.imwrite(str(_cfg.PREPROCESS_DIR / f"{frame_number:04d}_padded.png"), padded)

    return padded


__all__ = ["preprocess"]
