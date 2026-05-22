import cv2
import numpy as np
from skimage.filters import threshold_sauvola

from arabic_ocr.config import SAUVOLA_WINDOW, MORPH_KERNEL


def binarize(gray: np.ndarray) -> np.ndarray:
    """Sauvola adaptive threshold + morphological opening.

    Returns uint8 binary with white background (255) and black text (0).
    Paper: Sauvola & Pietikäinen, Pattern Recognition 2000.

    The Sauvola window is scaled to ~1/8 of the shorter image dimension so
    that small input images (e.g. 100px tall) don't over-binarize.  If the
    result still has >50% black pixels, Otsu global threshold is used instead.
    """
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    # Scale window to image size; ensure odd, clamp to [11, SAUVOLA_WINDOW]
    win = max(11, min(SAUVOLA_WINDOW, min(h, w) // 8))
    win = win | 1  # make odd (required by skimage)

    # k=0.15 (reduced from 0.2): lower k is more lenient — keeps faint strokes
    # that appear in aged/scanned documents where ink has faded or bled.
    thresh = threshold_sauvola(blurred, window_size=win, k=0.15)
    binary = (blurred < thresh).astype(np.uint8) * 255  # black text on white

    # Otsu fallback: Sauvola can over-binarize dark/noisy background images
    if np.mean(binary == 0) > 0.50:
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # cv2 THRESH_BINARY: pixel > thresh → 255 (background), else → 0 (text) ✓
        # If the image is inverted (light text on dark bg), flip
        if np.mean(binary == 0) > 0.50:
            binary = cv2.bitwise_not(binary)

    kernel = np.ones((MORPH_KERNEL, MORPH_KERNEL), np.uint8)
    # Opening removes isolated noise pixels.
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    # Closing reconnects short stroke gaps caused by faded ink or scan artefacts.
    # Arabic letters have thin connecting strokes (e.g. ـبـ medial baseline) that
    # a scan at low contrast can break; a 1-px closing seals most of these gaps
    # without merging distinct characters.
    close_kernel = np.ones((2, 2), np.uint8)
    return cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel)
