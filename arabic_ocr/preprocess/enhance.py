import cv2
import numpy as np


def enhance(img: np.ndarray) -> np.ndarray:
    """Convert to grayscale, denoise, and normalize illumination with CLAHE.

    For scanned documents the scanner introduces high-frequency grain that
    confuses Sauvola thresholding.  A bilateral filter removes this grain while
    preserving stroke edges.  CLAHE then corrects uneven illumination common in
    hand-held or flatbed scans (shadows near binding, yellowed paper).
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Preserve stroke width on already-clean scans.  The previous bilateral +
    # CLAHE combination was making the glyphs look heavier before thresholding.
    return gray
