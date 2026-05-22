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

    # Bilateral filter: removes scanner grain, preserves stroke edges.
    # d=9 neighbourhood, sigmaColor/Space=75 — safe defaults for 300 dpi scans.
    denoised = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # clipLimit=3.0 (up from 2.0): scanned docs have stronger illumination
    # gradients than photos, so we allow more local contrast amplification.
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(denoised)
