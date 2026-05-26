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

    # Try a few Sauvola k values and keep the first result that does not
    # produce an overly dense foreground mask. This adapts the thresholding
    # per page without hard-coding a single setting for all documents.
    binary = None
    for k in (0.24, 0.32, 0.40, 0.48):
        thresh = threshold_sauvola(blurred, window_size=win, k=k)
        candidate = (blurred < thresh).astype(np.uint8) * 255  # black text on white
        binary = candidate
        if np.mean(candidate == 0) <= 0.10:
            break

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

    # Option A: thin strokes slightly by a 1-px erosion, but restore very
    # small connected components (likely dots) from the original binary so
    # we don't lose diacritics. This usually separates close glyphs without
    # harming dot detection.
    erode_k = np.ones((2, 2), np.uint8)
    eroded = cv2.erode(opened, erode_k, iterations=1)

    # Restore small components from the pre-opened binary (use inverted CCs)
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if num_labels > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        median_area = float(np.median(areas)) if len(areas) > 0 else 3.0
        restore_thresh = max(1, int(0.02 * median_area))
        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            if area <= restore_thresh:
                # labels==lab corresponds to foreground in inverted; set those
                # pixels back to black (0) in the eroded image.
                eroded[labels == lab] = 0

    return eroded
