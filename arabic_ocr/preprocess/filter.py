import cv2
import numpy as np


def filter_noise(binary: np.ndarray) -> np.ndarray:
    """Remove noise via connected-component analysis.

    Drops components that are:
    - single-pixel specks (area < 3) — always scanner noise
    - much smaller than the smallest plausible Arabic dot:
        an Arabic dot is roughly (ah/6)² ≈ 1–2% of a typical letter body.
        We estimate a safe floor as 1% of the median body area, but cap it
        at 3 px so that true tiny-image scenarios don't erase real marks.
    - extreme ruling lines: aspect ratio > 25 (relaxed from 20 to avoid
        accidentally removing ا alef or ل lam strokes at border angles)

    We deliberately do NOT use 0.5% of median as a hard threshold because
    on a page with large headline letters the median is dominated by those
    bodies, making the threshold too high and erasing Arabic dots (نقاط).
    """
    inverted = cv2.bitwise_not(binary)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)

    if num_labels <= 1:
        return binary.copy()

    areas = stats[1:, cv2.CC_STAT_AREA]  # skip background label 0
    if len(areas) == 0:
        return binary.copy()

    median_area = float(np.median(areas))
    # 1% of median — generous enough to keep dots even when headline text
    # inflates the median.  Absolute floor of 3 px catches sub-pixel speckles
    # regardless of image scale.
    area_threshold = max(3.0, 0.01 * median_area)

    cleaned = np.zeros_like(inverted)
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        w    = stats[label, cv2.CC_STAT_WIDTH]
        h    = stats[label, cv2.CC_STAT_HEIGHT]

        if area < area_threshold:
            continue
        # Aspect ratio > 25 catches scanner ruling lines / horizontal borders
        # without removing tall narrow letters (ا) or wide ones (و at angle).
        aspect = max(w, h) / max(min(w, h), 1)
        if aspect > 25:
            continue

        cleaned[labels == label] = 255

    return cv2.bitwise_not(cleaned)
