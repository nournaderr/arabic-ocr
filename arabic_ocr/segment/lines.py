import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d

from arabic_ocr.config import AH_HEIGHT_MIN


def _estimate_ah_from_ccs(binary: np.ndarray) -> float:
    """Estimate average character height from connected component stats.

    Uses 70th-percentile CC height, matching the estimator in chars.py.
    This is far more accurate than measuring projection-profile run lengths,
    which yield line-band heights (2–4× larger than true character height).
    """
    inv = cv2.bitwise_not(binary)
    _, _, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=8)
    if stats.shape[0] <= 1:
        return 20.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    return float(np.percentile(heights, 70))


def segment_lines(
    binary: np.ndarray,
) -> list[tuple[int, int, np.ndarray]]:
    """Split binary page image into horizontal line bands.

    Algorithm: horizontal projection profile with Gaussian smoothing
    (Lorigo & Govindaraju 2006).

    Returns list of (y1, y2, line_crop) sorted top-to-bottom.
    """
    row_proj = np.sum(binary == 0, axis=1).astype(float)
    smoothed = gaussian_filter1d(row_proj, sigma=2)

    estimated_ah = _estimate_ah_from_ccs(binary)

    threshold = smoothed.max() * 0.10
    in_line = smoothed > threshold

    lines = []
    i = 0
    n = len(in_line)
    while i < n:
        if in_line[i]:
            j = i
            while j < n and in_line[j]:
                j += 1
            y1, y2 = i, j
            if (y2 - y1) >= AH_HEIGHT_MIN * estimated_ah:
                crop = binary[y1:y2, :]
                lines.append((y1, y2, crop))
            i = j
        else:
            i += 1

    # Merge bands separated by a small gap — a slight dip in the projection
    # (e.g. at a lam-alef junction) can split one text line into two bands.
    lines = _merge_close_lines(lines, binary, gap_threshold=int(0.5 * estimated_ah))

    return lines  # already top-to-bottom


def _merge_close_lines(
    lines: list[tuple[int, int, np.ndarray]],
    binary: np.ndarray,
    gap_threshold: int,
) -> list[tuple[int, int, np.ndarray]]:
    if not lines:
        return lines
    merged = [lines[0]]
    for y1, y2, _ in lines[1:]:
        prev_y1, prev_y2, _ = merged[-1]
        if (y1 - prev_y2) <= gap_threshold:
            merged[-1] = (prev_y1, y2, binary[prev_y1:y2, :])
        else:
            merged.append((y1, y2, binary[y1:y2, :]))
    return merged


