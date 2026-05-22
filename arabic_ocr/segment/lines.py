import numpy as np
from scipy.ndimage import gaussian_filter1d

from arabic_ocr.config import AH_HEIGHT_MIN


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

    # Estimate average character height from median run-length of non-zero rows
    nonzero_runs = _run_lengths(smoothed > smoothed.max() * 0.05)
    estimated_ah = float(np.median(nonzero_runs)) if len(nonzero_runs) else 20.0

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

    return lines  # already top-to-bottom


def _run_lengths(mask: np.ndarray) -> np.ndarray:
    """Return lengths of consecutive True runs in a 1-D boolean array."""
    if not mask.any():
        return np.array([], dtype=float)
    padded = np.concatenate(([False], mask, [False]))
    starts = np.where(~padded[:-1] & padded[1:])[0]
    ends   = np.where(padded[:-1] & ~padded[1:])[0]
    return (ends - starts).astype(float)
