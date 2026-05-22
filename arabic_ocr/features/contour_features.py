import cv2
import numpy as np

from arabic_ocr.config import OUTLINE_SAMPLES


def contour_features(norm_img: np.ndarray) -> np.ndarray:
    """192-d contour feature: sampled (x, y, tangent) along the outer boundary.

    Equidistant sampling of OUTLINE_SAMPLES=64 points on the external contour.
    Arabic relevance: captures distinctive loops (ع) and tails (ر).
    """
    inverted = cv2.bitwise_not(norm_img)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    size = float(norm_img.shape[0])  # normalisation denominator

    if not contours:
        return np.zeros(OUTLINE_SAMPLES * 3, dtype=np.float32)

    # Use longest contour
    contour = max(contours, key=len).reshape(-1, 2).astype(float)
    n = len(contour)

    if n < 2:
        return np.zeros(OUTLINE_SAMPLES * 3, dtype=np.float32)

    # Compute cumulative arc-length for equidistant sampling
    diffs = np.diff(contour, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum_len = np.concatenate(([0.0], np.cumsum(seg_lens)))
    total_len = cum_len[-1]

    if total_len == 0:
        return np.zeros(OUTLINE_SAMPLES * 3, dtype=np.float32)

    sample_dists = np.linspace(0, total_len, OUTLINE_SAMPLES, endpoint=False)
    indices = np.searchsorted(cum_len, sample_dists, side="right") - 1
    indices = np.clip(indices, 0, n - 1)

    pts = contour[indices]          # (OUTLINE_SAMPLES, 2)
    next_idx = (indices + 1) % n
    next_pts = contour[next_idx]

    tangent_angles = np.arctan2(
        next_pts[:, 1] - pts[:, 1],
        next_pts[:, 0] - pts[:, 0],
    )
    tangent_norm = (tangent_angles + np.pi) / (2 * np.pi)  # [0, 1]

    feat = np.stack([
        pts[:, 0] / size,
        pts[:, 1] / size,
        tangent_norm,
    ], axis=1).reshape(-1).astype(np.float32)

    return feat
