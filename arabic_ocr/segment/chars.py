import numpy as np
from scipy.signal import argrelmin
from itertools import combinations

from arabic_ocr.config import (
    AH_HEIGHT_MIN, AH_HEIGHT_MAX,
    MIN_CHAR_WIDTH, MAX_CHAR_WIDTH, AH_AREA_MIN,
    CHOP_MIN_VALLEY,
)


def segment_chars(
    paw_binary: np.ndarray,
    paw_x: int,
    paw_y: int,
    ah: float,
) -> list[tuple[int, int, int, int]]:
    """Over-segment a PAW into individual character bounding boxes.

    Algorithm: column projection valley-finding + trellis validation
    (Abandah & Khedher 2014).

    Returns list of (abs_x1, abs_y1, abs_x2, abs_y2) sorted RIGHT-TO-LEFT.

    Positional form note
    --------------------
    The returned list is sorted descending by x1 (RTL order).  The caller in
    segment/__init__.py derives each character's positional tag from its index
    in this list and stores it in CharCrop.position:
        index 0             → "initial"  (rightmost  = word-start in RTL)
        index 1 … n-2       → "medial"
        index n-1           → "final"    (leftmost   = word-end   in RTL)
        single character    → "isolated"

    These tags map directly to HMDB label suffixes via HMDB_POSITION_MAP:
        "isolated" → "_Isolated"
        "initial"  → "_Start"
        "medial"   → "_Middle"
        "final"    → "_End"

    At inference time the pipeline calls filter_candidates_by_position() to
    restrict the top-K classifier output to the matching HMDB position class,
    giving a free accuracy boost when the model is trained on Option-A labels.
    """
    col_proj = np.sum(paw_binary == 0, axis=0).astype(float)
    h, w = paw_binary.shape

    if w < 2 or col_proj.max() == 0:
        return [(paw_x, paw_y, paw_x + w, paw_y + h)]

    # Local minima in column projection
    order = max(1, int(0.05 * w))
    min_indices = list(argrelmin(col_proj, order=order)[0])

    local_max = np.percentile(col_proj[col_proj > 0], 90) if col_proj.any() else 1.0
    valleys = [
        idx for idx in min_indices
        if col_proj[idx] < CHOP_MIN_VALLEY * local_max
    ]

    if not valleys:
        return [(paw_x, paw_y, paw_x + w, paw_y + h)]

    # All possible cut points (including start=0 and end=w)
    cut_candidates = sorted(set([0] + valleys + [w]))

    best_cuts = _best_segmentation(
        paw_binary, cut_candidates, ah,
        paw_h=h, paw_w=w
    )

    result = []
    for c1, c2 in zip(best_cuts[:-1], best_cuts[1:]):
        if c2 - c1 < 1:
            continue
        char_crop = paw_binary[:, c1:c2]
        char_h, char_w = char_crop.shape
        result.append((
            paw_x + c1, paw_y,
            paw_x + c2, paw_y + char_h,
        ))

    # Sort right-to-left
    result.sort(key=lambda t: t[0], reverse=True)
    return result if result else [(paw_x, paw_y, paw_x + w, paw_y + h)]


def _valid_char(crop: np.ndarray, ah: float) -> bool:
    """Check if a crop is plausibly a single Arabic character.

    Uses tight bounding-box height (rows containing black pixels) instead of
    the raw crop height, which equals the full line-band height and is always
    larger than ah — causing every cut candidate to fail the height check.
    """
    if not np.any(crop == 0):
        return False
    # Tight height: count rows that actually contain ink
    tight_h = int(np.sum(np.any(crop == 0, axis=1)))
    _, w = crop.shape
    area = int(np.sum(crop == 0))
    return (
        AH_HEIGHT_MIN * ah <= tight_h <= AH_HEIGHT_MAX * ah and
        MIN_CHAR_WIDTH * ah <= w <= MAX_CHAR_WIDTH * ah and
        area >= AH_AREA_MIN * ah * ah
    )


def _best_segmentation(
    paw_binary: np.ndarray,
    cuts: list[int],
    ah: float,
    paw_h: int,
    paw_w: int,
) -> list[int]:
    """Choose the cut set that maximises valid character segments."""
    best_score = -1
    best_cuts = [0, paw_w]

    # Enumerate all subsets of internal cuts; limit search for performance
    internal = cuts[1:-1]
    if len(internal) > 10:
        internal = internal[:10]

    for r in range(len(internal) + 1):
        for subset in combinations(range(len(internal)), r):
            chosen = [0] + [internal[i] for i in subset] + [paw_w]
            score = sum(
                1 for a, b in zip(chosen[:-1], chosen[1:])
                if _valid_char(paw_binary[:, a:b], ah)
            )
            # Prefer higher score; break ties by choosing more segments
        # (more cuts = better for multi-character PAWs where some segments
        # are valid and others are not yet correctly sized).
        if score > best_score or (
            score == best_score and len(chosen) > len(best_cuts)
        ):
                best_score = score
                best_cuts = chosen

    return best_cuts


def _estimate_ah(line_binary: np.ndarray) -> float:
    """Estimate average character height as 70th percentile of CC heights."""
    import cv2
    inverted = cv2.bitwise_not(line_binary)
    _, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if stats.shape[0] <= 1:
        return 20.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    return float(np.percentile(heights, 70))
