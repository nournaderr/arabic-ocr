import logging
import numpy as np

from arabic_ocr.config import (
    AH_HEIGHT_MIN, AH_HEIGHT_MAX,
    MIN_CHAR_WIDTH, MAX_CHAR_WIDTH, AH_AREA_MIN,
    CHOP_MIN_VALLEY,
)

logger = logging.getLogger(__name__)


def _segment_chars_impl(
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
    try:
        col_proj = np.sum(paw_binary == 0, axis=0).astype(float)
        h, w = paw_binary.shape

        if w < 2 or col_proj.max() == 0:
            return [(paw_x, paw_y, paw_x + w, paw_y + h)]

        # Find valleys as midpoints of contiguous low-ink regions.
        # argrelmin requires a strict local minimum and misses plateau connections
        # (multiple equal-value columns at a tatweel junction) — a common pattern
        # in printed Naskh where argrelmin finds no valley even when the ink drops
        # to near-zero across several adjacent columns.
        local_max = np.percentile(col_proj[col_proj > 0], 90) if col_proj.any() else 1.0
        threshold = CHOP_MIN_VALLEY * local_max
        # Minimum gap width: scale with average character height (ah).
        # Use AH-based sizing so the same code works across DPIs and scans.
        # Allow very small floors for tiny images but prefer a fraction of ah.
        min_gap_w = max(1, int(round(0.06 * ah)))

        valleys: list[int] = []
        in_gap = False
        gap_start = 0
        for i, v in enumerate(col_proj):
            if v < threshold and not in_gap:
                gap_start = i
                in_gap = True
            elif v >= threshold and in_gap:
                if (i - gap_start) >= min_gap_w:
                    valleys.append((gap_start + i - 1) // 2)
                in_gap = False
        if in_gap and (w - gap_start) >= min_gap_w:
            valleys.append((gap_start + w - 1) // 2)

        if not valleys:
            return [(paw_x, paw_y, paw_x + w, paw_y + h)]

        # All possible cut points (including start=0 and end=w)
        cut_candidates = sorted(set([0] + valleys + [w]))

        try:
            best_cuts = _best_segmentation(
                paw_binary, cut_candidates, ah,
                paw_h=h, paw_w=w
            )
        except Exception:
            logger.exception("_best_segmentation failed, returning full PAW as one character")
            return [(paw_x, paw_y, paw_x + w, paw_y + h)]

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
        if not result:
            return [(paw_x, paw_y, paw_x + w, paw_y + h)]

        # Merge any accidentally tiny slivers into neighbours.
        # Use a threshold relative to average character height so behavior adapts to DPI/font size.
        # Choose a modest minimum character width based on AH (not a fixed pixel floor).
        min_char_w = max(2, int(round(0.25 * ah)))
        sliver_thresh = min_char_w
        cleaned: list[tuple[int, int, int, int]] = []
        for idx, box in enumerate(result):
            x1, y1, x2, y2 = box
            w_box = x2 - x1
            if w_box <= sliver_thresh:
                # merge small sliver into previous if exists, else into next
                if cleaned:
                    px1, py1, px2, py2 = cleaned[-1]
                    cleaned[-1] = (px1, py1, x2, py2)
                else:
                    if idx + 1 < len(result):
                        nx1, ny1, nx2, ny2 = result[idx + 1]
                        cleaned.append((x1, y1, nx2, y2))
                    else:
                        cleaned.append(box)
            else:
                cleaned.append(box)

        # If some segments look invalid, attempt to repair by merging adjacent
        # segments and re-checking validity. This handles over-segmentation.
        def all_valid(segs):
            for sx1, sy1, sx2, sy2 in segs:
                crop = paw_binary[:, sx1 - paw_x:sx2 - paw_x]
                if not _valid_char(crop, ah):
                    return False
            return True

        repaired = list(cleaned)
        max_iters = len(repaired) * 2
        it = 0
        while it < max_iters and not all_valid(repaired) and len(repaired) > 1:
            # find first invalid and merge with neighbor that yields larger area
            for i, (sx1, sy1, sx2, sy2) in enumerate(repaired):
                crop = paw_binary[:, sx1 - paw_x:sx2 - paw_x]
                if not _valid_char(crop, ah):
                    # prefer merging with previous if exists, else next
                    if i > 0:
                        px1, py1, px2, py2 = repaired[i - 1]
                        repaired[i - 1] = (px1, py1, sx2, py2)
                        repaired.pop(i)
                    elif i + 1 < len(repaired):
                        nx1, ny1, nx2, ny2 = repaired[i + 1]
                        repaired[i + 1] = (sx1, sy1, nx2, ny2)
                        repaired.pop(i)
                    break
            it += 1

        # If we still have too many fragments, merge the smallest gaps until the
        # PAW has a reasonable number of character boxes. This avoids returning
        # one box per stroke cluster on dense printed text.
        max_segments = max(2, int(round(w / max(1.0, 0.9 * ah))))
        if len(repaired) > max_segments:
            repaired = sorted(repaired, key=lambda t: t[0])
            while len(repaired) > max_segments:
                gap_info = []
                for i in range(len(repaired) - 1):
                    left = repaired[i]
                    right = repaired[i + 1]
                    gap = right[0] - left[2]
                    gap_info.append((gap, i))
                _, merge_idx = min(gap_info, key=lambda t: t[0])
                lx1, ly1, lx2, ly2 = repaired[merge_idx]
                rx1, ry1, rx2, ry2 = repaired[merge_idx + 1]
                repaired[merge_idx] = (
                    lx1,
                    min(ly1, ry1),
                    rx2,
                    max(ly2, ry2),
                )
                repaired.pop(merge_idx + 1)
            repaired.sort(key=lambda t: t[0], reverse=True)

        # Heuristic: if many returned segments are tiny (width << AH), merge
        # adjacent tiny ones until the fraction of tiny segments is reasonable.
        tiny_thresh = max(1, int(round(0.25 * ah)))
        tiny_segs = [1 for x1, y1, x2, y2 in repaired if (x2 - x1) <= tiny_thresh]
        # Be more aggressive: allow at most 25% tiny segments before merging.
        TINY_FRACTION_LIMIT = 0.25
        if repaired and (sum(tiny_segs) / len(repaired)) > TINY_FRACTION_LIMIT:
            # Repeat merging passes until tiny fraction is under the limit or
            # only one segment remains. Prefer merging adjacent tiny segments
            # first, then merge nearest neighbors by smallest gap.
            merged = list(sorted(repaired, key=lambda t: t[0]))
            iter_limit = len(merged) * 3
            iters = 0
            while iters < iter_limit:
                iters += 1
                tiny_count = sum(1 for a, b, c, d in merged if (c - a) <= tiny_thresh)
                if tiny_count / len(merged) <= TINY_FRACTION_LIMIT or len(merged) <= 1:
                    break
                # First pass: merge any adjacent pair where at least one is tiny
                did_merge = False
                i = 0
                while i < len(merged) - 1:
                    a1, b1, c1, d1 = merged[i]
                    a2, b2, c2, d2 = merged[i + 1]
                    w1 = c1 - a1
                    w2 = c2 - a2
                    if w1 <= tiny_thresh or w2 <= tiny_thresh:
                        merged[i] = (a1, min(b1, b2), c2, max(d1, d2))
                        merged.pop(i + 1)
                        did_merge = True
                        # don't advance index to consider new neighbour
                    else:
                        i += 1
                if did_merge:
                    continue
                # Second pass: merge the pair with smallest gap
                if len(merged) > 1:
                    gap_info = []
                    for i in range(len(merged) - 1):
                        left = merged[i]
                        right = merged[i + 1]
                        gap = right[0] - left[2]
                        gap_info.append((gap, i))
                    _, merge_idx = min(gap_info, key=lambda t: t[0])
                    lx1, ly1, lx2, ly2 = merged[merge_idx]
                    rx1, ry1, rx2, ry2 = merged[merge_idx + 1]
                    merged[merge_idx] = (
                        lx1,
                        min(ly1, ry1),
                        rx2,
                        max(ly2, ry2),
                    )
                    merged.pop(merge_idx + 1)
                else:
                    break
            repaired = merged

        if all_valid(repaired):
            return repaired

        # If repair failed, fall back to whole PAW
        return [(paw_x, paw_y, paw_x + w, paw_y + h)]

    except Exception:
        logger.exception("segment_chars failed, returning full PAW")
        h, w = paw_binary.shape
        return [(paw_x, paw_y, paw_x + w, paw_y + h)]


def segment_chars(
    paw_binary: np.ndarray,
    paw_x: int,
    paw_y: int,
    ah: float,
) -> list[tuple[int, int, int, int]]:
    """Public wrapper that currently delegates to the implementation.

    This wrapper exists so callers can be left unchanged while we add
    `segment_chars_with_fallback` which also uses `_segment_chars_impl`.
    """
    return _segment_chars_impl(paw_binary, paw_x, paw_y, ah)


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
    """Choose the cut set that maximises valid character segments (DP, O(n²)).

    Unlike the previous exhaustive 2^n search capped at 10 internal cuts, this
    DP evaluates every pair of cut positions in O(n²) time and is therefore not
    limited by the number of valley candidates.
    """
    n = len(cuts)
    if n <= 2:
        return cuts

    # dp[i] = (best_valid_count, n_segments_so_far, prev_cut_index)
    # Initialise with sentinel (-1, 0, -1) meaning "not reachable".
    dp: list[tuple[int, int, int]] = [(-1, 0, -1)] * n
    dp[0] = (0, 0, -1)

    for i in range(1, n):
        for j in range(i):
            if dp[j][0] < 0:
                continue
            valid = _valid_char(paw_binary[:, cuts[j]:cuts[i]], ah)
            score = dp[j][0] + (1 if valid else 0)
            segs  = dp[j][1] + 1
            # Primary: maximise valid-char count. Secondary: discourage many segments
            # by penalising the number of segments in the effective score. This
            # helps avoid over-segmentation when several low-quality cuts exist.
            # Compute an effective score for comparison.
            effective = score * 100 - segs * 12
            current_effective = dp[i][0] * 100 - dp[i][1] * 12 if dp[i][0] >= 0 else -10_000
            if effective > current_effective:
                dp[i] = (score, segs, j)

    # If no valid character was ever found, return the whole PAW unsplit.
    if dp[n - 1][0] <= 0:
        return [0, paw_w]

    # Backtrack through the DP table.
    path: list[int] = []
    i = n - 1
    while i >= 0:
        path.append(cuts[i])
        i = dp[i][2]

    return sorted(path)


def _estimate_ah(line_binary: np.ndarray) -> float:
    """Estimate average character height as 70th percentile of CC heights."""
    import cv2
    inverted = cv2.bitwise_not(line_binary)
    _, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if stats.shape[0] <= 1:
        return 20.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT].astype(float)
    return float(np.percentile(heights, 70))


def segment_chars_with_fallback(
    paw_binary: np.ndarray,
    paw_x: int,
    paw_y: int,
    ah: float,
) -> list[tuple[int, int, int, int]]:
    """Try several AH-scaled segmentations and pick the best result.

    The base `segment_chars` function uses AH for thresholds but a single
    AH value may under- or over-segment. We therefore attempt a small set
    of AH multipliers and score the outputs, preferring segmentations with
    more valid characters and fewer tiny fragments.
    """
    paw_h, paw_w = paw_binary.shape

    def score_segments(segs: list[tuple[int, int, int, int]]) -> float:
        if not segs:
            return -1.0
        # valid count
        valids = 0
        tiny = 0
        for x1, y1, x2, y2 in segs:
            crop = paw_binary[:, x1 - paw_x:x2 - paw_x]
            if _valid_char(crop, ah):
                valids += 1
            if (x2 - x1) <= max(1, int(round(0.25 * ah))):
                tiny += 1
        expected = max(1, int(round(paw_w / max(1.0, 0.9 * ah))))
        # prefer more valids, penalise tiny fraction and deviation from expected
        tiny_frac = tiny / len(segs)
        dev = abs(len(segs) - expected) / max(1, expected)
        return valids - tiny_frac * 0.5 - dev * 0.3

    candidates = []
    tried = set()
    # AH multipliers to try (centered at 1.0)
    multipliers = [1.0, 0.85, 1.15, 0.7, 1.3]
    for m in multipliers:
        scaled_ah = max(1.0, ah * m)
        key = int(round(scaled_ah))
        if key in tried:
            continue
        tried.add(key)
        try:
            segs = segment_chars(paw_binary, paw_x, paw_y, scaled_ah) if False else None
        except Exception:
            segs = None
        # We cannot call segment_chars directly (it would recurse); instead,
        # call the internal implementation by invoking the module-level
        # function body: call _segment_chars_impl using same signature.
        # For safety, replicate by calling the original logic via a helper.
        try:
            # call the original segmentation logic by temporarily binding ah
            segs = _segment_chars_impl(paw_binary, paw_x, paw_y, scaled_ah)
        except Exception:
            segs = None
        if segs is None:
            continue
        candidates.append((score_segments(segs), segs))

    if not candidates:
        return [(paw_x, paw_y, paw_x + paw_w, paw_y + paw_h)]

    # pick best-scoring segmentation
    candidates.sort(key=lambda t: t[0], reverse=True)
    best = candidates[0][1]
    return best


# To allow the fallback wrapper to call the original implementation body
# without recursion, we extract the large `segment_chars` body into
# `_segment_chars_impl` and make `segment_chars` call it. This keeps
# external API stable while enabling the fallback wrapper above.
