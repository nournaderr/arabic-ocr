import cv2
import numpy as np


def segment_paws(
    line_binary: np.ndarray,
    ah: float = 20.0,
) -> list[tuple[int, int, np.ndarray]]:
    """Segment a line image into PAWs (Pieces of Arabic Word).

    A PAW is a maximal horizontally connected stroke group (Amin 1998).

    Algorithm
    ---------
    Non-connecting Arabic letters (ا و ر ز د ذ ...) leave small horizontal
    gaps inside a single word.  Naively finding zero-column runs therefore
    over-segments.  We use a two-step approach instead:

    1. Dilate the text horizontally by `dil_width` pixels so that small
       intra-word gaps (≤ dil_width) are closed.  Inter-word spaces, which
       are wider, survive as zero-column runs.
    2. Find zero-column runs in the dilated image → word boundaries.
    3. Apply those boundaries to the ORIGINAL (undilated) line image to
       extract each PAW crop.

    The dilation width is 0.3 × ah, which closes intra-word gaps (typically
    1–3 pixels) while keeping inter-word spaces (typically ≥ 4 × ah / 10).

    Returns list of (x1, x2, paw_crop) sorted RIGHT-TO-LEFT (descending x1).
    """
    h, w = line_binary.shape
    dil_width = max(3, int(0.3 * ah))

    # Dilate text blobs horizontally to close intra-word gaps.
    # cv2.dilate expands bright regions, so we work on the inverted image
    # (text = white = 255) then re-invert.
    kernel = np.ones((1, dil_width), np.uint8)
    text_white = cv2.bitwise_not(line_binary)
    merged_text = cv2.dilate(text_white, kernel)
    merged = cv2.bitwise_not(merged_text)  # back to text=0, bg=255

    col_proj = np.sum(merged == 0, axis=0)
    candidates = _group_nonzero(col_proj)   # zero columns = word gaps
    if not candidates:
        return []

    result = []
    for px1, px2 in candidates:
        crop = line_binary[:, px1:px2]
        if crop.shape[1] > 0:
            result.append((px1, px2, crop))

    # Sort right-to-left (Arabic reading order)
    result.sort(key=lambda t: t[0], reverse=True)
    return result


def _group_nonzero(proj: np.ndarray) -> list[tuple[int, int]]:
    """Return (start, end) pairs for runs of consecutive non-zero values."""
    groups = []
    in_group = False
    start = 0
    for i, v in enumerate(proj):
        if v > 0 and not in_group:
            start = i
            in_group = True
        elif v == 0 and in_group:
            groups.append((start, i))
            in_group = False
    if in_group:
        groups.append((start, len(proj)))
    return groups
