import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Dot:
    cx: float
    cy: float
    cluster_size: int          # number of dot blobs in this cluster
    position: str              # 'above' or 'below'


def separate_dots(
    paw_binary: np.ndarray,
    ah: float,
) -> tuple[np.ndarray, list[Dot]]:
    """Separate dot components from the body of an Arabic PAW.

    Critical for Arabic: ب ت ث ن ي share the same body — only dots differ.
    Returns (body_binary, dot_list).
    """
    inverted = cv2.bitwise_not(paw_binary)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        inverted, connectivity=8
    )

    baseline_y = _estimate_baseline(stats, labels, num_labels, ah)

    body = inverted.copy()
    raw_dots: list[tuple[float, float]] = []

    # Gather statistics for adaptive thresholds
    heights = []
    widths = []
    areas = []
    for lbl in range(1, num_labels):
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        area = stats[lbl, cv2.CC_STAT_AREA]
        heights.append(h)
        widths.append(w)
        areas.append(area)

    heights = np.array(heights) if heights else np.array([ah])
    widths = np.array(widths) if widths else np.array([ah])
    areas = np.array(areas) if areas else np.array([ah * ah])

    # Adaptive thresholds based on component size distribution and ah
    # Use lower percentiles so dots are detected relative to small components
    h_thresh = max(1.0, min(0.6 * ah, float(np.percentile(heights, 25) * 1.25)))
    w_thresh = max(1.0, min(0.6 * ah, float(np.percentile(widths, 25) * 1.25)))
    area_thresh = max(1.0, min(0.2 * ah * ah, float(np.percentile(areas, 25) * 1.5)))
    aspect_min, aspect_max = 0.2, 6.0

    # Detection rule: accept components that are small relative to ah OR have small area.
    # This OR rule helps capture both tiny ink dots and slightly larger bolder dots.
    candidate_lbls = set()
    max_dot_dim = max(2.0, 0.6 * ah)
    for lbl in range(1, num_labels):
        h = stats[lbl, cv2.CC_STAT_HEIGHT]
        w = stats[lbl, cv2.CC_STAT_WIDTH]
        area = stats[lbl, cv2.CC_STAT_AREA]
        cx, cy = centroids[lbl]
        aspect = h / (w + 1e-5)
        small_dim = max(h, w) <= max_dot_dim
        small_area = area <= area_thresh
        # Require both small dimension and small area to reduce false positives from stroke fragments.
        if not (small_dim and small_area):
            continue
        if not (aspect_min <= aspect <= aspect_max):
            continue
        # Reject components that clearly touch other ink in their bbox (likely stroke fragments)
        left = int(stats[lbl, cv2.CC_STAT_LEFT])
        top = int(stats[lbl, cv2.CC_STAT_TOP])
        bw = int(stats[lbl, cv2.CC_STAT_WIDTH])
        bh = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        # extract bbox from inverted image and labels
        inv_bbox = inverted[top:top+bh, left:left+bw]
        lbl_bbox = labels[top:top+bh, left:left+bw]
        # other ink pixels within bbox that are not part of this component
        other_ink = np.any((inv_bbox > 0) & (lbl_bbox != lbl))
        if other_ink:
            continue
        candidate_lbls.add(lbl)

    for lbl in sorted(candidate_lbls):
        body[labels == lbl] = 0  # erase dot from body
        cx, cy = centroids[lbl]
        raw_dots.append((cx, cy))

    body_binary = cv2.bitwise_not(body)

    # Cluster nearby dots (within 0.5*ah)
    dot_list = _cluster_dots(raw_dots, baseline_y, cluster_radius=0.5 * ah)

    # Log if unusually many dots were found (likely over-detection)
    if len(dot_list) > 8:
        import logging
        logging.getLogger(__name__).warning("Unusually many dots detected: %d", len(dot_list))

    return body_binary, dot_list


def _estimate_baseline(stats, labels, num_labels, ah) -> float:
    """Mode of bottom edges of non-dot components."""
    bottoms = []
    for lbl in range(1, num_labels):
        h    = stats[lbl, cv2.CC_STAT_HEIGHT]
        area = stats[lbl, cv2.CC_STAT_AREA]
        top  = stats[lbl, cv2.CC_STAT_TOP]
        w    = stats[lbl, cv2.CC_STAT_WIDTH]
        aspect = h / (w + 1e-5)
        if not (h < 0.4 * ah and w < 0.4 * ah and area < 0.15 * ah * ah
                and 0.25 <= aspect <= 4.0):
            bottoms.append(top + h)
    if not bottoms:
        return float(labels.shape[0]) * 0.75
    values, counts = np.unique(bottoms, return_counts=True)
    return float(values[np.argmax(counts)])


def _cluster_dots(
    raw_dots: list[tuple[float, float]],
    baseline_y: float,
    cluster_radius: float,
) -> list[Dot]:
    """Group nearby raw dot centroids into Dot objects using union-find.

    Union-find gives correct transitive clustering: if A-B and B-C are within
    radius, all three merge even if A-C are not. The previous seed-only greedy
    loop would leave C isolated, breaking ث/ش (3-dot letters) whenever the
    outer dots are spaced beyond cluster_radius from the seed.
    """
    if not raw_dots:
        return []

    n = len(raw_dots)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if np.hypot(raw_dots[i][0] - raw_dots[j][0],
                        raw_dots[i][1] - raw_dots[j][1]) <= cluster_radius:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups: dict[int, list[tuple[float, float]]] = {}
    for i, pt in enumerate(raw_dots):
        groups.setdefault(find(i), []).append(pt)

    clusters: list[Dot] = []
    for group in groups.values():
        mean_cx = float(np.mean([p[0] for p in group]))
        mean_cy = float(np.mean([p[1] for p in group]))
        position = "above" if mean_cy < baseline_y else "below"
        clusters.append(Dot(cx=mean_cx, cy=mean_cy,
                            cluster_size=len(group), position=position))
    return clusters
