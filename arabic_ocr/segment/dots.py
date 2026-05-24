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

    for lbl in range(1, num_labels):
        h   = stats[lbl, cv2.CC_STAT_HEIGHT]
        w   = stats[lbl, cv2.CC_STAT_WIDTH]
        area = stats[lbl, cv2.CC_STAT_AREA]
        cx, cy = centroids[lbl]

        aspect = h / (w + 1e-5)
        is_dot = (
            h    < 0.4 * ah and
            w    < 0.4 * ah and
            area < 0.15 * ah * ah and
            0.25 <= aspect <= 4.0   # reject thin ligature slivers
        )
        if is_dot:
            body[labels == lbl] = 0  # erase dot from body
            raw_dots.append((cx, cy))

    body_binary = cv2.bitwise_not(body)

    # Cluster nearby dots (within 0.5*ah)
    dot_list = _cluster_dots(raw_dots, baseline_y, cluster_radius=0.5 * ah)

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
