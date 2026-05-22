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

        is_dot = (
            h   < 0.4 * ah and
            w   < 0.4 * ah and
            area < 0.15 * ah * ah
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
        if not (h < 0.4 * ah and w < 0.4 * ah and area < 0.15 * ah * ah):
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
    """Group nearby raw dot centroids into Dot objects."""
    if not raw_dots:
        return []

    used = [False] * len(raw_dots)
    clusters: list[Dot] = []

    for i, (cx_i, cy_i) in enumerate(raw_dots):
        if used[i]:
            continue
        group = [(cx_i, cy_i)]
        used[i] = True
        for j, (cx_j, cy_j) in enumerate(raw_dots):
            if used[j]:
                continue
            dist = np.hypot(cx_i - cx_j, cy_i - cy_j)
            if dist <= cluster_radius:
                group.append((cx_j, cy_j))
                used[j] = True

        mean_cx = float(np.mean([p[0] for p in group]))
        mean_cy = float(np.mean([p[1] for p in group]))
        position = "above" if mean_cy < baseline_y else "below"
        clusters.append(Dot(cx=mean_cx, cy=mean_cy,
                            cluster_size=len(group), position=position))

    return clusters
