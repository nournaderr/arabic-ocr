from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


# ── Annotation helpers ────────────────────────────────────────────────────────

def _thickness(img: np.ndarray) -> int:
    """1px for images under 300px tall, 2px otherwise."""
    return 1 if img.shape[0] < 300 else 2


def draw_lines(
    img: np.ndarray,
    line_bounds: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Draw green horizontal bands for each (y1, y2) line boundary."""
    out = _ensure_bgr(img)
    t = _thickness(out)
    for y1, y2 in line_bounds:
        cv2.rectangle(out, (0, y1), (out.shape[1] - 1, y2), (0, 220, 0), t)
    return out


def draw_paws(
    img: np.ndarray,
    paw_boxes: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw blue boxes for each (x1, y1, x2, y2) PAW; number them RTL."""
    out = _ensure_bgr(img)
    t = _thickness(out)
    for idx, (x1, y1, x2, y2) in enumerate(paw_boxes):
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 100, 0), t)
        cv2.putText(out, str(idx), (max(0, x1 + 1), max(8, y1 + 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 0, 200), 1)
    return out


def draw_chars(
    img: np.ndarray,
    char_boxes: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    """Draw red boxes for each (x1, y1, x2, y2) character bounding box."""
    out = _ensure_bgr(img)
    t = _thickness(out)
    for x1, y1, x2, y2 in char_boxes:
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 220), t)
    return out


def draw_dots(
    img: np.ndarray,
    dot_list: list,
) -> np.ndarray:
    """Draw yellow circles at each detected dot centroid."""
    out = _ensure_bgr(img)
    for dot in dot_list:
        cx, cy = int(dot.cx), int(dot.cy)
        cv2.circle(out, (cx, cy), max(2, out.shape[0] // 60), (0, 220, 220), -1)
    return out


def save_debug_visualization(
    img: np.ndarray,
    stage_name: str,
    output_dir: str | Path,
) -> None:
    """Write an annotated image to output_dir/<stage_name>.png."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_dir / f"{stage_name}.png"), img)


# ── Internal ──────────────────────────────────────────────────────────────────

def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    return img.copy()
