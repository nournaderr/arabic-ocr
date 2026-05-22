import cv2
import numpy as np

from arabic_ocr.config import SKEW_ANGLE_MAX


def deskew(binary: np.ndarray) -> tuple[np.ndarray, float]:
    """Rotate the binary image to correct text skew.

    Two-method strategy:
      1. Projection profile — finds the rotation angle that maximises the
         variance of horizontal ink projections.  Reliable for pages with
         multiple text lines and unaffected by dark scanner borders.
      2. Min-area bounding box fallback — used only when the projection
         method cannot find a confident angle (sparse text, single line).

    Dark scanner borders distort the bounding-box approach because the border
    pixels pull the foreground cloud toward a ±45° rectangle.  The projection
    method works on the interior text and is immune to this artefact.

    Returns (corrected_image, angle_degrees).
    """
    h, w = binary.shape

    # Strip outer 2% of the image to exclude scanner border artefacts before
    # falling back to the bounding-box method.
    margin_h = max(1, int(h * 0.02))
    margin_w = max(1, int(w * 0.02))
    interior = binary[margin_h: h - margin_h, margin_w: w - margin_w]

    angle = _projection_skew(interior)
    if angle is None:
        angle = _bbox_skew(interior)
    if angle is None:
        return binary.copy(), 0.0

    center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        binary, M, (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return rotated, float(angle)


def _projection_skew(binary: np.ndarray, angle_range: float = 15.0) -> float | None:
    """Estimate skew by maximising horizontal-projection variance.

    Sweeps angles in [-angle_range, +angle_range] in 0.5° steps and returns
    the angle whose rotated image has the highest row-projection variance
    (well-aligned text has sharp alternating dark/light rows).

    Returns None when the winning variance is not significantly better than the
    0° baseline (image is already straight, or too sparse to judge).
    """
    h, w = binary.shape
    if np.sum(binary == 0) < 50:
        return None

    center = (w / 2, h / 2)
    best_angle = 0.0
    best_var = _row_proj_var(binary)
    baseline_var = best_var

    for deg_10 in range(int(-angle_range * 10), int(angle_range * 10) + 1, 5):
        angle = deg_10 / 10.0
        if angle == 0.0:
            continue
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            binary, M, (w, h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )
        var = _row_proj_var(rotated)
        if var > best_var:
            best_var = var
            best_angle = angle

    # Only apply correction when variance improvement is meaningful (>5%)
    if best_var < baseline_var * 1.05:
        return 0.0
    if abs(best_angle) > SKEW_ANGLE_MAX:
        return None
    return best_angle


def _row_proj_var(binary: np.ndarray) -> float:
    proj = np.sum(binary == 0, axis=1).astype(float)
    return float(np.var(proj))


def _bbox_skew(binary: np.ndarray) -> float | None:
    """Min-area bounding-box skew estimate (original method, border-stripped)."""
    coords = np.column_stack(np.where(binary == 0))
    if len(coords) < 10:
        return None

    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle += 90
    if abs(angle) > SKEW_ANGLE_MAX:
        return None
    return float(angle)
