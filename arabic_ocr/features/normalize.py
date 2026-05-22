import cv2
import numpy as np

from arabic_ocr.config import NORM_SIZE


def normalize(char_img: np.ndarray) -> np.ndarray:
    """Resize char image proportionally and center on a NORM_SIZE×NORM_SIZE canvas.

    All downstream feature extractors expect a 32×32 grayscale input.
    """
    if char_img.size == 0:
        return np.full((NORM_SIZE, NORM_SIZE), 255, dtype=np.uint8)

    h, w = char_img.shape[:2]
    if h == 0 or w == 0:
        return np.full((NORM_SIZE, NORM_SIZE), 255, dtype=np.uint8)

    target = NORM_SIZE - 4  # 4-pixel margin
    scale = target / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(char_img, (new_w, new_h), interpolation=interp)

    canvas = np.full((NORM_SIZE, NORM_SIZE), 255, dtype=np.uint8)
    y_off = (NORM_SIZE - new_h) // 2
    x_off = (NORM_SIZE - new_w) // 2
    canvas[y_off: y_off + new_h, x_off: x_off + new_w] = resized

    return canvas
