import numpy as np

from arabic_ocr.config import GRID_ROWS, GRID_COLS, NORM_SIZE


def grid_density(norm_img: np.ndarray) -> np.ndarray:
    """64-d feature: fraction of black pixels per cell in an 8×8 grid.

    Reference: Cheriet et al. 2007 — standard grid-based handwriting feature.
    Input must be 32×32 grayscale (white background, black text).
    """
    assert norm_img.shape == (NORM_SIZE, NORM_SIZE), (
        f"Expected {NORM_SIZE}×{NORM_SIZE}, got {norm_img.shape}"
    )

    cell_h = NORM_SIZE // GRID_ROWS  # 4 pixels
    cell_w = NORM_SIZE // GRID_COLS  # 4 pixels
    cell_area = cell_h * cell_w

    features = np.empty(GRID_ROWS * GRID_COLS, dtype=np.float32)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            cell = norm_img[
                r * cell_h: (r + 1) * cell_h,
                c * cell_w: (c + 1) * cell_w,
            ]
            features[r * GRID_COLS + c] = float(np.sum(cell == 0)) / cell_area

    return features
