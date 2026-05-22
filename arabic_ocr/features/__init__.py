import numpy as np

from arabic_ocr.segment.dots import Dot
from .normalize import normalize
from .grid_density import grid_density
from .hog_features import hog_features
from .contour_features import contour_features
from .zernike_moments import zernike_features
from .dot_features import dot_features


def extract(
    char_img: np.ndarray,
    dot_list: list[Dot] | None = None,
) -> np.ndarray:
    """Concatenated feature vector for one character image (~400-d float32).

    Pipeline:
        grid_density (64) + HOG (144) + contour (192) + Zernike (~45) + dots (4)
    """
    norm = normalize(char_img)
    parts = [
        grid_density(norm),          # 64-d
        hog_features(norm),          # ~144-d
        contour_features(norm),      # 192-d
        zernike_features(norm),      # ~45-d
        dot_features(dot_list),      # 4-d
    ]
    return np.concatenate(parts).astype(np.float32)


def extract_batch(
    char_imgs: list[np.ndarray],
    dot_lists: list[list[Dot] | None] | None = None,
) -> np.ndarray:
    """Extract features for a batch of character images.

    Returns 2-D array of shape (N, feature_dim).
    """
    if dot_lists is None:
        dot_lists = [None] * len(char_imgs)
    vectors = [extract(img, dots) for img, dots in zip(char_imgs, dot_lists)]
    return np.stack(vectors, axis=0) if vectors else np.empty((0,), dtype=np.float32)


__all__ = ["extract", "extract_batch"]
