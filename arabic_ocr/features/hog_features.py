import numpy as np
from skimage.feature import hog


def hog_features(norm_img: np.ndarray) -> np.ndarray:
    """HOG descriptor for a 32×32 character image.

    Parameters match Dalal & Triggs CVPR 2005:
      orientations=9, pixels_per_cell=(4,4), cells_per_block=(2,2), L2-Hys.
    Critical for Arabic: captures local stroke direction differences
    (e.g. ر vs ز which differ only in a dot and stroke curvature).
    """
    fd = hog(
        norm_img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        visualize=False,
        feature_vector=True,
    )
    return fd.astype(np.float32)
