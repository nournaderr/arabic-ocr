import numpy as np

from arabic_ocr.config import NORM_SIZE


def zernike_features(norm_img: np.ndarray, degree: int = 8) -> np.ndarray:
    """Zernike moment descriptors up to the given degree (~45 features at degree=8).

    Rotation-invariant shape descriptors.
    Paper: Khotanzad & Hong, IEEE TPAMI 1990.
    Arabic relevance: discriminates rotationally similar letters (و vs ر).

    Uses mahotas if available, falls back to a manual implementation.
    """
    try:
        import mahotas
        radius = NORM_SIZE // 2
        # mahotas.features.zernike_moments expects a binary image and works on
        # the magnitude of the moments
        feats = mahotas.features.zernike_moments(
            (norm_img < 128).astype(np.uint8), radius=radius, degree=degree
        )
        return feats.astype(np.float32)
    except ImportError:
        return _zernike_manual(norm_img, degree)


# ── Manual Zernike implementation (no mahotas dependency) ───────────────────

def _zernike_manual(img: np.ndarray, degree: int) -> np.ndarray:
    """Pure-NumPy Zernike moments (magnitude only)."""
    binary = (img < 128).astype(float)
    h, w = binary.shape

    # Map pixel coordinates to unit disk
    y_idx, x_idx = np.indices((h, w))
    x_norm = (x_idx - w / 2.0) / (w / 2.0)
    y_norm = (y_idx - h / 2.0) / (h / 2.0)
    rho   = np.hypot(x_norm, y_norm)
    theta = np.arctan2(y_norm, x_norm)

    inside = rho <= 1.0
    moments: list[float] = []

    for n in range(degree + 1):
        for m in range(-n, n + 1):
            if (n - abs(m)) % 2 != 0:
                continue
            R = _radial_poly(n, abs(m), rho)
            V = R * np.exp(1j * m * theta)
            Z = np.sum(binary[inside] * np.conj(V[inside]))
            Z *= (n + 1) / np.pi
            moments.append(abs(Z))

    return np.array(moments, dtype=np.float32)


def _radial_poly(n: int, m: int, rho: np.ndarray) -> np.ndarray:
    """Zernike radial polynomial R_n^m(rho)."""
    result = np.zeros_like(rho)
    for s in range((n - m) // 2 + 1):
        coeff = (
            ((-1) ** s)
            * _factorial(n - s)
            / (
                _factorial(s)
                * _factorial((n + m) // 2 - s)
                * _factorial((n - m) // 2 - s)
            )
        )
        result += coeff * rho ** (n - 2 * s)
    return result


def _factorial(n: int) -> int:
    if n <= 1:
        return 1
    r = 1
    for i in range(2, n + 1):
        r *= i
    return r
