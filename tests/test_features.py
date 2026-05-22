import numpy as np
import pytest

from arabic_ocr.features.normalize import normalize
from arabic_ocr.features.grid_density import grid_density
from arabic_ocr.features.hog_features import hog_features
from arabic_ocr.features.contour_features import contour_features
from arabic_ocr.features.dot_features import dot_features
from arabic_ocr.segment.dots import Dot
from arabic_ocr.features import extract


def _char_img():
    img = np.full((20, 15), 255, dtype=np.uint8)
    img[3:17, 2:13] = 0
    return img


def test_normalize_output_shape():
    out = normalize(_char_img())
    assert out.shape == (32, 32)
    assert out.dtype == np.uint8


def test_normalize_empty_image():
    out = normalize(np.array([[]], dtype=np.uint8))
    assert out.shape == (32, 32)


def test_grid_density_shape():
    norm = normalize(_char_img())
    fd = grid_density(norm)
    assert fd.shape == (64,)
    assert fd.dtype == np.float32
    assert np.all((fd >= 0) & (fd <= 1))


def test_hog_features_shape():
    norm = normalize(_char_img())
    fd = hog_features(norm)
    assert fd.ndim == 1
    assert fd.dtype == np.float32


def test_contour_features_shape():
    norm = normalize(_char_img())
    fd = contour_features(norm)
    assert fd.shape == (192,)
    assert fd.dtype == np.float32


def test_dot_features_no_dots():
    fd = dot_features(None)
    assert fd.shape == (4,)
    assert fd[2] == 0.0   # has_any_dot == 0


def test_dot_features_above_below():
    dots = [
        Dot(cx=10, cy=5,  cluster_size=2, position="above"),
        Dot(cx=15, cy=20, cluster_size=1, position="below"),
    ]
    fd = dot_features(dots)
    assert fd[0] == 2   # above
    assert fd[1] == 1   # below
    assert fd[2] == 1.0 # has_any_dot


def test_extract_returns_1d_float32():
    img = _char_img()
    fd = extract(img)
    assert fd.ndim == 1
    assert fd.dtype == np.float32
    assert len(fd) > 100
