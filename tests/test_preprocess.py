import numpy as np
import pytest

from arabic_ocr.preprocess.enhance import enhance
from arabic_ocr.preprocess.binarize import binarize
from arabic_ocr.preprocess.deskew import deskew
from arabic_ocr.preprocess.filter import filter_noise


def _white_image(h=100, w=100):
    return np.full((h, w), 255, dtype=np.uint8)


def _synthetic_page():
    img = np.full((200, 400), 255, dtype=np.uint8)
    img[40:60, 50:350] = 0   # fake text line
    img[100:120, 50:350] = 0
    return img


def test_enhance_bgr_to_gray():
    bgr = np.zeros((50, 50, 3), dtype=np.uint8)
    out = enhance(bgr)
    assert out.ndim == 2


def test_enhance_already_gray():
    gray = np.full((50, 50), 128, dtype=np.uint8)
    out = enhance(gray)
    assert out.shape == (50, 50)


def test_binarize_output_dtype():
    gray = _white_image()
    out = binarize(gray)
    assert out.dtype == np.uint8
    assert set(np.unique(out)).issubset({0, 255})


def test_deskew_no_crash_empty():
    binary = _white_image()
    corrected, angle = deskew(binary)
    assert corrected.shape == binary.shape
    assert angle == 0.0


def test_deskew_returns_same_shape():
    page = _synthetic_page()
    out, _ = deskew(page)
    assert out.shape == page.shape


def test_filter_noise_removes_nothing_on_clean():
    page = _synthetic_page()
    out = filter_noise(page)
    assert out.shape == page.shape
    assert out.dtype == np.uint8
