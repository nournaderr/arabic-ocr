import numpy as np
import pytest

from arabic_ocr.segment.lines import segment_lines
from arabic_ocr.segment.paws import segment_paws
from arabic_ocr.segment.dots import separate_dots
from arabic_ocr.segment.chars import _estimate_ah


def _two_line_page():
    img = np.full((200, 400), 255, dtype=np.uint8)
    img[30:60,  50:350] = 0
    img[110:140, 50:350] = 0
    return img


def _single_line():
    img = np.full((60, 300), 255, dtype=np.uint8)
    img[10:50, 20:280] = 0
    return img


def test_segment_lines_finds_two_lines():
    page = _two_line_page()
    lines = segment_lines(page)
    assert len(lines) == 2
    for y1, y2, crop in lines:
        assert y2 > y1
        assert crop.ndim == 2


def test_segment_lines_rtl_order_irrelevant():
    # Line order is top-to-bottom regardless of RTL
    page = _two_line_page()
    lines = segment_lines(page)
    assert lines[0][0] < lines[1][0]  # first line starts higher


def test_segment_paws_sorted_rtl():
    line = _single_line()
    paws = segment_paws(line)
    if len(paws) >= 2:
        # x1 should be descending (RTL)
        xs = [p[0] for p in paws]
        assert xs == sorted(xs, reverse=True)


def test_separate_dots_no_crash():
    paw = np.full((40, 60), 255, dtype=np.uint8)
    paw[10:30, 5:55] = 0   # body
    paw[5:9, 25:29]  = 0   # small dot above
    body, dots = separate_dots(paw, ah=20.0)
    assert body.shape == paw.shape


def test_estimate_ah_returns_positive():
    line = _single_line()
    ah = _estimate_ah(line)
    assert ah > 0
