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


def test_segment_paws_falls_back_to_full_line_when_gap_detection_fails():
    line = np.full((50, 120), 255, dtype=np.uint8)
    line[10:40, :] = 0
    paws = segment_paws(line, ah=20.0)
    assert len(paws) == 1
    x1, x2, crop = paws[0]
    assert (x1, x2) == (0, line.shape[1])
    assert crop.shape == line.shape


def test_segment_chars_handles_dp_exception(monkeypatch):
    # Create a PAW image with simple body and force _best_segmentation to raise
    import numpy as np
    from arabic_ocr.segment.chars import segment_chars

    paw = np.full((40, 80), 255, dtype=np.uint8)
    paw[5:35, 5:75] = 0

    # Monkeypatch _best_segmentation to raise
    import arabic_ocr.segment.chars as ch

    def _bad_best(*args, **kwargs):
        raise RuntimeError("DP crashed")

    monkeypatch.setattr(ch, "_best_segmentation", _bad_best)
    boxes = segment_chars(paw, paw_x=10, paw_y=20, ah=20.0)
    assert len(boxes) == 1
    x1, y1, x2, y2 = boxes[0]
    assert x1 == 10 and x2 == 10 + paw.shape[1]


def test_separate_dots_detects_multi_scale():
    import numpy as np
    from arabic_ocr.segment.dots import separate_dots

    paw = np.full((60, 120), 255, dtype=np.uint8)
    # big body
    paw[20:50, 10:110] = 0
    # small dot above
    paw[10:13, 30:33] = 0
    # larger dot above (simulating a bolder font)
    paw[8:16, 70:78] = 0

    body, dots = separate_dots(paw, ah=20.0)
    # Expect at least two dot clusters detected
    assert isinstance(dots, list)
    assert len(dots) >= 2


def test_segment_chars_merges_tiny_sliver(monkeypatch):
    import numpy as np
    from arabic_ocr.segment.chars import segment_chars

    paw = np.full((40, 80), 255, dtype=np.uint8)
    paw[5:35, 5:75] = 0

    # Force best_cuts that produce a tiny sliver at position 10..11
    def _fake_best(paw_binary, cuts, ah, paw_h=None, paw_w=None):
        w = paw_binary.shape[1]
        return [0, 10, 11, w]

    import arabic_ocr.segment.chars as ch
    monkeypatch.setattr(ch, "_best_segmentation", _fake_best)

    boxes = segment_chars(paw, paw_x=0, paw_y=0, ah=20.0)
    # Ensure no sliver remains (width <= sliver_thresh should be merged)
    for x1, y1, x2, y2 in boxes:
        assert (x2 - x1) > 1


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
