"""Per-PAW diagnostics: prints chars/dots per PAW and flags suspicious ones.
Usage: python tools/paw_diagnostics.py data/test_images/arabic2.jpg
"""
import sys
from pathlib import Path
import numpy as np

from arabic_ocr.preprocess import preprocess
from arabic_ocr.utils.image_io import load_image, resize_if_large
from arabic_ocr.segment.lines import segment_lines
from arabic_ocr.segment.paws import segment_paws
from arabic_ocr.segment.dots import separate_dots
from arabic_ocr.segment.chars import segment_chars, _estimate_ah
from arabic_ocr.config import MIN_CHAR_WIDTH

IMG = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/arabic2.jpg"
img = load_image(IMG)
img = resize_if_large(img)
binary = preprocess(img)

lines = segment_lines(binary)

print(f"Preprocessed: {binary.shape}  black={100*np.mean(binary==0):.1f}%\n")
print(f"Found {len(lines)} lines")

flags = []
total_paws = 0

for line_idx, (y1, y2, line_img) in enumerate(lines):
    ah = _estimate_ah(line_img)
    paws = segment_paws(line_img, ah=ah)
    print(f"Line {line_idx}: y={y1}-{y2} ah={ah:.1f} PAWs={len(paws)}")
    for paw_idx, (px1, px2, paw_img) in enumerate(paws):
        body, dots = separate_dots(paw_img, ah)
        chars = segment_chars(body, px1, y1, ah)
        paw_w = px2 - px1
        num_chars = len(chars)
        num_dots = sum(d.cluster_size for d in dots)
        avg_char_w = np.mean([c[2]-c[0] for c in chars]) if chars else paw_w

        issues = []
        # too many chars for PAW width (very small average width)
        if avg_char_w < (MIN_CHAR_WIDTH * ah * 0.8):
            issues.append(f"tiny_avg_w={avg_char_w:.1f}")
        # many characters relative to PAW width
        if num_chars > max(1, paw_w / max(1.0, 0.4 * ah) + 3):
            issues.append(f"high_char_count={num_chars}")
        # many dots (possible over-detection)
        if num_dots > max(3, num_chars // 2):
            issues.append(f"many_dots={num_dots}")

        print(f"  PAW {paw_idx}: x={px1}-{px2} w={paw_w} chars={num_chars} dots={num_dots} avg_char_w={avg_char_w:.1f} issues={issues}")
        if issues:
            flags.append((line_idx, paw_idx, px1, px2, paw_w, num_chars, num_dots, avg_char_w, issues))
        total_paws += 1

print('\nFlagged PAWs:')
for f in flags:
    print(f)
print(f"\nTotal PAWs={total_paws}")
print('Done')
