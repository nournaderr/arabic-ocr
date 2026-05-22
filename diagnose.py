"""Full segmentation diagnostic."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from arabic_ocr.preprocess import preprocess
from arabic_ocr.segment.lines import segment_lines
from arabic_ocr.segment.paws import segment_paws
from arabic_ocr.segment.chars import segment_chars, _estimate_ah
from arabic_ocr.segment.dots import separate_dots
from arabic_ocr.utils.image_io import load_image, resize_if_large

IMG = sys.argv[1] if len(sys.argv) > 1 else r"data\test_images\arabic.jpg"

img = load_image(IMG)
img = resize_if_large(img)
binary = preprocess(img)
print(f"Preprocessed: {binary.shape}  black={100*np.mean(binary==0):.1f}%\n")

lines = segment_lines(binary)
print(f"Lines: {len(lines)}")
total_paws = 0
total_chars = 0
for i, (y1, y2, limg) in enumerate(lines):
    ah = _estimate_ah(limg)
    paws = segment_paws(limg, ah=ah)
    n_chars = 0
    for px1, px2, pimg in paws:
        body, dots = separate_dots(pimg, ah)
        chars = segment_chars(body, px1, y1, ah)
        n_chars += len(chars)
    total_paws  += len(paws)
    total_chars += n_chars
    print(f"  Line {i}: y={y1}-{y2}  ah={ah:.1f}  PAWs={len(paws)}  chars={n_chars}")

print(f"\nTotal  PAWs={total_paws}  chars={total_chars}")
