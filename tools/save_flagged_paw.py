"""Save flagged PAW crops (body, dots overlay, char boxes) for manual inspection.
This script targets line 3, paw 5 from previous diagnostics by default.
Usage: python tools/save_flagged_paw.py [image_path] [line_idx] [paw_idx]
"""
import sys
from pathlib import Path
import cv2
import numpy as np

from arabic_ocr.preprocess import preprocess
from arabic_ocr.utils.image_io import load_image, resize_if_large
from arabic_ocr.segment.lines import segment_lines
from arabic_ocr.segment.paws import segment_paws
from arabic_ocr.segment.dots import separate_dots
from arabic_ocr.segment.chars import segment_chars, _estimate_ah

IMG = sys.argv[1] if len(sys.argv) > 1 else "data/test_images/arabic2.jpg"
LINE_IDX = int(sys.argv[2]) if len(sys.argv) > 2 else 3
PAW_IDX = int(sys.argv[3]) if len(sys.argv) > 3 else 5

out_dir = Path("output") / "debug_flagged"
out_dir.mkdir(parents=True, exist_ok=True)

img = load_image(IMG)
img = resize_if_large(img)
binary = preprocess(img)
lines = segment_lines(binary)

if LINE_IDX >= len(lines):
    print("Line index out of range")
    sys.exit(1)

y1, y2, line_img = lines[LINE_IDX]
ah = _estimate_ah(line_img)
paws = segment_paws(line_img, ah=ah)
if PAW_IDX >= len(paws):
    print("PAW index out of range")
    sys.exit(1)

px1, px2, paw_img = paws[PAW_IDX]
body, dots = separate_dots(paw_img, ah)
chars = segment_chars(body, px1, y1, ah)

# Save paw image
cv2.imwrite(str(out_dir / f"line{LINE_IDX}_paw{PAW_IDX}_paw.png"), paw_img)
# Save body (dots removed)
cv2.imwrite(str(out_dir / f"line{LINE_IDX}_paw{PAW_IDX}_body.png"), body)

# Dots overlay
dots_overlay = cv2.cvtColor(paw_img.copy(), cv2.COLOR_GRAY2BGR)
for d in dots:
    cx, cy = int(d.cx), int(d.cy)
    cv2.circle(dots_overlay, (cx, cy), 3, (0, 0, 255), -1)
cv2.imwrite(str(out_dir / f"line{LINE_IDX}_paw{PAW_IDX}_dots.png"), dots_overlay)

# Char boxes overlay (boxes are absolute coords: convert to local coords)
chars_overlay = cv2.cvtColor(paw_img.copy(), cv2.COLOR_GRAY2BGR)
for box in chars:
    x1, y1b, x2, y2b = box
    lx1 = x1 - px1
    lx2 = x2 - px1
    cv2.rectangle(chars_overlay, (lx1, 0), (lx2, paw_img.shape[0]-1), (0,255,0), 1)
cv2.imwrite(str(out_dir / f"line{LINE_IDX}_paw{PAW_IDX}_chars.png"), chars_overlay)

print(f"Saved flagged paw images to {out_dir}")
