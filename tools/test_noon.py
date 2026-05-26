import cv2
import numpy as np
import sys
sys.path.insert(0, ".")
from arabic_ocr.segment.chars import segment_chars, _estimate_ah
from arabic_ocr.utils.image_io import load_image
from PIL import Image, ImageDraw, ImageFont

img_pil = Image.new("L", (200, 64), 255)
draw = ImageDraw.Draw(img_pil)
font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
draw.text((10, 10), "نحن", font=font, fill=0)
binary = (np.array(img_pil) < 128).astype(np.uint8) * 255  # 255 is ink
binary_01 = (binary == 255).astype(np.uint8) # 1 is ink
binary_inv = (binary == 255)

# in chars.SegmentChars_impl: paw_binary=0 is ink. Thus we need paw_binary where 0 is ink
paw_binary = np.where(binary == 255, 0, 255).astype(np.uint8)

# Calculate AH
ah = _estimate_ah(binary)
print("AH:", ah)

# Projection
col_proj = np.sum(paw_binary == 0, axis=0).astype(float)
import matplotlib.pyplot as plt
plt.subplot(2, 1, 1)
plt.imshow(paw_binary, cmap='gray')
plt.subplot(2, 1, 2)
plt.plot(col_proj)
plt.savefig("noon_proj.png")

# Segment
boxes = segment_chars(paw_binary, 0, 0, ah)
print("Boxes (n_expected=1):", len(boxes))
print("Cuts:", boxes)

