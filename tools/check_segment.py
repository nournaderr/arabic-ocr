from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from arabic_ocr.utils.image_io import load_image
from arabic_ocr.preprocess import preprocess
from arabic_ocr.segment import segment

imgp = sys.argv[1] if len(sys.argv) > 1 else 'data/test_images/arabic2.jpg'
img = load_image(imgp)
bin = preprocess(img)
chars = segment(bin)
print('chars:', len(chars))
if chars:
    print('first char dims', chars[0].img.shape)
