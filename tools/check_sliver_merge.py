from arabic_ocr.segment.chars import segment_chars
import numpy as np

paw = np.full((40,80),255,dtype=np.uint8)
paw[5:35,5:75]=0

# monkeypatch-like replacement by temporarily assigning
import arabic_ocr.segment.chars as ch
orig = ch._best_segmentation

def _fake_best(paw_binary, cuts, ah, paw_h=None, paw_w=None):
    w = paw_binary.shape[1]
    return [0, 10, 11, w]

ch._best_segmentation = _fake_best
boxes = segment_chars(paw, paw_x=0, paw_y=0, ah=20.0)
print('boxes:', boxes)
ch._best_segmentation = orig
