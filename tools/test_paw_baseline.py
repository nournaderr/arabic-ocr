import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt
import os
sys.path.insert(0, ".")
from arabic_ocr.utils.image_io import load_image
from arabic_ocr.preprocess import preprocess
from arabic_ocr.segment.lines import segment_lines
from arabic_ocr.segment.paws import segment_paws

os.makedirs("output/debug", exist_ok=True)
img = load_image("data/test_images/arabic2.jpg")
binary = preprocess(img)
lines = segment_lines(binary)

count = 0
for l_idx, line in enumerate(lines[:3]):
    paws = segment_paws(line[2])
    for p_idx, paw in enumerate(paws):
        paw_bin = paw[2]
        col_proj = np.sum(paw_bin == 0, axis=0)
        row_proj = np.sum(paw_bin == 0, axis=1)
        baseline = np.argmax(row_proj)
        
        if paw_bin.shape[0] > 30 and paw_bin.shape[1] > 30:
            plt.figure(figsize=(10,4))
            plt.subplot(131)
            plt.imshow(paw_bin, cmap='gray')
            plt.axhline(baseline, color='r')
            
            plt.subplot(132)
            plt.plot(col_proj)
            plt.title("Col Proj")
            
            plt.subplot(133)
            y_coords = np.arange(paw_bin.shape[0])
            centers = []
            for col in range(paw_bin.shape[1]):
                col_ink = paw_bin[:, col] == 0
                if np.any(col_ink):
                    center = np.average(y_coords[col_ink])
                else:
                    center = -1
                centers.append(center)
            plt.plot(centers, label='Ink Center')
            plt.axhline(baseline, color='r')
            plt.ylim(paw_bin.shape[0], 0)
            plt.title("Center of Ink")
            
            plt.savefig(f"output/debug/paw_{l_idx}_{p_idx}.png")
            plt.close()
            count += 1
            if count >= 30:
                break
    if count >= 30:
        break
