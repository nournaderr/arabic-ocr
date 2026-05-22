"""Walk a labelled image directory and dump feature vectors to .npz.

Expected directory layout:
    data/
      raw/
        ب/  (folder name = Arabic letter)
          img_001.png
          ...
        ت/
          ...
"""
import argparse
from pathlib import Path

import cv2
import numpy as np

from arabic_ocr.features import extract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Root of labelled image tree")
    parser.add_argument("--out", required=True, help="Output .npz path")
    args = parser.parse_args()

    root = Path(args.data_dir)
    Xs, ys = [], []

    for label_dir in sorted(root.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for img_path in label_dir.glob("*.png"):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            feat = extract(img, dot_list=None)
            Xs.append(feat)
            ys.append(label)

    if not Xs:
        print("No images found.")
        return

    X = np.stack(Xs)
    y = np.array(ys)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(out), X=X, y=y)
    print(f"Saved {X.shape[0]} samples → {out}")


if __name__ == "__main__":
    main()
