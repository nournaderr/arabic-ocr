"""Run OCR on one image and print per-character candidates plus final transcription.
Usage: python tools/inspect_ocr.py data/test_images/arabic2.jpg --classifier cnn
"""
import argparse
from pathlib import Path

import numpy as np

from arabic_ocr.utils.image_io import load_image, resize_if_large
from arabic_ocr.pipeline import ArabicOCRPipeline
from arabic_ocr.segment import segment, CharCrop
from arabic_ocr.classifiers import get_classifier
from arabic_ocr.postprocess import postprocess, ArabicLanguageModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--classifier', default='svm', choices=['svm','rf','cnn'])
    args = parser.parse_args()

    img = load_image(args.image)
    img = resize_if_large(img)

    pipe = ArabicOCRPipeline(classifier=args.classifier, debug=False)
    # segmentation will be done on preprocessed binary below; skip early call

    # Simpler: use public API to get final text and separately re-segment using pipeline.segment
    text = pipe.run(args.image)
    print('\nFinal transcription:')
    print(text)

    # Re-segment using pipeline.segment on preprocessed binary
    from arabic_ocr.preprocess import preprocess
    binary = preprocess(load_image(args.image))
    char_crops = segment(binary)

    # Classify
    imgs = [c.img for c in char_crops]
    dot_lists = [c.dots for c in char_crops]
    classifier = get_classifier(args.classifier)
    try:
        all_candidates = classifier.predict_batch(imgs, dot_lists)
    except Exception as e:
        print('Classifier failed:', e)
        all_candidates = [[('', 1.0)] for _ in imgs]

    # Print per-char info
    from arabic_ocr.utils.visualize import save_debug_visualization
    for c, cands in zip(char_crops, all_candidates):
        print(f"Line {c.line_idx} PAW {c.paw_idx} CHAR {c.char_idx} POS {c.position} ABS ({c.abs_x},{c.abs_y}) DOTS {sum(d.cluster_size for d in c.dots)}")
        for label, score in cands[:10]:
            print(f"  {label:30} {score:.4f}")
        print('')

if __name__ == '__main__':
    main()
